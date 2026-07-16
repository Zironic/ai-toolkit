"""Controlled two-process CUDA probe for torch.compile Mega-Cache.

This script isolates Mega-Cache from TorchInductor's ordinary persistent disk
cache. Run it in four fresh processes:

1. produce into an empty cache directory (cold compile + artifact save),
2. consume without loading from a different empty directory (cold control),
3. consume without loading while reusing the producer directory (disk cache),
4. consume after loading the artifact into a third empty directory (Mega-Cache).

The probe uses a small BF16 training graph and records AOTAutograd/FXGraph
cache counters, serialized artifact inventory, compile timing, and forward and
gradient checksums. It never compiles a CPU graph.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.megacache_diagnostics import (  # noqa: E402
    cache_dir_stats,
    cache_evidence,
    counter_snapshot,
    install_backend_instrumentation,
)
from toolkit.compile_cache import (  # noqa: E402
    compile_cache_artifact_counts,
    load_compile_cache_artifact,
    save_compile_cache_artifact,
    windows_triton_bundle_fallback_stats,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("produce", "consume"), required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--expected-arm",
        choices=("cold", "empty-control", "shared-disk", "megacache"),
        required=True,
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Do not load the artifact in consume mode (control arms only).",
    )
    parser.add_argument(
        "--allow-populated-cache",
        action="store_true",
        help="Permit a non-empty cache dir (required by the shared-disk arm).",
    )
    parser.add_argument(
        "--trace-dir",
        help="Optional empty/new directory for TORCH_TRACE output.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--graph-kind",
        choices=(
            "plain",
            "mm-wrapper",
            "mm-wrapper-fullgraph",
            "mm-wrapper-outer-fullgraph",
            "arena-dispatcher-fullgraph",
        ),
        default="plain",
        help=(
            "plain uses one fullgraph training function; mm-wrapper compiles a "
            "block with two active LinearLayerMemoryManager forwards and "
            "fullgraph=False; mm-wrapper-fullgraph keeps those forwards eager "
            "around a tensor-only fullgraph=True core; "
            "mm-wrapper-outer-fullgraph is the expected-failure diagnostic."
            " arena-dispatcher-fullgraph uses the production arena runtime's "
            "strict pure block kernels."
        ),
    )
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--no-gpu-lock", action="store_true")
    parser.add_argument("--ignore-contention", action="store_true")
    return parser.parse_args()


def _prepare_environment(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir).resolve()
    if cache_dir.exists() and any(cache_dir.iterdir()) and not args.allow_populated_cache:
        raise SystemExit(f"cache directory is not empty: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    # Keep compiler work in this process so the diagnostic monkeypatches below
    # observe real Triton compiler and autotuner calls.
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

    if args.trace_dir:
        trace_dir = Path(args.trace_dir).resolve()
        trace_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_TRACE"] = str(trace_dir)


def _manifest(torch, args: argparse.Namespace) -> dict:
    import torch._functorch.config as functorch_config
    import torch._inductor.config as inductor_config

    try:
        import triton

        triton_version = triton.__version__
    except Exception:
        triton_version = None

    device = torch.device(args.device)
    manifest = {
        "schema": "torch_megacache_cuda_probe.v1",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton_version,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "compile": {
            "backend": "inductor",
            "mode": "default",
            "fullgraph": args.graph_kind != "mm-wrapper",
            "dynamic": False,
        },
        "configuration": {
            "aot_autograd_cache": bool(functorch_config.enable_autograd_cache),
            "fx_graph_cache": bool(inductor_config.fx_graph_cache),
            "autotune_local_cache": bool(inductor_config.autotune_local_cache),
            "coordinate_descent_tuning": bool(
                inductor_config.coordinate_descent_tuning
            ),
            "coordinate_descent_check_all_directions": bool(
                inductor_config.coordinate_descent_check_all_directions
            ),
            "strict_autograd_cache": bool(functorch_config.strict_autograd_cache),
            "compile_threads": int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"]),
        },
        "graph": {
            "kind": args.graph_kind,
            "dtype": "bfloat16",
            "x_shape": [64, 64],
            "weight_shape": [64, 64],
            "bias_shape": [64],
            "seed": int(args.seed),
        },
    }
    if args.graph_kind.startswith("mm-wrapper"):
        manifest["graph"]["managed_linear_layers"] = 2
        manifest["graph"]["pinned_weight_gib"] = 0.0
        manifest["graph"]["compile_boundary"] = (
            "tensor_core" if args.graph_kind == "mm-wrapper-fullgraph" else "block"
        )
    elif args.graph_kind == "arena-dispatcher-fullgraph":
        manifest["graph"].update(
            {
                "managed_blocks": 3,
                "compile_boundary": "arena_functional_block",
                "checkpoint_keep_last": 1,
                "x_shape": [2, 4, 32],
                "weight_shape": [32, 32],
                "bias_shape": [32],
            }
        )
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    manifest["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def _validate_arm(args: argparse.Namespace, evidence: dict[str, int | bool]) -> None:
    if evidence["aot_bypass"] or evidence["fx_bypass"]:
        raise RuntimeError(f"cache bypass observed: {evidence}")
    if evidence["graph_breaks"]:
        raise RuntimeError(f"internal graph breaks observed: {evidence}")

    cold_arm = args.expected_arm in ("cold", "empty-control")
    if cold_arm:
        if not evidence["aot_miss"] or not evidence["fx_miss"]:
            raise RuntimeError(f"cold arm did not compile from cache misses: {evidence}")
        return

    if not evidence["aot_hit"] or not evidence["fx_hit"]:
        raise RuntimeError(f"warm arm did not hit AOT and FX caches: {evidence}")
    if (
        evidence["backend_codegen_observed"]
        or evidence["triton_compile_calls"]
        or evidence["autotune_benchmark_calls"]
        or evidence["coordinate_descent_calls"]
    ):
        raise RuntimeError(f"warm arm performed compiler work: {evidence}")


def _write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="ascii")
    os.replace(tmp_path, path)


def _build_plain_workload(torch, args: argparse.Namespace, device):
    x = torch.randn(64, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        64, 64, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    bias = torch.randn(64, device=device, dtype=torch.bfloat16, requires_grad=True)
    return {
        "x": x,
        "weight": weight,
        "bias": bias,
        "wrapper_state": None,
    }


def _build_mm_wrapper_workload(torch, args: argparse.Namespace, device):
    import torch.nn as nn

    from toolkit.memory_management.manager import MemoryManager
    from toolkit.memory_management import manager_modules
    from toolkit.memory_management.manager_modules import LinearLayerMemoryManager

    class ManagedBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(64, 64, bias=True, dtype=torch.bfloat16)
            self.out_proj = nn.Linear(64, 64, bias=True, dtype=torch.bfloat16)
            self.scale = nn.Parameter(
                torch.randn(64, device=device, dtype=torch.bfloat16)
            )
            for linear in (self.in_proj, self.out_proj):
                for parameter in linear.parameters():
                    parameter.requires_grad_(False)

        def core(self, x):
            x = torch.nn.functional.gelu(x)
            return x * torch.sigmoid(self.scale)

        def forward(self, x):
            x = x * self.scale
            x = self.in_proj(x)
            x = self.core(x)
            x = self.out_proj(x)
            return x

    block = ManagedBlock()
    manager = MemoryManager(block, device, pinned_weight_gib=0.0)
    managed_layers = (block.in_proj, block.out_proj)
    for index, linear in enumerate(managed_layers):
        linear._mm_layer_key = f"managed.{index}"
        LinearLayerMemoryManager.attach(linear, manager)

    wrapper_calls = {"forward_stage": 0, "backward_stage": 0}
    original_forward_stage = manager_modules._stage_forward_weight
    original_backward_stage = manager_modules._stage_backward_weight

    @functools.wraps(original_forward_stage)
    def counted_forward_stage(*args, **kwargs):
        wrapper_calls["forward_stage"] += 1
        return original_forward_stage(*args, **kwargs)

    @functools.wraps(original_backward_stage)
    def counted_backward_stage(*args, **kwargs):
        wrapper_calls["backward_stage"] += 1
        return original_backward_stage(*args, **kwargs)

    manager_modules._stage_forward_weight = counted_forward_stage
    manager_modules._stage_backward_weight = counted_backward_stage

    attached = sum(
        1 for linear in managed_layers if hasattr(linear, "_layer_memory_manager")
    )
    hijacked = sum(1 for linear in managed_layers if "forward" in linear.__dict__)
    cpu_weights = sum(
        1 for linear in managed_layers if linear.weight.device.type == "cpu"
    )
    if (attached, hijacked, cpu_weights) != (2, 2, 2):
        raise RuntimeError(
            "MM wrapper setup failed: "
            f"attached={attached} hijacked={hijacked} cpu_weights={cpu_weights}"
        )

    x = torch.randn(64, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    return {
        "x": x,
        "block": block,
        "managed_layers": managed_layers,
        "wrapper_calls": wrapper_calls,
        "wrapper_state": {
            "attached_layers": attached,
            "hijacked_forwards": hijacked,
            "cpu_weight_layers": cpu_weights,
        },
    }


def _build_arena_dispatcher_workload(torch, args: argparse.Namespace, device):
    from dataclasses import replace
    from types import MethodType
    from unittest import mock

    from torch.utils.checkpoint import checkpoint

    from toolkit.memory_management.arena_offload import (
        ArenaOffloadConfig,
        prepare_arena_offload,
    )

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(32, 32, dtype=torch.bfloat16)

        def forward(self, value):
            return torch.nn.functional.silu(self.proj(value))

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(Block() for _ in range(3))
            self.gradient_checkpointing = True
            self._checkpoint_keep_last = 1

        def forward(self, value):
            for index, block in enumerate(self.blocks):
                if torch.is_grad_enabled() and index < 2:
                    value = checkpoint(block, value, use_reentrant=False)
                else:
                    value = block(value)
            return value

    model = Transformer().to(device)
    model.requires_grad_(False)
    config = ArenaOffloadConfig(
        enabled=True,
        compile_blocks=True,
        _compile_dynamic=False,
        _compile_fullgraph=True,
    )
    config = replace(
        config,
        _policy=replace(
            config._policy,
            working_reserve_gib=0.0,
            physical_vram_headroom_gib=0.0,
            wddm_hard_gib=1.0,
            checkpoint_keep_last=1,
            prefetch_depth=1,
        ),
    )
    with mock.patch(
        "toolkit.memory_management.arena_offload.planner.vram_budget.device_mem_info",
        return_value=(3_000, 12 * 1024**3),
    ):
        runtime = prepare_arena_offload(
            model,
            device=device,
            block_names=("blocks",),
            config=config,
        )

    adapters = []
    for block in model.blocks:
        saved = block.forward
        block.adapter_gain = torch.nn.Parameter(
            torch.zeros((), device=device, dtype=torch.float32)
        )

        def installed_forward(self, value, _saved=saved):
            return _saved(value) + self.adapter_gain.to(value.dtype) * value

        block.forward = MethodType(installed_forward, block)
        adapters.append(block.adapter_gain)
    runtime.finalize()
    diagnostics = runtime.diagnostics()
    if not diagnostics["compile_fullgraph"]:
        raise RuntimeError("arena runtime did not preserve strict fullgraph mode")
    if not diagnostics["accounting"]["mixed_residency"]:
        raise RuntimeError(
            f"arena probe did not establish mixed residency: {diagnostics['accounting']}"
        )
    x = torch.randn(
        2, 4, 32, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    return {
        "x": x,
        "model": model,
        "runtime": runtime,
        "adapters": adapters,
        "wrapper_state": {
            "backend": "arena",
            "managed_blocks": runtime.block_count,
            "compile_fullgraph": diagnostics["compile_fullgraph"],
            "mixed_residency": diagnostics["accounting"]["mixed_residency"],
            "planned_training_h2d_bytes": diagnostics["accounting"][
                "planned_training_h2d_bytes"
            ],
        },
    }


def _main(args: argparse.Namespace) -> int:
    import torch
    import torch._functorch.config as functorch_config
    import torch._inductor.config as inductor_config

    from scripts.smoke_runtime import fail_if_vram_contended

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise SystemExit("this probe requires a CUDA device")
    fail_if_vram_contended(
        args.device,
        ignore_contention=bool(args.ignore_contention),
    )

    inductor_config.coordinate_descent_tuning = False
    inductor_config.coordinate_descent_check_all_directions = False
    functorch_config.strict_autograd_cache = True
    torch._dynamo.utils.counters.clear()
    backend_calls = install_backend_instrumentation()

    artifact_path = Path(args.artifact).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    result_path = Path(args.result).resolve()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.graph_kind == "plain":
        workload = _build_plain_workload(torch, args, device)
    elif args.graph_kind == "arena-dispatcher-fullgraph":
        workload = _build_arena_dispatcher_workload(torch, args, device)
    else:
        workload = _build_mm_wrapper_workload(torch, args, device)

    # For the MM arms, the real CPU-weight streaming forwards are installed
    # before Mega-Cache population, matching the requested lifecycle.
    disk_before = cache_dir_stats(cache_dir)

    load_info = None
    load_seconds = 0.0
    artifact_bytes = None
    if args.mode == "consume" and not args.skip_load:
        if not artifact_path.is_file():
            raise SystemExit(f"artifact does not exist: {artifact_path}")
        started = time.perf_counter()
        loaded = load_compile_cache_artifact(artifact_path)
        load_seconds = time.perf_counter() - started
        if loaded is None:
            raise RuntimeError("torch rejected the serialized cache artifact")
        load_info = loaded.info
        artifact_bytes = artifact_path.read_bytes()
    elif args.mode == "produce" and args.skip_load:
        raise SystemExit("--skip-load is only meaningful in consume mode")

    disk_after_load = cache_dir_stats(cache_dir)

    if args.graph_kind == "plain":

        @torch.compile(
            backend="inductor",
            mode="default",
            fullgraph=True,
            dynamic=False,
        )
        def training_graph(x, weight, bias):
            hidden = torch.nn.functional.gelu(x @ weight + bias)
            loss = hidden.float().square().mean()
            return hidden, loss

        def execute_workload():
            return training_graph(
                workload["x"], workload["weight"], workload["bias"]
            )

    elif args.graph_kind != "arena-dispatcher-fullgraph":
        block = workload["block"]
        if args.graph_kind == "mm-wrapper-fullgraph":
            block.core = torch.compile(
                block.core,
                backend="inductor",
                mode="default",
                fullgraph=True,
                dynamic=False,
            )
            executable = block
        else:
            executable = torch.compile(
                block,
                backend="inductor",
                mode="default",
                fullgraph=args.graph_kind == "mm-wrapper-outer-fullgraph",
                dynamic=False,
            )

        def execute_workload():
            hidden = executable(workload["x"])
            return hidden, hidden.float().square().mean()

    else:

        def execute_workload():
            hidden = workload["model"](workload["x"])
            return hidden, hidden.float().square().mean()

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    arena_transfer_before = None
    if args.graph_kind == "arena-dispatcher-fullgraph":
        from toolkit.memory_management.arena_offload import transfer

        arena_transfer_before = transfer.lifetime_fetch_stats()["bytes"]
        with workload["runtime"].training_step(shape_key=(2, 4, 32), step_num=1):
            output, loss = execute_workload()
            loss.backward()
    else:
        output, loss = execute_workload()
        loss.backward()
    torch.cuda.synchronize(device)
    execute_seconds = time.perf_counter() - started

    checksums = {
        "output_sum": float(output.float().sum().item()),
        "loss": float(loss.item()),
        "x_grad_sum": float(workload["x"].grad.float().sum().item()),
    }
    wrapper_state = workload["wrapper_state"]
    if args.graph_kind == "plain":
        checksums.update(
            {
                "weight_grad_sum": float(
                    workload["weight"].grad.float().sum().item()
                ),
                "bias_grad_sum": float(
                    workload["bias"].grad.float().sum().item()
                ),
            }
        )
    elif args.graph_kind == "arena-dispatcher-fullgraph":
        from toolkit.memory_management.arena_offload import transfer

        observed_transfer = transfer.lifetime_fetch_stats()["bytes"] - int(
            arena_transfer_before
        )
        wrapper_state = {
            **wrapper_state,
            "observed_training_h2d_bytes": observed_transfer,
            "adapter_grads": sum(
                int(parameter.grad is not None)
                for parameter in workload["adapters"]
            ),
        }
        checksums["adapter_grad_sum"] = float(
            sum(
                parameter.grad.float().sum()
                for parameter in workload["adapters"]
                if parameter.grad is not None
            ).item()
        )
        if wrapper_state["adapter_grads"] != len(workload["adapters"]):
            raise RuntimeError(f"arena adapter gradients were incomplete: {wrapper_state}")
        if observed_transfer != wrapper_state["planned_training_h2d_bytes"]:
            raise RuntimeError(f"arena transfer accounting mismatch: {wrapper_state}")
    else:
        wrapper_state = {
            **wrapper_state,
            **workload["wrapper_calls"],
            "base_weight_grads": sum(
                int(linear.weight.grad is not None)
                for linear in workload["managed_layers"]
            ),
        }
        checksums["scale_grad_sum"] = float(
            workload["block"].scale.grad.float().sum().item()
        )
        if wrapper_state["forward_stage"] != 2:
            raise RuntimeError(f"MM forward staging was not exercised: {wrapper_state}")
        if wrapper_state["backward_stage"] != 2:
            raise RuntimeError(f"MM backward staging was not exercised: {wrapper_state}")
        if wrapper_state["base_weight_grads"] != 0:
            raise RuntimeError(f"frozen MM weights received gradients: {wrapper_state}")
    if not all(math.isfinite(value) for value in checksums.values()):
        raise RuntimeError(f"non-finite result: {checksums}")

    saved_info = None
    save_seconds = 0.0
    if args.mode == "produce":
        started = time.perf_counter()
        saved = save_compile_cache_artifact(artifact_path)
        save_seconds = time.perf_counter() - started
        if saved is None:
            raise RuntimeError("torch produced no cache artifacts")
        artifact_bytes = artifact_path.read_bytes()
        saved_info = saved.info

    counters = counter_snapshot(torch)
    evidence = cache_evidence(counters, backend_calls)

    if artifact_bytes is None and artifact_path.is_file():
        artifact_bytes = artifact_path.read_bytes()
    result = {
        "schema": "torch_megacache_cuda_probe_result.v1",
        "mode": args.mode,
        "expected_arm": args.expected_arm,
        "cache_dir": str(cache_dir),
        "artifact": str(artifact_path),
        "manifest": _manifest(torch, args),
        "artifact_bytes": len(artifact_bytes) if artifact_bytes is not None else 0,
        "artifact_sha256": (
            hashlib.sha256(artifact_bytes).hexdigest()
            if artifact_bytes is not None
            else None
        ),
        "loaded_artifacts": compile_cache_artifact_counts(load_info),
        "saved_artifacts": compile_cache_artifact_counts(saved_info),
        "load_seconds": load_seconds,
        "execute_seconds": execute_seconds,
        "save_seconds": save_seconds,
        "cache_disk": {
            "before_load": disk_before,
            "after_load": disk_after_load,
            "after_execute": cache_dir_stats(cache_dir),
        },
        "checksums": checksums,
        "wrapper_state": wrapper_state,
        "evidence": evidence,
        "counters": counters,
        "compile_times": torch._dynamo.utils.compile_times(
            repr="str", aggregate=False
        ),
        "windows_triton_bundle_fallback": (
            windows_triton_bundle_fallback_stats()
        ),
    }
    _write_result(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    _validate_arm(args, evidence)
    if args.graph_kind == "arena-dispatcher-fullgraph":
        from toolkit.memory_management.arena_offload import close_arena_offload

        close_arena_offload(workload["model"])
    return 0


if __name__ == "__main__":
    parsed_args = _parse_args()
    _prepare_environment(parsed_args)

    from scripts.smoke_runtime import run_locked

    raise SystemExit(
        run_locked(
            "torch-megacache-probe",
            lambda: _main(parsed_args),
            detail=parsed_args.expected_arm,
        )
    )
