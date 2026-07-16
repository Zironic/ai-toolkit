"""Deterministic Phase 3/4 oracle for the production Krea dispatcher runtime.

Compares a compiled resident Krea block loop with the production
prepare_arena_offload / GenericBlockDispatcherRuntime path. Both arms use real
SingleStreamBlock math, selected production quantization and adapter forwards,
fixed inputs, and model-owned non-reentrant checkpointing.

The same narrow runtime also carries the Phase 3 correctness-only observation
for an outer whole-model torch.compile wrapper. Graph breaks at the disabled
dispatcher boundaries are allowed; numerical correctness is the gate.

The default remains the Phase 3 TorchAO FP8 + LoRA oracle. Phase 4 runs one
distinct mechanism per process, for example:

    --qtype qfloat8 --adapter-variant none --skip-outer-compile
    --qtype convrot8 --adapter-variant none --skip-outer-compile
    --qtype orbit4 --adapter-variant none --skip-outer-compile
    --qtype float8 --adapter-variant dora --skip-outer-compile
    --qtype float8 --adapter-variant full --skip-outer-compile
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.seam_proof_dispatcher import (  # noqa: E402
    adapter_grad_diff,
    adapter_grads,
    apply_lora,
    build_block,
    make_inputs,
)
from scripts.smoke_profiles import (  # noqa: E402
    assert_representation,
    audit_quantized_representation,
)
from scripts.smoke_quantized_linear_cuda import (  # noqa: E402
    _apply_adapter,
    _quantize_synthetic,
)
from scripts.smoke_runtime import add_lock_args  # noqa: E402
from toolkit.memory_management.arena_offload import (  # noqa: E402
    ArenaOffloadConfig,
    close_arena_offload,
    prepare_arena_offload,
)
from toolkit.memory_management.runtime import get_memory_runtime  # noqa: E402


class _NarrowKreaTransformer(torch.nn.Module):
    """The real Krea repeated-block callable with model-owned checkpointing."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)
        self.gradient_checkpointing = True
        self._checkpoint_keep_last = 0

    def forward(self, x, vec, freqs):
        for block in self.blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, vec, freqs, None, use_reentrant=False)
            else:
                x = block(x, vec, freqs, None)
        return x


def _block_args(args, seed):
    return SimpleNamespace(
        seed=seed,
        features=args.features,
        heads=args.heads,
        multiplier=args.multiplier,
        variant="torchao" if args.qtype == "float8" else "dense",
    )


def _build_model(args, device):
    blocks = [
        build_block(_block_args(args, args.seed + index), device, torch.bfloat16)
        for index in range(args.blocks)
    ]
    model = _NarrowKreaTransformer(blocks).train()
    if args.qtype != "float8":
        _quantize_synthetic(model, args.qtype)
        # Quanto replaces frozen dense weights with fresh Parameters whose
        # requires_grad flag defaults to True. Canonical model storage is
        # always frozen; only subsequently installed adapters are trainable.
        model.requires_grad_(False)
    return model


def _install_adapter(model, args, device):
    if args.adapter_variant == "none":
        return [], {}
    if args.adapter_variant != "lora":
        # LoRASpecialNetwork constructors initialize adapter state from the
        # process RNG. Reset both arms to the same state before installation.
        torch.manual_seed(args.seed + 100)
        torch.cuda.manual_seed_all(args.seed + 100)
        network = _apply_adapter(
            model,
            device,
            torch.bfloat16,
            args.adapter_variant,
            rank=args.lora_rank,
            alpha=float(args.lora_rank),
            target_lin_modules=["_NarrowKreaTransformer"],
            full_if_contains=["blocks.0.attn.wq"],
        )
        modules = {
            getattr(module, "lora_name", f"adapter_{index}"): module
            for index, module in enumerate(network.unet_loras)
        }
        return [network], modules

    modules = {}
    networks = []
    for index, block in enumerate(model.blocks):
        network, installed = apply_lora(
            block, device, args.lora_rank, args.seed + 100 + index
        )
        networks.append(network)
        modules.update(
            {f"blocks.{index}.{path}": module for path, module in installed.items()}
        )
    return networks, modules


def _compile_resident_blocks(model):
    for block in model.blocks:
        installed_forward = block.forward
        block.forward = torch.compile(
            installed_forward,
            mode="default",
            fullgraph=False,
            dynamic=False,
        )


def _run_arm(
    model,
    lora_modules,
    x,
    vec,
    freqs,
    *,
    context=None,
    networks=(),
):
    value = x.detach().clone().requires_grad_(True)
    manager = context if context is not None else torch.enable_grad()
    with contextlib.ExitStack() as stack:
        stack.enter_context(manager)
        for network in networks:
            if hasattr(network, "__enter__") and hasattr(network, "__exit__"):
                stack.enter_context(network)
        output = model(value, vec, freqs)
        loss = output.float().square().mean()
        loss.backward()
    torch.cuda.synchronize(x.device)
    gradients = adapter_grads(lora_modules)
    return {
        "output": output.detach(),
        "input_grad": value.grad.detach(),
        "adapter_grads": gradients,
        "loss": float(loss.detach()),
        "finite": bool(
            torch.isfinite(output).all()
            and torch.isfinite(value.grad).all()
            and torch.isfinite(loss)
            and all(
                grad is not None and torch.isfinite(grad).all()
                for grad in gradients.values()
            )
        ),
        "adapter_gradients": len(gradients),
        "adapter_gradients_present": sum(
            grad is not None for grad in gradients.values()
        ),
    }


def _max_abs(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max())


def _frames():
    return int(torch._dynamo.utils.counters["frames"].get("total", 0))


def _graph_breaks():
    return int(sum(torch._dynamo.utils.counters["graph_break"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--qtype",
        choices=("float8", "qfloat8", "convrot8", "orbit4"),
        default="float8",
        help="one production storage/kernel mechanism per invocation",
    )
    parser.add_argument(
        "--adapter-variant",
        choices=("none", "lora", "lokr", "dora", "full"),
        default="lora",
        help="one installed-forward ownership mechanism per invocation",
    )
    parser.add_argument("--features", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--multiplier", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-tolerance",
        type=float,
        default=None,
        help="compiled BF16 max-absolute output tolerance; defaults to "
        "0.0625 for FP8 wrappers and 0.25 for Ostris custom operators",
    )
    parser.add_argument(
        "--gradient-tolerance",
        type=float,
        default=5e-3,
        help="max-absolute input and adapter gradient tolerance",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--compile-cache-dir", default="tmp/torch_compile_cache")
    parser.add_argument("--no-compile-cache", action="store_true")
    parser.add_argument(
        "--skip-outer-compile",
        action="store_true",
        help="skip the Phase 3 whole-model compile observation for P4 cases",
    )
    add_lock_args(parser)
    args = parser.parse_args()

    if args.output_tolerance is None:
        args.output_tolerance = (
            0.25 if args.qtype in ("convrot8", "orbit4") else 6.25e-2
        )

    if args.blocks < 2:
        raise SystemExit("--blocks must be at least 2 for repeated-block discovery")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("this oracle is CUDA-only")
    torch._dynamo.config.suppress_errors = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from toolkit.memory_management import pin_manager

    pinned_before = pin_manager.pinned_bytes_by_kind().get("weights", 0)
    baseline = _build_model(args, device)
    runtime_model = _build_model(args, device)

    baseline_representation = audit_quantized_representation(baseline)
    runtime_representation = audit_quantized_representation(runtime_model)
    representation_failures = assert_representation(
        baseline_representation, args.qtype
    ) + assert_representation(runtime_representation, args.qtype)

    config = ArenaOffloadConfig(
        enabled=True,
        compile_blocks=True,
        _compile_dynamic=False,
    )
    runtime = prepare_arena_offload(
        runtime_model,
        device=device,
        block_names=("blocks",),
        config=config,
    )
    runtime.transition_training_blocks(
        tuple(f"blocks.{index}" for index in range(args.blocks)),
        resident=True,
    )

    _baseline_networks, baseline_loras = _install_adapter(
        baseline, args, device
    )
    _runtime_networks, runtime_loras = _install_adapter(
        runtime_model, args, device
    )
    # Keep both numerical arms compiled. For Ostris, checkpointing the
    # ordinary compiled control can reorder AOT saved tensors on recompute;
    # disable checkpointing only in that control. The production dispatcher
    # arm remains model-checkpointed, so P4 still exercises the required
    # custom-op backward under the real runtime lifetime.
    resident_checkpointing = args.qtype not in ("convrot8", "orbit4")
    baseline.gradient_checkpointing = resident_checkpointing
    _compile_resident_blocks(baseline)
    runtime.finalize()
    restored_targets = tuple(runtime._executor._saved_forwards)

    from toolkit.compile_cache import CompileCacheSession

    compile_cache = CompileCacheSession(
        args.compile_cache_dir,
        f"smoke_krea2_dispatcher_oracle_{torch.__version__}_{args.qtype}",
        enabled=not args.no_compile_cache,
        logger=lambda message: print(f"[oracle] {message}"),
    )
    compile_cache.load()

    input_args = SimpleNamespace(
        seed=args.seed,
        batch=args.batch,
        seq=args.seq,
        features=args.features,
    )
    x, vec, freqs = make_inputs(
        input_args, device, torch.bfloat16, args.features // args.heads
    )

    resident = _run_arm(
        baseline,
        baseline_loras,
        x,
        vec,
        freqs,
        networks=_baseline_networks,
    )
    frames_before = _frames()
    # This oracle isolates state substitution. Enter the production executor
    # directly so the live residency controller cannot demote the forced-
    # resident comparison in response to unrelated desktop/GPU pressure.
    dispatcher = _run_arm(
        runtime_model,
        runtime_loras,
        x,
        vec,
        freqs,
        context=runtime._executor.execution(runtime._executor.TRAIN),
        networks=_runtime_networks,
    )
    dispatcher_frames = _frames() - frames_before
    diagnostics = runtime.diagnostics()
    accounting = diagnostics["accounting"]

    numerical = {
        "output_max_abs_diff": _max_abs(dispatcher["output"], resident["output"]),
        "input_grad_max_abs_diff": _max_abs(
            dispatcher["input_grad"], resident["input_grad"]
        ),
        "adapter_grad_max_abs_diff": adapter_grad_diff(
            dispatcher["adapter_grads"], resident["adapter_grads"]
        ),
    }
    numerical_ok = (
        resident["finite"]
        and dispatcher["finite"]
        and numerical["output_max_abs_diff"] <= args.output_tolerance
        and numerical["input_grad_max_abs_diff"] <= args.gradient_tolerance
        and numerical["adapter_grad_max_abs_diff"] <= args.gradient_tolerance
        and dispatcher["adapter_gradients_present"]
        == dispatcher["adapter_gradients"]
    )

    if args.skip_outer_compile:
        outer_ok = True
        outer_compile = {"skipped": True, "correct": None}
    else:
        frames_before = _frames()
        breaks_before = _graph_breaks()
        outer_compiled = torch.compile(
            lambda value, modulation, rope: runtime_model(
                value, modulation, rope
            ),
            mode="default",
            fullgraph=False,
            dynamic=False,
        )
        outer = _run_arm(
            outer_compiled,
            runtime_loras,
            x,
            vec,
            freqs,
            context=runtime._executor.execution(runtime._executor.TRAIN),
            networks=_runtime_networks,
        )
        outer_numerical = {
            "output_max_abs_diff": _max_abs(
                outer["output"], dispatcher["output"]
            ),
            "input_grad_max_abs_diff": _max_abs(
                outer["input_grad"], dispatcher["input_grad"]
            ),
            "adapter_grad_max_abs_diff": adapter_grad_diff(
                outer["adapter_grads"], dispatcher["adapter_grads"]
            ),
        }
        outer_ok = (
            outer["finite"]
            and outer_numerical["output_max_abs_diff"]
            <= args.output_tolerance
            and outer_numerical["input_grad_max_abs_diff"]
            <= args.gradient_tolerance
            and outer_numerical["adapter_grad_max_abs_diff"]
            <= args.gradient_tolerance
            and outer["adapter_gradients_present"]
            == outer["adapter_gradients"]
        )
        outer_compile = {
            "correct": outer_ok,
            "new_frames": _frames() - frames_before,
            "graph_breaks_added": _graph_breaks() - breaks_before,
            **outer_numerical,
        }
        del outer_compiled

    compile_cache.save(force=True)
    executor = runtime._executor
    close_arena_offload(runtime_model)
    pinned_after = pin_manager.pinned_bytes_by_kind().get("weights", 0)
    teardown = {
        "installed_forwards_restored": all(
            block.forward is installed
            for block, installed in zip(
                runtime_model.blocks, restored_targets, strict=True
            )
        ),
        "runtime_marker_cleared": get_memory_runtime(runtime_model) is None,
        "executor_dispatchers_released": not executor._dispatchers,
        "executor_saved_forwards_released": not executor._saved_forwards,
        "executor_invokers_released": not executor._invokers,
        "pinned_weights_before": int(pinned_before),
        "pinned_weights_after": int(pinned_after),
    }
    teardown_ok = (
        teardown["installed_forwards_restored"]
        and teardown["runtime_marker_cleared"]
        and teardown["executor_dispatchers_released"]
        and teardown["executor_saved_forwards_released"]
        and teardown["executor_invokers_released"]
        and pinned_after == pinned_before
    )

    result = {
        "oracle": "production_krea_dispatcher",
        "torch": torch.__version__,
        "qtype": args.qtype,
        "adapter_variant": args.adapter_variant,
        "resident_compiled": True,
        "resident_checkpointing": resident_checkpointing,
        "representation": {
            "baseline": baseline_representation,
            "dispatcher": runtime_representation,
            "failures": representation_failures,
        },
        "tolerances": {
            "output_max_abs": args.output_tolerance,
            "gradient_max_abs": args.gradient_tolerance,
        },
        "resident": {
            "loss": resident["loss"],
            "finite": resident["finite"],
            "adapter_gradients": resident["adapter_gradients"],
            "adapter_gradients_present": resident["adapter_gradients_present"],
        },
        "dispatcher": {
            "loss": dispatcher["loss"],
            "finite": dispatcher["finite"],
            "adapter_gradients": dispatcher["adapter_gradients"],
            "adapter_gradients_present": dispatcher[
                "adapter_gradients_present"
            ],
            "compile_frames": dispatcher_frames,
        },
        "numerical": numerical,
        "accounting": accounting,
        "state_audit": diagnostics["state_audit"],
        "checkpoint_owner": diagnostics["checkpoint_owner"],
        "outer_compile": outer_compile,
        "teardown": teardown,
        "ok": bool(
            numerical_ok
            and not representation_failures
            and accounting["payload_reconciled"]
            and accounting["streamed_leaves"] == 0
            and diagnostics["checkpoint_owner"] == "model"
            and outer_ok
            and teardown_ok
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[oracle] wrote {output}")
    if not result["ok"]:
        raise SystemExit("P3/P4 deterministic dispatcher oracle failed")
    print("[oracle] production Krea dispatcher oracle passed")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from smoke_runtime import run_locked

    sys.exit(run_locked("smoke_krea2_dispatcher_oracle_cuda", main))
