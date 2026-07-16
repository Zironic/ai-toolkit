"""Shared multi-architecture full-model training smoke (CUDA).

Owns everything architecture-neutral: CLI parsing, deterministic seeding,
model-config/runtime wiring, fake flow-matching training steps, adapter and
optimizer setup, train -> sample -> train phase transitions, CUDA/DXGI
snapshots, compile counters, JSON output, common assertions, and teardown.
Architecture specifics (model construction, transformer loading, valid fake
inputs, conditioning validation) come from scripts/smoke_profiles.py:

    --profile krea2 | zimage | ideogram4 | anima

The default phase sequence `train,train,sample,train` proves training ->
sampling -> training survival: sample state publication, checkpoint state
staying usable, compiled training kernels surviving the transition.

Every profile defaults to the smoke-only direct-to-arena load lifecycle.
Use ``--load-mode production-model-load`` only when the production checkpoint
loading path itself is the behavior under test.

Examples:

    venv\\Scripts\\python.exe scripts\\smoke_transformer_train_cuda.py ^
      --profile zimage --model-path "<z-image-path>" ^
      --cond-cache "<cond.safetensors>" --qtype qfloat8 ^
      --adapter-variant lora --phase-sequence train,train,sample,train ^
      --resolution 256x256

    venv\\Scripts\\python.exe scripts\\smoke_transformer_train_cuda.py ^
      --profile ideogram4 --model-path "<ideogram4-path>" ^
      --cond-cache "<short.safetensors>" --cond-cache "<long.safetensors>" ^
      --batch-size 2 --qtype convrot4 --adapter-variant lora ^
      --resolution 256x256
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_profiles import (  # noqa: E402
    PROFILES,
    assert_representation,
    audit_quantized_representation,
)
from scripts.smoke_runtime import (  # noqa: E402
    CudaPhysicalFreeMonitor,
    SMOKE_DIRECT_LOAD_MODE,
    add_contention_args,
    add_load_mode_arg,
    add_lock_args,
    assert_smoke_load_mode,
    configure_smoke_load_mode,
    fail_if_vram_contended,
    smoke_model_load_session,
)

GIB = 1024**3


def _gib(value):
    if value is None:
        return None
    return float(value) / GIB


def _cuda_snapshot(label, device):
    from toolkit.memory_management import bounce_pool

    row = {"label": label, "available": torch.cuda.is_available()}
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return row
    free_b, total_b = torch.cuda.mem_get_info(device)
    row.update(
        {
            "free_gib": _gib(free_b),
            "total_gib": _gib(total_b),
            "torch_allocated_gib": _gib(torch.cuda.memory_allocated(device)),
            "torch_max_allocated_gib": _gib(torch.cuda.max_memory_allocated(device)),
            "torch_reserved_gib": _gib(torch.cuda.memory_reserved(device)),
            "pinned_ledger_gib": _gib(bounce_pool._pinned_bytes_total),
        }
    )
    return row


def _dxgi_snapshot(label, device_index=0):
    from toolkit.memory_management import dxgi_meminfo

    info = dxgi_meminfo.query_non_local_video_memory_info(
        cuda_device_index=device_index, min_interval_s=0.0
    )
    row = {"label": label, "budget_gib": None, "usage_gib": None}
    if info is not None:
        row.update(
            {
                "budget_gib": _gib(info.budget_bytes),
                "usage_gib": _gib(info.current_usage_bytes),
            }
        )
    return row


def _dynamo_counters():
    try:
        counters = torch._dynamo.utils.counters
        return {
            "graph_breaks": sum(counters["graph_break"].values()),
            "frames": dict(counters["frames"]),
        }
    except Exception as error:
        return {"error": repr(error)}


def _transfer_snapshot():
    from toolkit.memory_management.arena_offload import transfer

    return transfer.lifetime_fetch_stats()


def _counter_delta(after, before):
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in after
        if isinstance(after.get(key), (int, float))
    }


def _new_frames_since(before):
    return torch._dynamo.utils.counters["frames"].get("total", 0) - before


def _frames_total():
    return torch._dynamo.utils.counters["frames"].get("total", 0)


def _print_json(row):
    print(json.dumps(row, indent=2, sort_keys=True, default=str))


def _tensor_checksum(tensor) -> str:
    value = tensor.detach().float().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _gradient_checksum(module) -> str | None:
    digest = hashlib.sha256()
    found = False
    for name, parameter in sorted(module.named_parameters()):
        if parameter.grad is None:
            continue
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.grad.shape)).encode("ascii"))
        value = parameter.grad.detach().float().contiguous().cpu()
        digest.update(value.numpy().tobytes())
    return digest.hexdigest() if found else None


def _git_head() -> str | None:
    head_path = REPO_ROOT / ".git" / "HEAD"
    try:
        head = head_path.read_text(encoding="ascii").strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        ref_path = REPO_ROOT / ".git" / ref
        if ref_path.is_file():
            return ref_path.read_text(encoding="ascii").strip()
        packed = REPO_ROOT / ".git" / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="ascii").splitlines():
                if line.endswith(f" {ref}"):
                    return line.split(" ", 1)[0]
    except OSError:
        return None
    return None


def _megacache_manifest(args, model, phases, device) -> dict:
    import torch._functorch.config as functorch_config
    import torch._inductor.config as inductor_config
    from toolkit.memory_management.arena_offload import DISPATCHER_GENERATION

    try:
        import triton

        triton_version = triton.__version__
    except Exception:
        triton_version = None
    manifest = {
        "schema": "ai_toolkit.full_model_megacache.v1",
        "toolkit_git_head": _git_head(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton_version,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "model": {
            "profile": args.profile,
            "path": args.model_path,
            "qtype": model.model_config.qtype,
            "adapter_variant": args.adapter_variant,
            "lora_rank": int(args.lora_rank),
            "lora_alpha": float(args.lora_alpha),
            "assistant_lora": args.assistant_lora,
        },
        "workload": {
            "resolution": args.resolution,
            "batch_size": int(args.batch_size),
            "phase_sequence": list(phases),
            "seed": int(args.seed),
            "conditioning": [str(Path(path).resolve()) for path in args.cond_cache],
        },
        "arena": {
            "simulated_vram_gib": float(args.simulated_vram_gib),
            "working_reserve_gib": str(args.working_reserve_gib),
            "residency_policy_frozen": bool(args.freeze_arena_residency),
            "expected_residency": args.expected_residency,
        },
        "compile": {
            "backend": "inductor",
            "mode": "default",
            "fullgraph": bool(args.compile_fullgraph),
            "dynamic": args.compile_dynamic_resolved,
            "dispatcher_generation": str(DISPATCHER_GENERATION),
            "aot_autograd_cache": bool(functorch_config.enable_autograd_cache),
            "strict_autograd_cache": bool(functorch_config.strict_autograd_cache),
            "fx_graph_cache": bool(inductor_config.fx_graph_cache),
            "autotune_local_cache": bool(inductor_config.autotune_local_cache),
            "coordinate_descent_tuning": bool(
                inductor_config.coordinate_descent_tuning
            ),
            "coordinate_descent_check_all_directions": bool(
                inductor_config.coordinate_descent_check_all_directions
            ),
            "compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
        },
    }
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _manifest_sidecar_path(args) -> Path | None:
    if args.megacache_manifest:
        return Path(args.megacache_manifest).resolve()
    if args.megacache_artifact:
        artifact = Path(args.megacache_artifact).resolve()
        return artifact.with_name(f"{artifact.name}.manifest.json")
    return None


def _artifact_manifest_path(artifact) -> Path:
    artifact = Path(artifact).resolve()
    return artifact.with_name(f"{artifact.name}.manifest.json")


def _artifact_inventory_has(counts: dict[str, int], name: str) -> bool:
    name = name.lower()
    return any(name in key.lower() and int(value) > 0 for key, value in counts.items())


def _artifact_inventory_count(counts: dict[str, int], name: str) -> int:
    name = name.lower()
    return sum(
        int(value) for key, value in counts.items() if name in key.lower()
    )


def _validate_megacache_evidence(
    expected_arm: str,
    evidence: dict,
    expected_variants: int | None,
) -> list[str]:
    failures = []
    if evidence.get("aot_bypass") or evidence.get("fx_bypass"):
        failures.append(f"{expected_arm}: compiler cache bypass observed: {evidence}")
    if evidence.get("graph_breaks"):
        failures.append(f"{expected_arm}: internal graph breaks observed: {evidence}")

    cold = expected_arm in ("cold", "empty-control", "invalidation")
    if cold:
        if not evidence.get("aot_miss") or not evidence.get("fx_miss"):
            failures.append(
                f"{expected_arm}: cold execution did not miss both AOT and FX caches: "
                f"{evidence}"
            )
        if not evidence.get("backend_codegen_observed"):
            failures.append(
                f"{expected_arm}: cold execution performed no observable backend codegen"
            )
    else:
        if not evidence.get("aot_hit") or not evidence.get("fx_hit"):
            failures.append(
                f"{expected_arm}: warm execution did not hit both AOT and FX caches: "
                f"{evidence}"
            )
        if evidence.get("fx_miss"):
            failures.append(f"{expected_arm}: warm execution had an FX cache miss: {evidence}")

        # ZImage's train/sample lifecycle makes four AOT lookups but PyTorch
        # persists only the three autograd entries (the inference lookup is
        # not saved). A restored process therefore reports three AOT hits and
        # one frontend miss even though all five FX graphs hit and no backend
        # code is generated. Permit only that measured non-serializable tail:
        # every expected AOT entry must hit, at most one AOT lookup may miss,
        # and its one runtime autotune selection must not grow beyond it.
        expected_hits = 0 if expected_variants is None else expected_variants
        residual_aot_misses = int(evidence.get("aot_miss", 0))
        residual_autotune = int(evidence.get("autotune_benchmark_calls", 0))
        if residual_aot_misses and (
            int(evidence.get("aot_hit", 0)) < expected_hits
            or residual_aot_misses > 1
        ):
            failures.append(
                f"{expected_arm}: warm AOT misses exceed the single "
                f"non-serializable lookup: {evidence}"
            )
        if residual_autotune > residual_aot_misses:
            failures.append(
                f"{expected_arm}: warm autotune work exceeds the "
                f"non-serializable AOT tail: {evidence}"
            )
        warm_backend_work = (
            evidence.get("backend_codegen_observed")
            or evidence.get("inductor_codegen_calls")
            or evidence.get("triton_compile_calls")
            or evidence.get("coordinate_descent_calls")
        )
        if warm_backend_work:
            failures.append(
                f"{expected_arm}: warm execution performed backend compiler work: "
                f"{evidence}"
            )

    # AOT hit/miss counters count lookup attempts, not distinct variants. A
    # full checkpointed lifecycle can request the same cache entry more than
    # once, so compare the Dynamo unique-graph count with the serialized AOT
    # inventory instead. An intentional invalidation changes the graph/config
    # contract and therefore is not required to preserve producer cardinality.
    if expected_variants is not None and expected_arm != "invalidation":
        actual = int(evidence.get("unique_graphs", 0))
        if actual != expected_variants:
            failures.append(
                f"{expected_arm}: expected {expected_variants} unique graph "
                f"variants, observed {actual}"
            )
    return failures


def _start_megacache_diagnostics(args) -> dict | None:
    if args.megacache_mode == "off":
        return None

    import torch._functorch.config as functorch_config
    import torch._inductor.config as inductor_config

    from scripts.megacache_diagnostics import (
        cache_dir_stats,
        install_backend_instrumentation,
    )

    cache_dir_value = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not cache_dir_value:
        raise SystemExit(
            "MegaCache matrix modes require TORCHINDUCTOR_CACHE_DIR to be set "
            "before this process imports torch; use run_full_model_megacache_matrix.py"
        )
    cache_dir = Path(cache_dir_value).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    initial_disk = cache_dir_stats(cache_dir)
    shared = args.megacache_expected_arm == "shared-disk"
    if shared and not initial_disk["files"]:
        raise SystemExit("shared-disk arm requires the producer's populated cache dir")
    if shared and not args.megacache_allow_populated_inductor_cache:
        raise SystemExit(
            "shared-disk arm requires --megacache-allow-populated-inductor-cache"
        )
    if not shared and initial_disk["files"]:
        raise SystemExit(
            f"{args.megacache_expected_arm} arm requires an empty Inductor cache "
            f"directory, found {initial_disk['files']} files in {cache_dir}"
        )

    functorch_config.enable_autograd_cache = True
    functorch_config.strict_autograd_cache = True
    inductor_config.fx_graph_cache = True
    inductor_config.autotune_local_cache = True
    return {
        "cache_dir": cache_dir,
        "initial_disk": initial_disk,
        "backend_calls": install_backend_instrumentation(),
    }


def _activate_megacache(
    args,
    model,
    phases,
    device,
    diagnostics: dict | None,
    failures: list[str],
) -> dict | None:
    if diagnostics is None:
        return None

    from scripts.megacache_diagnostics import cache_dir_stats
    from toolkit.compile_cache import load_compile_cache_artifact

    torch._dynamo.utils.counters.clear()
    diagnostics["backend_baseline"] = dict(diagnostics["backend_calls"])
    manifest = _megacache_manifest(args, model, phases, device)
    artifact_path = (
        None
        if args.megacache_artifact is None
        else Path(args.megacache_artifact).resolve()
    )
    manifest_path = _manifest_sidecar_path(args)
    producer_manifest = None
    manifest_matches = None
    loaded = None
    load_seconds = 0.0

    if args.megacache_mode == "consume":
        if manifest_path is None or not manifest_path.is_file():
            failures.append(f"MegaCache producer manifest is absent: {manifest_path}")
        else:
            try:
                producer_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                failures.append(f"failed to read MegaCache producer manifest: {error}")
        if producer_manifest is not None:
            manifest_matches = (
                producer_manifest.get("fingerprint") == manifest.get("fingerprint")
            )
            if (
                args.megacache_expected_arm == "megacache"
                and not manifest_matches
                and not args.megacache_measure_only
            ):
                failures.append(
                    "MegaCache consumer manifest does not match the producer; "
                    "artifact load was rejected"
                )
            elif args.megacache_expected_arm == "invalidation" and manifest_matches:
                failures.append(
                    "invalidation arm did not change a compile-relevant manifest field"
                )
        should_load = (
            manifest_matches
            or args.megacache_expected_arm == "invalidation"
            or args.megacache_measure_only
        )
        if should_load and artifact_path is not None:
            started = time.perf_counter()
            loaded = load_compile_cache_artifact(artifact_path)
            load_seconds = time.perf_counter() - started
            if loaded is None:
                failures.append(f"torch rejected or did not find MegaCache artifact {artifact_path}")

    diagnostics.update(
        {
            "manifest": manifest,
            "producer_manifest": producer_manifest,
            "manifest_matches": manifest_matches,
            "measure_only": bool(args.megacache_measure_only),
            "artifact": None if artifact_path is None else str(artifact_path),
            "manifest_path": None if manifest_path is None else str(manifest_path),
            "loaded": loaded is not None,
            "loaded_artifacts": (
                {} if loaded is None else loaded.artifact_counts
            ),
            "artifact_bytes_loaded": 0 if loaded is None else loaded.byte_count,
            "load_seconds": load_seconds,
            "disk_after_load": cache_dir_stats(diagnostics["cache_dir"]),
        }
    )
    if loaded is not None:
        if not _artifact_inventory_has(loaded.artifact_counts, "aot"):
            failures.append(
                f"loaded MegaCache inventory has no AOT artifact: {loaded.artifact_counts}"
            )
        if not _artifact_inventory_has(loaded.artifact_counts, "inductor"):
            failures.append(
                "loaded MegaCache inventory has no Inductor artifact: "
                f"{loaded.artifact_counts}"
            )
        expected = args.megacache_expected_variants
        if expected is not None and _artifact_inventory_count(
            loaded.artifact_counts, "aot"
        ) != expected:
            failures.append(
                f"loaded MegaCache inventory does not contain {expected} AOT variants: "
                f"{loaded.artifact_counts}"
            )
    return diagnostics


def _finish_megacache(args, diagnostics: dict | None, failures: list[str]) -> dict | None:
    if diagnostics is None:
        return None

    from scripts.megacache_diagnostics import (
        cache_dir_stats,
        cache_evidence,
        counter_snapshot,
        numeric_delta,
    )
    from toolkit.compile_cache import (
        save_compile_cache_artifact,
        windows_triton_bundle_fallback_stats,
    )

    counters = counter_snapshot(torch)
    backend_delta = numeric_delta(
        diagnostics["backend_calls"], diagnostics["backend_baseline"]
    )
    evidence = cache_evidence(counters, backend_delta)
    saved = None
    save_seconds = 0.0
    save_target = None
    if args.megacache_mode == "produce":
        save_target = args.megacache_artifact
    elif args.megacache_update_artifact:
        save_target = args.megacache_update_artifact
    if save_target is not None:
        started = time.perf_counter()
        saved = save_compile_cache_artifact(save_target)
        save_seconds = time.perf_counter() - started
        if saved is None:
            failures.append("torch produced no MegaCache artifact")
        else:
            manifest_path = _artifact_manifest_path(save_target)
            if manifest_path is None:
                failures.append("MegaCache producer has no manifest path")
            else:
                _write_json_atomic(manifest_path, diagnostics["manifest"])
            if not _artifact_inventory_has(saved.artifact_counts, "aot"):
                failures.append(
                    f"saved MegaCache inventory has no AOT artifact: {saved.artifact_counts}"
                )
            if not _artifact_inventory_has(saved.artifact_counts, "inductor"):
                failures.append(
                    "saved MegaCache inventory has no Inductor artifact: "
                    f"{saved.artifact_counts}"
                )
            expected = args.megacache_expected_variants
            if (
                expected is not None
                and args.megacache_mode == "produce"
                and _artifact_inventory_count(
                    saved.artifact_counts, "aot"
                ) != expected
            ):
                failures.append(
                    f"saved MegaCache inventory does not contain {expected} AOT variants: "
                    f"{saved.artifact_counts}"
                )

    if args.megacache_expected_arm in ("megacache", "invalidation") and not diagnostics[
        "loaded"
    ]:
        failures.append(
            f"{args.megacache_expected_arm} consumer did not accept an artifact"
        )
    if not args.megacache_measure_only:
        failures.extend(
            _validate_megacache_evidence(
                args.megacache_expected_arm,
                evidence,
                args.megacache_expected_variants,
            )
        )
    diagnostics.update(
        {
            "counters": counters,
            "backend_calls": backend_delta,
            "evidence": evidence,
            "saved_artifacts": {} if saved is None else saved.artifact_counts,
            "artifact_bytes_saved": 0 if saved is None else saved.byte_count,
            "updated_artifact": (
                None
                if args.megacache_update_artifact is None
                else str(Path(args.megacache_update_artifact).resolve())
            ),
            "save_seconds": save_seconds,
            "disk_after_execute": cache_dir_stats(diagnostics["cache_dir"]),
            "windows_triton_bundle_fallback": (
                windows_triton_bundle_fallback_stats()
            ),
        }
    )
    diagnostics["cache_dir"] = str(diagnostics["cache_dir"])
    return diagnostics


def _install_nonfinite_trace(transformer):
    """Report the first non-finite major-stage output without tracing leaves."""
    state = {"first": None}
    handles = []

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)

    def hook(name):
        def record(_module, _args, output):
            if state["first"] is not None:
                return
            for value in tensors(output):
                if value.is_floating_point() and not bool(
                    torch.isfinite(value).all().item()
                ):
                    state["first"] = {
                        "module": name,
                        "shape": tuple(value.shape),
                        "dtype": str(value.dtype),
                    }
                    print(f"[smoke] first non-finite output: {state['first']}")
                    return

        return record

    for name, module in transformer.named_modules():
        if not name:
            continue
        depth = name.count(".")
        if depth == 0 or (name.startswith("blocks.") and depth == 1):
            handles.append(module.register_forward_hook(hook(name)))
    return state, handles


# ---------------------------------------------------------------------------
# Arena attach (shared; profiles must not own this)
# ---------------------------------------------------------------------------

def _attach_arena_runtime(
    model, transformer, profile, device, *, canonical_build=None
):
    """Attach the arena runtime, adopting any smoke-prepared canonical build."""
    ignore_modules = profile.arena_ignore_modules(transformer)
    attach = getattr(model, "_attach_immutable_training_memory", None)
    if attach is not None:
        if canonical_build is not None:
            if getattr(model, "_prepared_canonical_build", None) is not None:
                canonical_build.rollback()
                raise RuntimeError("smoke_multiple_prepared_canonical_builds")
            model._prepared_canonical_build = canonical_build
        runtime = attach(transformer, ignore_modules)
        # Mirror the production load_model sequence. Ranged quantized loading
        # leaves non-block modules on CPU; after canonical commit the runtime
        # moves only those permanent regions without disturbing arena views.
        runtime.place_permanent_modules(device, model.torch_dtype)
        return
    from toolkit.memory_management.arena_offload import (
        ArenaOffloadConfig,
        prepare_arena_offload,
    )

    runtime = prepare_arena_offload(
        transformer,
        device=device,
        block_names=model.get_transformer_block_names(),
        config=ArenaOffloadConfig.from_model_config(model.model_config),
        ignore_modules=ignore_modules,
        canonical_build=canonical_build,
    )
    runtime.place_permanent_modules(device, model.torch_dtype)


# ---------------------------------------------------------------------------
# Adapter setup (mirrors BaseSDTrainProcess order)
# ---------------------------------------------------------------------------

def _apply_adapter(model, transformer, device, args, profile):
    from toolkit.config_modules import NetworkConfig
    from toolkit.lora_special import LoRASpecialNetwork

    transformer.requires_grad_(False)
    transformer.train()
    full_targets = list(args.full_if_contains or [])
    if args.adapter_variant == "full" and not full_targets:
        full_targets = [profile.default_fullmodule_target(transformer)]
    network_type = "lora" if args.adapter_variant == "full" else args.adapter_variant
    network_config = NetworkConfig(
        type=network_type,
        linear=args.lora_rank,
        linear_alpha=args.lora_alpha,
        transformer_only=True,
        lokr_factor=args.lokr_factor,
    )
    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=transformer,
        lora_dim=network_config.linear,
        multiplier=1.0,
        alpha=network_config.linear_alpha,
        train_unet=True,
        train_text_encoder=False,
        network_config=network_config,
        network_type=network_config.type,
        full_if_contains=full_targets,
        transformer_only=True,
        is_transformer=True,
        target_lin_modules=model.target_lora_modules,
        base_model=model,
    )
    network.force_to(device, dtype=torch.float32)
    model.network = network
    network._update_torch_multiplier()
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    network.can_merge_in = False
    network.prepare_grad_etc(None, transformer)
    network.enable_gradient_checkpointing()
    return network


# ---------------------------------------------------------------------------
# Block-container / discovery audit
# ---------------------------------------------------------------------------

def _discovery_audit(memory_runtime, transformer, profile):
    """What the runtime discovered, checked against the profile's expected
    container -- without the profile ever handing block paths to the arena."""
    executor = getattr(memory_runtime, "_executor", None)
    block_abis = getattr(executor, "_block_abis", None) if executor else None
    discovered = len(block_abis) if block_abis is not None else None
    container = getattr(transformer, profile.expected_block_container, None)
    container_len = len(container) if container is not None else None
    return {
        "expected_block_container": profile.expected_block_container,
        "container_present": container is not None,
        "container_blocks": container_len,
        "discovered_blocks": discovered,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_resolution(spec):
    w_s, _, h_s = spec.partition("x")
    if not h_s:
        raise SystemExit(f"--resolution expects WxH, got {spec!r}")
    return int(w_s), int(h_s)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Shared multi-architecture CUDA training smoke. "
        "Do not hook this up to pytest."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--cond-cache",
        action="append",
        default=None,
        help="cached conditioning safetensors; repeat for multiple samples "
        "(ideogram4 keeps them separate to preserve natural token lengths)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--qtype", default="qfloat8")
    parser.add_argument("--cache-dir", default=None)
    add_load_mode_arg(parser)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--resolution", default="256x256")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--phase-sequence",
        default="train,train,sample,train",
        help="comma-separated phases; 'sample' is a no-grad noise prediction "
        "under the runtime's sampling boundary (no VAE decode)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument(
        "--adapter-variant",
        choices=("lora", "lokr", "dora", "full"),
        default="lora",
    )
    parser.add_argument("--full-if-contains", action="append", default=None)
    parser.add_argument("--lokr-factor", type=int, default=-1)
    parser.add_argument(
        "--assistant-lora",
        default=None,
        help="zimage only: assistant adapter merged before quantization; this "
        "model-specific path converts qfloat8 -> float8",
    )
    parser.add_argument("--working-reserve-gib", default="-1")
    parser.add_argument("--wddm-margin-gib", type=float, default=1.0)
    parser.add_argument("--wddm-hard-gib", type=float, default=1.0)
    parser.add_argument(
        "--simulated-vram-gib",
        type=float,
        default=0.0,
        help=(
            "pretend the card has this many GiB so residency/streaming and "
            "allocator caps match the smaller card; 0 uses the real card"
        ),
    )
    parser.add_argument("--checkpoint-keep-last", type=int, default=0)
    parser.add_argument("--prefetch-depth", type=int, default=3)
    parser.add_argument(
        "--freeze-arena-residency",
        action="store_true",
        help=(
            "smoke-only: hold the finalized training residency plan fixed so "
            "compiler-memory differences cannot change cache variants"
        ),
    )
    parser.add_argument(
        "--expected-residency",
        choices=("mixed", "full"),
        default=None,
        help="assert the finalized smoke plan has this residency class",
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--compile-cache-dir",
        default="tmp/torch_compile_cache",
        help="shared default-on MegaCache directory for ordinary smoke runs",
    )
    parser.add_argument(
        "--no-compile-cache",
        action="store_true",
        help="explicitly disable MegaCache for an ordinary smoke run",
    )
    parser.add_argument(
        "--compile-dynamic", choices=("true", "false", "none"), default="true"
    )
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument(
        "--compile-coordinate-descent",
        choices=("true", "false", "none"),
        default="none",
        help="explicit TorchAO/Inductor coordinate-descent policy",
    )
    parser.add_argument(
        "--megacache-mode",
        choices=("off", "produce", "control", "consume"),
        default="off",
        help="full-model MegaCache matrix role; the matrix runner sets this",
    )
    parser.add_argument("--megacache-artifact", default=None)
    parser.add_argument(
        "--megacache-update-artifact",
        default=None,
        help=(
            "benchmark-only: after consuming and executing, save the combined "
            "hit/miss artifact inventory to this path"
        ),
    )
    parser.add_argument(
        "--megacache-measure-only",
        action="store_true",
        help=(
            "benchmark-only: load across manifest/residency changes and report "
            "cache misses and compiler work instead of failing on them"
        ),
    )
    parser.add_argument("--megacache-manifest", default=None)
    parser.add_argument(
        "--megacache-expected-arm",
        choices=("cold", "empty-control", "shared-disk", "megacache", "invalidation"),
        default=None,
    )
    parser.add_argument("--megacache-expected-variants", type=int, default=None)
    parser.add_argument(
        "--megacache-allow-populated-inductor-cache", action="store_true"
    )
    parser.add_argument(
        "--attention-backend",
        choices=("native", "flash"),
        default="native",
        help="optional flash_attn cross-check; native SDPA is the primary gate",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--trace-nonfinite",
        action="store_true",
        help="record the first non-finite major transformer-stage output",
    )
    add_contention_args(parser)
    add_lock_args(parser)
    args = parser.parse_args(argv)
    args.compile_dynamic_resolved = (
        None if args.compile_dynamic == "none" else args.compile_dynamic == "true"
    )
    args.compile_coordinate_descent_resolved = (
        None
        if args.compile_coordinate_descent == "none"
        else args.compile_coordinate_descent == "true"
    )
    if args.megacache_mode != "off":
        if args.no_compile:
            parser.error("MegaCache matrix modes require block compilation")
        if not args.compile_fullgraph:
            parser.error("MegaCache full-model acceptance requires --compile-fullgraph")
        if args.megacache_expected_arm is None:
            parser.error("MegaCache matrix modes require --megacache-expected-arm")
        if args.megacache_mode in ("produce", "consume") and not args.megacache_artifact:
            parser.error(f"--megacache-mode {args.megacache_mode} requires an artifact")
        if args.megacache_measure_only and args.megacache_mode != "consume":
            parser.error("--megacache-measure-only requires --megacache-mode consume")
        if args.megacache_update_artifact and args.megacache_mode != "consume":
            parser.error("--megacache-update-artifact requires --megacache-mode consume")
        if args.megacache_expected_variants is not None and args.megacache_expected_variants < 1:
            parser.error("--megacache-expected-variants must be positive")
    return args


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def _train_phase(
    step_num,
    model,
    transformer,
    network,
    embeds,
    latents_cpu,
    generator,
    device,
    include_checksums=False,
):
    from toolkit.memory_management.runtime import get_memory_runtime

    latents = latents_cpu.to(device, model.torch_dtype)
    noise = torch.randn(
        latents_cpu.shape, generator=generator
    ).to(device, model.torch_dtype)
    batch = latents.shape[0]
    timestep = (torch.rand(batch, generator=generator) * 1000.0).to(device)
    t_frac = (timestep.float() / 1000.0).view(-1, 1, 1, 1).to(device)
    noisy = ((1.0 - t_frac) * latents.float() + t_frac * noise.float()).to(
        model.torch_dtype
    )
    target = (noise.float() - latents.float()).detach()

    memory_runtime = get_memory_runtime(transformer)
    execution_context = (
        memory_runtime.training_step(step_num=step_num)
        if memory_runtime is not None
        else contextlib.nullcontext()
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    physical_free_monitor = CudaPhysicalFreeMonitor(device).start()
    try:
        with execution_context, network:
            runtime_accounting = (
                memory_runtime.diagnostics().get("accounting")
                if memory_runtime is not None
                else None
            )
            pred = model.get_noise_prediction(noisy, timestep, embeds)
            loss = torch.nn.functional.mse_loss(pred.float(), target)
            loss.backward()
        torch.cuda.synchronize(device)
    except BaseException:
        physical_free_monitor.stop()
        raise
    result = {
        "seconds": time.perf_counter() - started,
        "loss": loss.item(),
        "pred_norm": pred.detach().float().norm().item(),
        "pred_finite": bool(torch.isfinite(pred.detach().float()).all()),
        "pred_shape": tuple(pred.shape),
        "runtime_accounting": runtime_accounting,
        "_physical_free_monitor": physical_free_monitor,
    }
    if include_checksums:
        result["pred_checksum"] = _tensor_checksum(pred)
    return result


def _sample_phase(
    model,
    transformer,
    network,
    embeds,
    latents_cpu,
    generator,
    device,
    include_checksums=False,
):
    from toolkit.memory_management.runtime import get_memory_runtime

    latents = latents_cpu.to(device, model.torch_dtype)
    batch = latents.shape[0]
    timestep = torch.full((batch,), 500.0, device=device)
    memory_runtime = get_memory_runtime(transformer)
    shape_key = ("smoke_sample", *tuple(latents.shape))
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    if memory_runtime is not None:
        with memory_runtime.sampling_session():
            with memory_runtime.sampling_image(
                shape_key=shape_key, cold_working_bytes=3 * GIB
            ):
                runtime_accounting = memory_runtime.diagnostics().get("accounting")
                with torch.no_grad():
                    pred = model.get_noise_prediction(latents, timestep, embeds)
    else:
        runtime_accounting = None
        with torch.no_grad():
            pred = model.get_noise_prediction(latents, timestep, embeds)
    torch.cuda.synchronize(device)
    result = {
        "seconds": time.perf_counter() - started,
        "pred_norm": pred.detach().float().norm().item(),
        "pred_finite": bool(torch.isfinite(pred.detach().float()).all()),
        "pred_shape": tuple(pred.shape),
        "runtime_accounting": runtime_accounting,
    }
    if include_checksums:
        result["pred_checksum"] = _tensor_checksum(pred)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    profile = PROFILES[args.profile]
    failures: list[str] = []
    rows: list[dict] = []
    megacache = _start_megacache_diagnostics(args)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("this smoke is CUDA-only; pass --device cuda")
    fail_if_vram_contended(device, ignore_contention=args.ignore_contention)
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("AI_TOOLKIT_MEMORY_DIAGNOSTICS", "1")
    if args.attention_backend == "flash":
        # Optional cross-check, gated on flash_attn actually being installed;
        # native SDPA stays the primary gate.
        try:
            import flash_attn  # noqa: F401
        except ImportError as error:
            raise SystemExit(
                "--attention-backend flash requires flash_attn to be installed"
            ) from error

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    phases = [p.strip() for p in args.phase_sequence.split(",") if p.strip()]
    for p in phases:
        if p not in ("train", "sample"):
            raise SystemExit(f"unknown phase {p!r} in --phase-sequence")
    if not args.cond_cache:
        raise SystemExit("at least one --cond-cache is required")
    resolution = _parse_resolution(args.resolution)

    # --- config + exact requested-representation gate --------------------
    config = profile.build_model_config(args)
    from toolkit.compile_utils import configure_quantized_compile_tuning

    if config.qtype != args.qtype and args.assistant_lora is None:
        raise SystemExit(
            f"requested qtype {args.qtype!r} became {config.qtype!r} in the "
            "ModelConfig -- silent qtype replacement"
        )

    print(f"[smoke] constructing {profile.name} model without text encoder")
    model = profile.construct_model(args, config)
    configure_smoke_load_mode(model, args.load_mode)
    canonical_build = None
    rows.append(
        {
            "event": "start",
            "profile": profile.name,
            "load_mode": args.load_mode,
            "requested_qtype": args.qtype,
            "cuda": _cuda_snapshot("start", device),
            "dxgi": _dxgi_snapshot("start"),
        }
    )
    _print_json(rows[-1])

    with smoke_model_load_session(model, args.load_mode):
        print(f"[smoke] loading {profile.name} transformer")
        t0 = time.perf_counter()
        transformer = profile.load_transformer(model, args)
        if profile.name == "krea2":
            assert_smoke_load_mode(model, args.load_mode)
        rows.append(
            {
                "event": "loaded_transformer",
                "load_mode": args.load_mode,
                "seconds": time.perf_counter() - t0,
            }
        )
        _print_json(rows[-1])
        if args.assistant_lora is not None and args.qtype == "qfloat8":
            # Z-Image documented behaviour: merged assistant converts the qtype.
            if model.model_config.qtype != "float8":
                failures.append(
                    "assistant-lora run did not perform the documented "
                    f"qfloat8 -> float8 conversion (qtype={model.model_config.qtype!r})"
                )

        from toolkit.basic import flush
        from toolkit.util.quantize import quantize_model

        flush(garbage_collect=False)
        if config.quantize and not getattr(
            model, "_transformer_quantized_during_load", False
        ):
            print("[smoke] quantizing transformer")
            t0 = time.perf_counter()
            if args.load_mode == SMOKE_DIRECT_LOAD_MODE:
                from toolkit.memory_management.arena_offload import (
                    prepare_canonical_storage,
                )

                canonical_build = prepare_canonical_storage(
                    transformer,
                    block_names=model.get_transformer_block_names(),
                    device=device,
                    defer_blocks=True,
                )
                try:
                    quantize_model(
                        model,
                        transformer,
                        canonical_build=canonical_build,
                    )
                except BaseException:
                    canonical_build.rollback()
                    canonical_build = None
                    raise
            else:
                quantize_model(model, transformer)
            flush()
            rows.append(
                {
                    "event": "quantized_transformer",
                    "seconds": time.perf_counter() - t0,
                }
            )
            _print_json(rows[-1])

        representation = audit_quantized_representation(transformer)
        effective_qtype = model.model_config.qtype
        rep_failures = assert_representation(representation, effective_qtype)
        failures.extend(rep_failures)
        rows.append({"event": "representation", **representation})
        _print_json(rows[-1])

        # TorchAO's quantization setup may install its preferred Inductor
        # tuning defaults. Match production loading by applying the explicit
        # job/smoke policy after model load and quantization are complete.
        configure_quantized_compile_tuning(model.model_config)

        # The canonical arena accepts immutable base storage only. Model-specific
        # loaders such as Krea2 already freeze before attach, but the generic
        # architecture path must establish the same lifecycle explicitly.
        transformer.requires_grad_(False)
        model_ckpt = profile.enable_model_checkpointing(
            transformer, keep_last=args.checkpoint_keep_last
        )

        print("[smoke] attaching arena runtime")
        t0 = time.perf_counter()
        direct_canonical_prepared = bool(
            getattr(model, "_prepared_canonical_build", None) is not None
            or canonical_build is not None
        )
        if args.load_mode == SMOKE_DIRECT_LOAD_MODE:
            assert_smoke_load_mode(
                model,
                args.load_mode,
                canonical_build=canonical_build,
            )
        _attach_arena_runtime(
            model,
            transformer,
            profile,
            device,
            canonical_build=canonical_build,
        )
        canonical_build = None
        model.model = transformer

        from toolkit.memory_management.runtime import (
            close_memory_runtime,
            get_memory_runtime,
        )

        memory_runtime = get_memory_runtime(transformer)
        if memory_runtime is None:
            raise SystemExit(
                "arena runtime absent after attach -- refusing to fall back to "
                "the legacy manager"
            )
        if getattr(transformer, "_memory_manager", None) is not None and not hasattr(
            model, "_attach_immutable_training_memory"
        ):
            failures.append(
                "legacy per-Linear memory manager attached alongside the arena "
                "runtime -- silent fallback"
            )
        rows.append(
            {
                "event": "attached_arena_runtime",
                "direct_canonical_prepared": direct_canonical_prepared,
                "seconds": time.perf_counter() - t0,
                "cuda": _cuda_snapshot("attached", device),
                "dxgi": _dxgi_snapshot("attached"),
            }
        )
        _print_json(rows[-1])

    print(f"[smoke] applying {args.adapter_variant} adapter (rank={args.lora_rank})")
    network = _apply_adapter(model, transformer, device, args, profile)
    trainable = [p for p in network.parameters() if p.requires_grad]
    params = network.prepare_optimizer_params(
        text_encoder_lr=args.lr, unet_lr=args.lr, default_lr=args.lr
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    rows.append(
        {
            "event": "applied_adapter",
            "adapter_variant": args.adapter_variant,
            "trainable_tensors": len(trainable),
            "trainable_params": sum(p.numel() for p in trainable),
            "model_owned_checkpointing": bool(model_ckpt),
        }
    )
    _print_json(rows[-1])

    print("[smoke] finalizing arena runtime")
    memory_runtime.finalize(network)
    if args.freeze_arena_residency:
        # The controlled cache matrix compares compiler behavior across cold
        # and restored processes. A cold compiler retains materially more VRAM
        # than a cache hit, which otherwise drives the live Arena policy to
        # promote a different set of blocks and changes tensor-layout guards.
        # Freeze only this diagnostic smoke's already-safe finalized plan; real
        # training retains the normal adaptive policy.
        memory_runtime._apply_training_policy = lambda *, shape_key=None: None
    discovery = _discovery_audit(memory_runtime, transformer, profile)
    runtime_diagnostics = memory_runtime.diagnostics()
    rows.append(
        {
            "event": "runtime_finalized",
            "residency_policy_frozen": bool(args.freeze_arena_residency),
            **discovery,
            "checkpoint_owner": runtime_diagnostics.get("checkpoint_owner"),
            "accounting": runtime_diagnostics.get("accounting"),
            "state_audit": runtime_diagnostics.get("state_audit"),
        }
    )
    _print_json(rows[-1])
    if not discovery["container_present"]:
        failures.append(
            f"expected block container transformer.{profile.expected_block_container} "
            "is absent"
        )
    if discovery["discovered_blocks"] in (None, 0):
        failures.append("zero execution blocks discovered by the arena runtime")
    elif (
        discovery["container_blocks"] is not None
        and discovery["discovered_blocks"] != discovery["container_blocks"]
    ):
        failures.append(
            f"discovered {discovery['discovered_blocks']} blocks but "
            f"transformer.{profile.expected_block_container} has "
            f"{discovery['container_blocks']}"
        )
    accounting = runtime_diagnostics.get("accounting") or {}
    if runtime_diagnostics.get("checkpoint_owner") != "model":
        failures.append("generic dispatcher runtime is not model-checkpoint-owned")
    if not accounting.get("payload_reconciled"):
        failures.append("canonical resident + streamed payload bytes do not reconcile")
    if (
        not accounting.get("mixed_residency")
        and not runtime_diagnostics.get("all_resident_fit")
    ):
        failures.append("production smoke did not establish mixed residency")
    if not accounting.get("protected_training_blocks_resident"):
        failures.append("model keep-last checkpoint blocks are not fully resident")
    if args.expected_residency == "mixed" and not accounting.get("mixed_residency"):
        failures.append(
            "expected mixed residency but finalized plan was not mixed: "
            f"{accounting}"
        )
    if args.expected_residency == "full" and (
        int(accounting.get("streamed_blocks", -1)) != 0
        or int(accounting.get("resident_blocks", -1)) != int(
            discovery.get("discovered_blocks", -2)
        )
        or not runtime_diagnostics.get("all_resident_fit")
    ):
        failures.append(
            "expected full residency but finalized plan was not all-resident: "
            f"{accounting}"
        )

    # The dispatcher is finalized but its compiled block kernels are still
    # lazy. Load the artifact here so restored entries exist before the first
    # model graph is created or executed.
    megacache = _activate_megacache(
        args, model, phases, device, megacache, failures
    )
    compile_cache = None
    if args.megacache_mode == "off":
        from toolkit.compile_cache import CompileCacheSession

        compile_cache = CompileCacheSession.for_model(
            model,
            model.model_config,
            default_cache_dir=args.compile_cache_dir,
            compile_enabled=not args.no_compile and not args.no_compile_cache,
            logger=lambda message: print(f"[smoke] {message}"),
        )
        compile_cache.load()

    # --- conditioning + fixed latents -------------------------------------
    embeds = profile.load_conditioning(args.cond_cache, args.batch_size)
    # Match SDTrainer: cached conditioning is stored on CPU and moved to the
    # training device/dtype immediately before the model prediction.
    embeds.to(device, dtype=model.torch_dtype)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents_cpu = profile.make_latents(resolution, args.batch_size, generator, model)
    expected_pred_shape = profile.prediction_shape_reference(latents_cpu)

    # --- phase loop --------------------------------------------------------
    print(f"[smoke] running phases: {phases}")
    nonfinite_trace = None
    trace_handles = []
    if args.trace_nonfinite:
        nonfinite_trace, trace_handles = _install_nonfinite_trace(transformer)
    step_num = 0
    steady_frames_baseline = None
    post_sample_train_seen = False
    saw_sample = False
    loss_series = []
    for index, phase in enumerate(phases):
        frames_before = _frames_total()
        transfer_before = _transfer_snapshot()
        if phase == "train":
            result = _train_phase(
                step_num,
                model,
                transformer,
                network,
                embeds,
                latents_cpu,
                generator,
                device,
                include_checksums=megacache is not None,
            )
            grads_present = sum(1 for p in trainable if p.grad is not None)
            grad_norm = torch.sqrt(
                sum(
                    p.grad.detach().float().pow(2).sum()
                    for p in trainable
                    if p.grad is not None
                )
            ).item() if grads_present else 0.0
            grad_finite = all(
                torch.isfinite(p.grad.float()).all().item()
                for p in trainable
                if p.grad is not None
            )
            grad_checksum = (
                _gradient_checksum(network) if megacache is not None else None
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            physical_free_sample = result.pop("_physical_free_monitor").stop()
            if memory_runtime is not None and physical_free_sample is not None:
                memory_runtime.record_training_physical_free_min(
                    physical_free_sample["min_free_bytes"]
                )
            result.update(
                {
                    "grad_tensors": grads_present,
                    "grad_norm": grad_norm,
                    "grad_finite": grad_finite,
                    "physical_free_sample": physical_free_sample,
                }
            )
            if grad_checksum is not None:
                result["grad_checksum"] = grad_checksum
            loss_series.append(result["loss"])
            if grads_present == 0:
                failures.append(f"phase {index} (train): no adapter gradients")
            if not grad_finite:
                failures.append(f"phase {index} (train): non-finite adapter gradient")
            if not result["pred_finite"]:
                failures.append(f"phase {index} (train): non-finite prediction")
            if not torch.isfinite(torch.tensor(result["loss"])):
                failures.append(f"phase {index} (train): non-finite loss")
            if tuple(result["pred_shape"]) != tuple(expected_pred_shape):
                failures.append(
                    f"phase {index} (train): prediction shape "
                    f"{result['pred_shape']} != latent shape {expected_pred_shape}"
                )
            if saw_sample:
                post_sample_train_seen = True
            # Same-shape steady training must not create new compile frames.
            new_frames = _new_frames_since(frames_before)
            result["new_compile_frames"] = new_frames
            if step_num == 0:
                steady_frames_baseline = True  # step 0 is the compile
            elif new_frames > 0 and steady_frames_baseline:
                failures.append(
                    f"phase {index} (train step {step_num}): {new_frames} new "
                    "compile frames on a same-shape steady step"
                )
            step_num += 1
        else:
            result = _sample_phase(
                model,
                transformer,
                network,
                embeds,
                latents_cpu,
                generator,
                device,
                include_checksums=megacache is not None,
            )
            result["new_compile_frames"] = _new_frames_since(frames_before)
            saw_sample = True
            if not result["pred_finite"]:
                failures.append(f"phase {index} (sample): non-finite prediction")
        transfer_after = _transfer_snapshot()
        transfer_delta = _counter_delta(transfer_after, transfer_before)
        result["transfer"] = transfer_delta
        phase_accounting = result.get("runtime_accounting") or {}
        planned_h2d = phase_accounting.get(
            "planned_training_h2d_bytes"
            if phase == "train"
            else "planned_forward_h2d_bytes"
        )
        if planned_h2d is not None and int(transfer_delta.get("bytes", 0)) != int(
            planned_h2d
        ):
            failures.append(
                f"phase {index} ({phase}): observed H2D bytes "
                f"{int(transfer_delta.get('bytes', 0))} != planned {int(planned_h2d)}"
            )
        row = {
            "event": "phase",
            "index": index,
            "phase": phase,
            **result,
            "cuda": _cuda_snapshot(f"phase_{index}", device),
            "dxgi": _dxgi_snapshot(f"phase_{index}"),
        }
        rows.append(row)
        if compile_cache is not None:
            compile_cache.save()
        print(
            f"[smoke] phase {index} ({phase}): {result['seconds']:.2f}s "
            + (
                f"loss={result['loss']:.4f} grads={result.get('grad_tensors')}"
                if phase == "train"
                else f"pred_norm={result['pred_norm']:.3f}"
            )
            + f" new_frames={result['new_compile_frames']}"
        )

    if saw_sample and not post_sample_train_seen:
        failures.append(
            "phase sequence never returned to training after sampling -- the "
            "train -> sample -> train transition was not proven"
        )

    megacache = _finish_megacache(args, megacache, failures)
    if compile_cache is not None:
        compile_cache.save(force=True)

    # --- teardown ----------------------------------------------------------
    from toolkit.memory_management import pin_manager

    diagnostics_before_close = memory_runtime.diagnostics()
    for handle in trace_handles:
        handle.remove()
    ledger_before = pin_manager.pinned_bytes_by_kind().get("weights", 0)
    close_memory_runtime(transformer)
    ledger_after = pin_manager.pinned_bytes_by_kind().get("weights", 0)
    runtime_after = get_memory_runtime(transformer)
    if runtime_after is not None:
        failures.append("runtime still present after close (teardown leak)")
    if ledger_after > 0:
        failures.append(
            f"pinned 'weights' ledger did not return to zero after close "
            f"({_gib(ledger_after):.3f} GiB retained)"
        )

    summary = {
        "event": "done",
        "profile": profile.name,
        "requested_qtype": args.qtype,
        "effective_qtype": effective_qtype,
        "representation": representation,
        "discovery": discovery,
        "phases": phases,
        "loss_series": loss_series,
        "megacache": megacache,
        "compile_cache_key": (
            compile_cache.key if compile_cache is not None and compile_cache.enabled else None
        ),
        "dynamo": _dynamo_counters(),
        "runtime_diagnostics_before_close": diagnostics_before_close,
        "nonfinite_trace": (
            None if nonfinite_trace is None else nonfinite_trace["first"]
        ),
        "teardown": {
            "ledger_weights_gib_before": _gib(ledger_before),
            "ledger_weights_gib_after": _gib(ledger_after),
            "runtime_present_after": runtime_after is not None,
        },
        "cuda": _cuda_snapshot("done", device),
        "dxgi": _dxgi_snapshot("done"),
        "failures": failures,
    }
    rows.append(summary)
    _print_json(summary)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(rows, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        print(f"[smoke] wrote {out}")

    if failures:
        raise SystemExit("\n".join(["SMOKE FAILURES:", *failures]))
    print(f"[smoke] {profile.name}/{args.qtype} full-model smoke passed")


if __name__ == "__main__":
    from smoke_runtime import run_locked

    sys.exit(run_locked("smoke_transformer_train_cuda", main))
