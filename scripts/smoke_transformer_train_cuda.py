"""Shared multi-architecture full-model training smoke (CUDA).

Owns everything architecture-neutral: CLI parsing, deterministic seeding,
model-config/runtime wiring, fake flow-matching training steps, adapter and
optimizer setup, train -> sample -> train phase transitions, CUDA/DXGI
snapshots, compile counters, JSON output, common assertions, and teardown.
Architecture specifics (model construction, transformer loading, valid fake
inputs, conditioning validation) come from scripts/smoke_profiles.py:

    --profile krea2 | zimage | ideogram4

The default phase sequence `train,train,sample,train` proves training ->
sampling -> training survival: sample state publication, checkpoint state
staying usable, compiled training kernels surviving the transition.

KNOWN BLOCKERS (the smoke exists ahead of the work it gates; see the
multi-architecture smoke plan):
  * zimage / ideogram4: no generic arena attach exists yet; those models
    still use the legacy MemoryManager.attach. The runner fails loudly with
    a pointer to the generic-dispatcher plan.

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
import json
import os
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
    add_contention_args,
    add_lock_args,
    assert_smoke_load_mode,
    configure_smoke_load_mode,
    fail_if_vram_contended,
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

def _attach_arena_runtime(model, transformer, profile, device):
    """Attach the arena runtime the way load_model() would.

    Krea2 already exposes the integration seam. The other architectures wait
    on the generic block dispatcher (tasks/open/GENERIC_BLOCK_DISPATCHER_PLAN.md,
    tickets b7dead1 / b1a13d2); until it lands this fails loudly rather than
    silently falling back to the legacy per-Linear manager.
    """
    ignore_modules = profile.arena_ignore_modules(transformer)
    attach = getattr(model, "_attach_immutable_training_memory", None)
    if attach is not None:
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


def _parse_args():
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
    parser.add_argument(
        "--load-mode",
        choices=("direct-arena", "normal"),
        default=None,
        help=(
            "Krea2 checkpoint lifecycle; defaults to direct-arena for Krea2 "
            "and normal for other profiles"
        ),
    )
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
    parser.add_argument("--checkpoint-keep-last", type=int, default=0)
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--compile-dynamic", choices=("true", "false", "none"), default="true"
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
    args = parser.parse_args()
    if args.load_mode is None:
        args.load_mode = "direct-arena" if args.profile == "krea2" else "normal"
    if args.profile != "krea2" and args.load_mode != "normal":
        parser.error("--load-mode direct-arena is currently supported only by Krea2")
    args.compile_dynamic_resolved = (
        None if args.compile_dynamic == "none" else args.compile_dynamic == "true"
    )
    return args


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def _train_phase(
    step_num, model, transformer, network, embeds, latents_cpu, generator, device
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
    return {
        "seconds": time.perf_counter() - started,
        "loss": loss.item(),
        "pred_norm": pred.detach().float().norm().item(),
        "pred_finite": bool(torch.isfinite(pred.detach().float()).all()),
        "pred_shape": tuple(pred.shape),
        "runtime_accounting": runtime_accounting,
    }


def _sample_phase(model, transformer, network, embeds, latents_cpu, generator, device):
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
    return {
        "seconds": time.perf_counter() - started,
        "pred_norm": pred.detach().float().norm().item(),
        "pred_finite": bool(torch.isfinite(pred.detach().float()).all()),
        "pred_shape": tuple(pred.shape),
        "runtime_accounting": runtime_accounting,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    profile = PROFILES[args.profile]
    failures: list[str] = []
    rows: list[dict] = []

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
    if config.qtype != args.qtype and args.assistant_lora is None:
        raise SystemExit(
            f"requested qtype {args.qtype!r} became {config.qtype!r} in the "
            "ModelConfig -- silent qtype replacement"
        )

    print(f"[smoke] constructing {profile.name} model without text encoder")
    model = profile.construct_model(args, config)
    configure_smoke_load_mode(model, args.load_mode)
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
        quantize_model(model, transformer)
        flush()
        rows.append(
            {"event": "quantized_transformer", "seconds": time.perf_counter() - t0}
        )
        _print_json(rows[-1])

    representation = audit_quantized_representation(transformer)
    effective_qtype = model.model_config.qtype
    rep_failures = assert_representation(representation, effective_qtype)
    failures.extend(rep_failures)
    rows.append({"event": "representation", **representation})
    _print_json(rows[-1])

    model_ckpt = profile.enable_model_checkpointing(
        transformer, keep_last=args.checkpoint_keep_last
    )

    print("[smoke] attaching arena runtime")
    t0 = time.perf_counter()
    _attach_arena_runtime(model, transformer, profile, device)
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
    discovery = _discovery_audit(memory_runtime, transformer, profile)
    runtime_diagnostics = memory_runtime.diagnostics()
    rows.append(
        {
            "event": "runtime_finalized",
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
    if not accounting.get("mixed_residency"):
        failures.append("production smoke did not establish mixed residency")
    if not accounting.get("protected_training_blocks_resident"):
        failures.append("model keep-last checkpoint blocks are not fully resident")

    # --- conditioning + fixed latents -------------------------------------
    embeds = profile.load_conditioning(args.cond_cache, args.batch_size)
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
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            result.update(
                {
                    "grad_tensors": grads_present,
                    "grad_norm": grad_norm,
                    "grad_finite": grad_finite,
                }
            )
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
                model, transformer, network, embeds, latents_cpu, generator, device
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
