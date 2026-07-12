"""Manual CUDA smoke for Krea2 LoRA *training* steps with smart offload.

Loads the full quantized Krea2 transformer, attaches the compile-neutral
immutable runtime exactly the way load_model() does, applies a fresh LoRA
network the way BaseSDTrainProcess does, then runs a handful of fake training
steps (random latents + cached TE embeddings, flow-matching velocity loss,
backward, AdamW step) and quits. No dataset, no dataloader, no trainer process.

The two-phase lifecycle is mirrored faithfully: the runtime is prepared during
the attach (before LoRA) and finalized after the network is applied.

Example:

    venv\\Scripts\\python.exe scripts\\smoke_krea2_train_cuda.py ^
      --cond-cache "C:\\GenAI\\ai-toolkit\\output\\LA Jinx Krea the 7th\\.te_cache\\sample_007_cond.safetensors"
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

from extensions_built_in.diffusion_models.krea2.krea2 import (  # noqa: E402
    DoubleSharedModulation,
    Krea2Model,
    SimpleModulation,
)
from toolkit.basic import flush  # noqa: E402
from toolkit.config_modules import ModelConfig, NetworkConfig  # noqa: E402
from toolkit.lora_special import LoRASpecialNetwork  # noqa: E402
from toolkit.memory_management import MemoryManager, bounce_pool, dxgi_meminfo, pin_manager  # noqa: E402
from toolkit.prompt_utils import PromptEmbeds  # noqa: E402
from toolkit.util.quantize import quantize_model  # noqa: E402
from scripts.smoke_runtime import add_contention_args, fail_if_vram_contended  # noqa: E402

GIB = 1024 ** 3
DEFAULT_COND_CACHE = (
    r"C:\GenAI\ai-toolkit\output\LA Jinx Krea the 7th"
    r"\.te_cache\sample_007_cond.safetensors"
)


def _gib(value):
    if value is None:
        return None
    return float(value) / GIB


def _cuda_snapshot(label, device):
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


def _arena_summary(transformer):
    """Ticket 534ea49 Phase 2 Slice E: arena presence/size instrumentation."""
    arena = getattr(transformer, "_mm_weight_arena", None)
    if arena is None:
        return {"present": False}
    stats = arena.stats()
    return {
        "present": True,
        "id": id(arena),
        "blocks": stats.blocks,
        "pinned_gib": _gib(stats.pinned_bytes),
        "pageable_blocks": stats.pageable_blocks,
        "pageable_gib": _gib(stats.pageable_bytes),
        "ledger_weights_gib": _gib(pin_manager.pinned_bytes_by_kind().get("weights", 0)),
    }


def _immutable_arena_summary(transformer):
    """Slice 6 diagnostics for the canonical immutable-arena backend: arena
    identity, host bytes, GPU sidecar bytes, the active residency-plan
    fingerprint, transfer-plan range/copy counts, and the pin-ledger 'weights'
    tier. Reports ``{"present": False}`` when the transformer never built one,
    so it can sit alongside ``_arena_summary`` (only one is present per run)."""
    arena = getattr(transformer, "_mm_canonical_arena", None)
    if arena is None:
        return {"present": False}
    stats = arena.stats()
    residency = getattr(transformer, "_mm_residency_state", None)
    runtime = getattr(transformer, "_immutable_runtime", None)
    summary = {
        "present": True,
        "id": id(arena),
        "blocks": stats.blocks,
        "pinned_gib": _gib(stats.pinned_bytes),
        "ledger_weights_gib": _gib(
            pin_manager.pinned_bytes_by_kind().get("weights", 0)
        ),
    }
    if residency is not None:
        summary["resident_sidecar_gib"] = _gib(residency.resident_bytes())
        summary["active_plan_fingerprint"] = residency.plan.fingerprint
    train_plan = getattr(transformer, "_mm_immutable_training_plan", None)
    if train_plan is not None:
        summary["training_plan_fingerprint"] = train_plan.fingerprint
    # Transfer-plan (compact multi-range) span/copy accounting per Invariant 8.
    if residency is not None and runtime is not None and residency.plan is not None:
        from toolkit.memory_management.immutable_runtime import build_source_snapshot

        streamed_blocks = 0
        total_ranges = 0
        streamed_bytes = 0
        for abi in runtime._block_abis:
            snapshot = build_source_snapshot(residency, residency.plan, abi)
            if snapshot.transfer is None:
                continue
            streamed_blocks += 1
            total_ranges += snapshot.transfer.num_ranges
            streamed_bytes += int(snapshot.transfer.compact_nbytes)
        summary["streamed_blocks"] = streamed_blocks
        # One coalesced copy per range; the runtime submits ranges as copies.
        summary["transfer_ranges"] = total_ranges
        summary["transfer_compact_gib"] = _gib(streamed_bytes)
    if runtime is not None:
        summary["runtime_stats"] = dict(runtime.stats)
    return summary


def _dxgi_snapshot(label, device_index=0):
    info = dxgi_meminfo.query_non_local_video_memory_info(
        cuda_device_index=device_index,
        min_interval_s=0.0,
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


def _print_json(row):
    print(json.dumps(row, indent=2, sort_keys=True))


def _dynamo_counters():
    try:
        counters = torch._dynamo.utils.counters
        return {
            "graph_breaks": sum(counters["graph_break"].values()),
            "frames": dict(counters["frames"]),
            "stats": dict(counters["stats"]),
        }
    except Exception as error:
        return {"error": repr(error)}


def _load_prompt_cache(path: str):
    embeds = PromptEmbeds.load(path)
    text_embeds = getattr(embeds, "text_embeds", None)
    if text_embeds is None:
        raise ValueError(f"{path} did not contain text_embeds")
    if isinstance(text_embeds, torch.Tensor):
        if text_embeds.dim() == 2:
            embeds.text_embeds = [text_embeds]
        elif text_embeds.dim() == 3 and text_embeds.shape[0] == 1:
            embeds.text_embeds = [text_embeds[0]]
        else:
            raise ValueError(
                f"{path} has unsupported text_embeds shape {tuple(text_embeds.shape)}"
            )
    elif isinstance(text_embeds, tuple):
        embeds.text_embeds = list(text_embeds)
    elif not isinstance(text_embeds, list):
        raise ValueError(f"{path} has unsupported text_embeds type {type(text_embeds)!r}")
    return embeds


def _parse_compile_dynamic(value):
    if value == "none":
        return None
    return value == "true"


def _parse_dynamic_hints(specs):
    """Parse repeated --compile-mark-dynamic DIM:MIN:MAX into hint tuples.

    MIN/MAX may be empty (unbounded on that side), e.g. "1:256:" or "1::4096".
    """
    hints = []
    for spec in specs or ():
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(
                f"--compile-mark-dynamic expects DIM:MIN:MAX, got {spec!r}"
            )
        dim_s, lo_s, hi_s = parts
        hints.append(
            (int(dim_s), int(lo_s) if lo_s else None, int(hi_s) if hi_s else None)
        )
    return tuple(hints)


def _parse_resolutions(spec, default_width, default_height):
    """Parse --resolutions "WxH,WxH,..." into a list of (width, height) buckets.

    Empty/None falls back to the single (default_width, default_height) bucket
    so existing single-resolution invocations are unaffected.
    """
    if not spec:
        return [(default_width, default_height)]
    buckets = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        w_s, _, h_s = part.partition("x")
        if not h_s:
            raise SystemExit(f"--resolutions expects WxH pairs, got {part!r}")
        buckets.append((int(w_s), int(h_s)))
    if not buckets:
        raise SystemExit(f"--resolutions parsed to no buckets: {spec!r}")
    return buckets


def _build_model_config(args):
    model_kwargs = {
        "max_text_length": args.max_text_length,
        "local_files_only": not args.allow_download,
    }
    if args.cache_dir:
        model_kwargs["quantized_transformer_cache_dir"] = args.cache_dir
    if args.checkpoint_filename:
        model_kwargs["checkpoint_filename"] = args.checkpoint_filename

    return ModelConfig(
        name_or_path=args.model_path,
        arch="krea2",
        dtype=args.dtype,
        quantize=True,
        qtype=args.qtype,
        layer_offloading=True,
        layer_offloading_smart=True,
        layer_offloading_pinned_weight_gb=args.pinned_weight_gib,
        layer_offloading_smart_working_reserve_gb=args.working_reserve_gib,
        layer_offloading_smart_wddm_margin_gb=args.wddm_margin_gib,
        layer_offloading_smart_wddm_hard_gb=args.wddm_hard_gib,
        layer_offloading_wddm_spill_reserve_pct=args.spill_reserve_pct,
        layer_offloading_checkpoint_keep_last=args.checkpoint_keep_last,
        layer_offloading_block_stream_only=args.block_stream_only,
        layer_offloading_fp8_forward=args.fp8_training_forward,
        layer_offloading_fp8_grad_input=args.fp8_grad_input,
        layer_offloading_pinned_arena=args.pinned_arena,
        # The immutable runtime is built during load_model, so its ring depth
        # and compile gate arrive through the config.
        layer_offloading_prefetch_depth=args.prefetch_depth,
        layer_offloading_simulated_vram_gb=getattr(args, "simulated_vram_gib", 0.0),
        layer_offloading_wddm_cap_strict=getattr(args, "wddm_cap_strict", False),
        compile=not args.no_compile,
        compile_dynamic=_parse_compile_dynamic(args.compile_dynamic),
        compile_dynamic_hints=_parse_dynamic_hints(args.compile_mark_dynamic),
        model_kwargs=model_kwargs,
    )


def _apply_lora(
    model,
    transformer,
    device,
    rank,
    alpha,
    *,
    adapter_variant="lora",
    full_if_contains=None,
    lokr_factor=-1,
    old_lokr_format=False,
):
    # Mirror BaseSDTrainProcess's network setup order: freeze the base first
    # (streamed backward computes + stages base grads for any param that still
    # requires grad), then construct, force_to, _update_torch_multiplier,
    # apply_to, prepare_grad_etc, optimizer params.
    transformer.requires_grad_(False)
    transformer.train()
    network_type = "lora" if adapter_variant == "full" else adapter_variant
    network_config = NetworkConfig(
        type=network_type,
        linear=rank,
        linear_alpha=alpha,
        transformer_only=True,
        lokr_factor=lokr_factor,
        old_lokr_format=old_lokr_format,
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
        full_if_contains=(full_if_contains or []),
        transformer_only=True,
        is_transformer=True,
        target_lin_modules=model.target_lora_modules,
        base_model=model,
    )
    network.force_to(device, dtype=torch.float32)
    model.network = network
    network._update_torch_multiplier()
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    network.can_merge_in = False  # quantized + offloading
    network.prepare_grad_etc(None, transformer)
    network.enable_gradient_checkpointing()
    return network


def add_adapter_args(parser):
    """Add adapter compatibility controls shared by both CUDA harnesses."""
    parser.add_argument(
        "--adapter-variant",
        choices=("lora", "lokr", "dora", "full"),
        default="lora",
        help="adapter construction exercised by the smoke (default: lora)",
    )
    parser.add_argument(
        "--full-if-contains",
        action="append",
        default=None,
        metavar="TEXT",
        help=(
            "target-name substring converted to FullModule; repeat for multiple "
            "targets. With --adapter-variant full, defaults to attn.wq on block 0."
        ),
    )
    parser.add_argument(
        "--lokr-factor",
        type=int,
        default=-1,
        help="LoKr factorization factor (default: automatic)",
    )
    parser.add_argument(
        "--old-lokr-format",
        action="store_true",
        help="exercise the legacy LoKr naming/layout mode",
    )


def adapter_options(args):
    full_targets = args.full_if_contains
    if args.adapter_variant == "full" and not full_targets:
        full_targets = ["blocks.0.attn.wq"]
    return {
        "adapter_variant": args.adapter_variant,
        "full_if_contains": full_targets,
        "lokr_factor": args.lokr_factor,
        "old_lokr_format": args.old_lokr_format,
    }


def _named_trainable(network):
    """LoRA params in native fp32, keyed by their qualified names.

    Divergence dumps (ticket fce0b45) must bypass save_weights entirely: it
    casts to save_dtype (typically bf16) and would put a rounding floor under
    every metric.
    """
    return [(n, p) for n, p in network.named_parameters() if p.requires_grad]


def _clone_cpu(named):
    return {n: p.detach().float().cpu().clone() for n, p in named}


def _clone_grads_cpu(named):
    return {
        n: p.grad.detach().float().cpu().clone()
        for n, p in named
        if p.grad is not None
    }


def _optim_state_cpu(optimizer):
    def to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().clone()
        if isinstance(obj, dict):
            return {k: to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_cpu(v) for v in obj]
        return obj

    return to_cpu(optimizer.state_dict())


def _make_fixed_batch(seed, args, device, torch_dtype):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    lat_h, lat_w = args.height // 8, args.width // 8
    latents = torch.randn(
        args.batch_size, 16, lat_h, lat_w, generator=gen
    ).to(device, torch_dtype)
    noise = torch.randn(
        args.batch_size, 16, lat_h, lat_w, generator=gen
    ).to(device, torch_dtype)
    timestep = (torch.rand(args.batch_size, generator=gen) * 1000.0).to(device)
    t_frac = (timestep.float() / 1000.0).view(-1, 1, 1, 1).to(device)
    noisy = ((1.0 - t_frac) * latents.float() + t_frac * noise.float()).to(torch_dtype)
    return noisy, timestep


def _run_fixed_eval(model, transformer, network, embeds, noisy, timestep, device):
    """One no-grad forward on the fixed eval batch, in BOTH eval-forward modes.

    bf16 eval isolates accumulated weight divergence (forward arm-invariant
    given the weights); fp8 eval answers the train/inference-consistency
    question. Forward mode is flipped by toggling the manager's
    _fp8_training_requested flag and refreshing the per-linear markers -- the
    exact seam _restore_offload uses -- then restored to the arm's own mode.
    """
    mm = getattr(transformer, "_memory_manager", None)
    if mm is None:
        raise SystemExit("fixed eval requires the smart memory manager attached")
    fp8_capable = torch.cuda.get_device_capability(device) >= (8, 9)
    original = bool(getattr(mm, "_fp8_training_requested", False))
    outs = {}
    try:
        for mode in ("bf16", "fp8"):
            mm._fp8_training_requested = (mode == "fp8") and fp8_capable
            MemoryManager._refresh_training_fp8_flags(transformer, mm)
            with torch.no_grad(), network:
                pred = model.get_noise_prediction(noisy, timestep, embeds)
            outs[mode] = pred.detach().float().cpu().clone()
    finally:
        mm._fp8_training_requested = original
        MemoryManager._refresh_training_fp8_flags(transformer, mm)
    return outs


def _dump_horizon(dump_dir, completed_steps, named, optimizer, grads, eval_outs):
    payload = {
        "step": completed_steps,
        "lora": _clone_cpu(named),
        "grads": grads if grads is not None else {},
        "optim": _optim_state_cpu(optimizer),
        "eval": eval_outs,
    }
    path = Path(dump_dir) / f"horizon_{completed_steps:04d}.pt"
    torch.save(payload, path)
    print(f"[smoke] dumped horizon {completed_steps} -> {path}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Manual full-CUDA Krea2 smoke for fake LoRA training steps with "
            "smart offload. Do not hook this up to pytest."
        )
    )
    parser.add_argument("--cond-cache", default=DEFAULT_COND_CACHE)
    parser.add_argument("--model-path", default="krea/Krea-2-Raw")
    parser.add_argument("--checkpoint-filename", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--qtype", default="float8")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--resolutions",
        default=None,
        help=(
            "comma-separated WxH buckets cycled round-robin across training "
            "steps, e.g. '512x512,768x512,512x768,896x640' -- simulates a "
            "multi-bucket dataset feeding the same run. Unset = single bucket "
            "from --width/--height (unchanged behaviour). The fixed eval batch "
            "(--dump-dir) still uses --width/--height regardless."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    add_adapter_args(parser)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--working-reserve-gib", default="-1")
    parser.add_argument("--wddm-margin-gib", type=float, default=1.0)
    parser.add_argument("--wddm-hard-gib", type=float, default=1.0)
    parser.add_argument(
        "--simulated-vram-gib",
        type=float,
        default=0.0,
        help=(
            "Pretend the card has this many GiB (e.g. 8 or 6) to validate "
            "small-card residency/streaming/OOM behaviour. 0 = real card."
        ),
    )
    parser.add_argument(
        "--wddm-cap-strict",
        action="store_true",
        help=(
            "Violating the allocator cap raises instead of widening it. Off in "
            "production; on when the OOM is the thing being measured."
        ),
    )
    parser.add_argument("--spill-reserve-pct", type=float, default=0.20)
    parser.add_argument("--pinned-weight-gib", type=float, default=-1.0)
    parser.add_argument(
        "--pinned-arena", action="store_true",
        help=(
            "Ticket 534ea49: pin offloaded weights once into a persistent "
            "per-block flat arena instead of re-pinning them at every "
            "sampling boundary."
        ),
    )
    parser.add_argument("--checkpoint-keep-last", type=int, default=0)
    parser.add_argument("--block-stream-only", action="store_true")
    parser.add_argument("--fp8-training-forward", action="store_true")
    parser.add_argument(
        "--fp8-grad-input", action="store_true",
        help="enable the fp8 grad-input backward gate "
        "(mirrors ModelConfig.layer_offloading_fp8_grad_input); ticket fce0b45",
    )
    parser.add_argument(
        "--dump-dir", default=None,
        help="divergence-experiment dump dir (ticket fce0b45): writes meta.json, "
        "horizon_NNNN.pt (LoRA/optim/grads/fixed-eval, native fp32), loss_series.json",
    )
    parser.add_argument(
        "--dump-horizons", default="0,1,5,10,25,50,100",
        help="comma-separated step counts at which to dump state (0 = pre-training)",
    )
    parser.add_argument(
        "--eval-seed", type=int, default=1234,
        help="seed for the fixed eval batch (identical across arms, distinct from --seed)",
    )
    parser.add_argument(
        "--init-lora", default=None,
        help="torch.save'd {name: tensor} dict to load into the LoRA before step 0 "
        "(warm-branch arms); names must match network.named_parameters()",
    )
    parser.add_argument(
        "--init-optim", default=None,
        help="torch.save'd optimizer state_dict to load before step 0",
    )
    parser.add_argument(
        "--prefetch-depth", type=int, default=2,
        help="how many blocks ahead the immutable runtime issues H2D fetches",
    )
    parser.add_argument(
        "--compile-stance",
        choices=("default", "eager_then_compile", "aot_eager_then_compile"),
        default="default",
        help=(
            "torch.compiler.set_stance() around each step's forward+backward. "
            "'eager_then_compile' runs the FIRST call for a given shape eager "
            "and defers the compile to that shape's second occurrence -- but "
            "CRASHES here: gradient checkpointing recomputes forward during "
            "backward, so the recompute (a 'second call') takes the compiled "
            "branch while the original forward took the eager branch, and "
            "checkpoint's saved-tensor-count check throws CheckpointError. "
            "'aot_eager_then_compile' is torch's fix for exactly this case "
            "(AOT-eager first invocation, checkpoint-compatible)."
        ),
    )
    parser.add_argument(
        "--blocking-h2d-timing",
        action="store_true",
        help="A/B control: settle each fetch's H2D timing inside fetch_wait "
        "(the old behaviour, which pins the host to the transfer stream once "
        "per fetch)",
    )
    parser.add_argument(
        "--ab-h2d-parity",
        action="store_true",
        help="interleaved A/B: blocking H2D timing on even steps, lazy on odd, "
        "within ONE run. Same clocks, same thermals, same allocator state -- "
        "the only way to compare arms without run-to-run variance swamping the "
        "effect. Reports a per-arm mean over the steady steps.",
    )
    parser.add_argument(
        "--trace",
        type=int,
        default=0,
        help="chrome-trace the last N steady steps (step 0 is the compile, never "
        "traced). Profiling inflates host launch cost, so the summary reports "
        "traced vs untraced step time -- read the timeline for structure and "
        "distrust its absolute numbers if the inflation is large.",
    )
    parser.add_argument(
        "--trace-out",
        default=".codex/krea2_train_trace.json",
        help="chrome-trace output path for --trace (open in chrome://tracing "
        "or ui.perfetto.dev)",
    )
    parser.add_argument(
        "--compile-cache-dir",
        default="tmp/torch_compile_cache",
        help="mega-cache dir for torch.compile artifacts (warm start turns "
        "the ~2.5 min cold trunk compile into seconds); empty string disables",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="run the immutable trunk eager (no torch.compile of block kernels)",
    )
    parser.add_argument(
        "--compile-dynamic",
        choices=("true", "false", "none"),
        default="true",
        help=(
            "dynamic= passed to the block-kernel torch.compile call. 'true'/"
            "'false' pin the shape assumption; 'none' lets Dynamo infer "
            "dynamism automatically (starts static, widens after the first "
            "shape-guard recompile). Ignored when --no-compile is set."
        ),
    )
    parser.add_argument(
        "--compile-mark-dynamic",
        action="append",
        default=None,
        metavar="DIM:MIN:MAX",
        help=(
            "explicit torch._dynamo.mark_dynamic(x, DIM, min=MIN, max=MAX) hint "
            "on the per-block hidden-state tensor, applied every block_fn call "
            "before it crosses the compile boundary. MIN/MAX may be empty for "
            "unbounded (e.g. '1:256:' or '1::4096'). Repeatable. x is "
            "[batch, seq_len, hidden] for Krea2's single-stream MMDiT blocks, "
            "so dim 1 is the resolution-dependent sequence length -- pass "
            "--compile-mark-dynamic 1:<min_seq>:<max_seq> to pre-declare the "
            "range your --resolutions buckets span instead of relying on "
            "Dynamo to discover it via recompiles."
        ),
    )
    add_contention_args(parser)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.compile_stance != "default":
        from toolkit.compile_cache import compiler_stance_supported

        if not compiler_stance_supported(args.compile_stance):
            raise SystemExit(
                f"--compile-stance {args.compile_stance!r} is not supported by "
                f"this torch build (torch {torch.__version__}). It's a recent/"
                f"experimental torch.compiler stance; older installs silently "
                f"accept the string and only fail on the first compiled call "
                f"mid-run, so this checks upfront instead."
            )
    fail_if_vram_contended(
        args.device,
        ignore_contention=args.ignore_contention,
    )
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("AI_TOOLKIT_MEMORY_DIAGNOSTICS", "1")

    # Seed the GLOBAL rng, not just the noise generator below: LoRA A/B are
    # initialized from it, and they feed the loss from step 0. Without this the
    # harness drifts run to run (observed: 6.6161 vs 6.6133 for identical math),
    # which is exactly the signal the loss-parity checks read.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("This smoke is CUDA-only; pass --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is false")

    cond_path = Path(args.cond_cache)
    if not cond_path.exists():
        raise FileNotFoundError(cond_path)
    embeds = _load_prompt_cache(str(cond_path))
    # Replicate the single cached caption across the fake batch.
    embeds.text_embeds = [embeds.text_embeds[0]] * args.batch_size

    config = _build_model_config(args)
    rows = []

    print("[smoke] constructing Krea2Model without text encoder")
    model = Krea2Model(device=args.device, model_config=config, dtype=args.dtype)
    model.skip_te = True
    rows.append(
        {
            "event": "start",
            "cuda": _cuda_snapshot("start", device),
            "dxgi": _dxgi_snapshot("start"),
        }
    )
    _print_json(rows[-1])

    print("[smoke] loading full Krea2 transformer")
    t0 = time.perf_counter()
    transformer = model._load_transformer()
    rows.append(
        {"event": "loaded_transformer", "seconds": time.perf_counter() - t0}
    )
    _print_json(rows[-1])
    flush(garbage_collect=False)

    if config.quantize and not getattr(model, "_transformer_quantized_during_load", False):
        print("[smoke] quantizing transformer")
        t0 = time.perf_counter()
        quantize_model(model, transformer)
        flush()
        rows.append(
            {"event": "quantized_transformer", "seconds": time.perf_counter() - t0}
        )
        _print_json(rows[-1])

    print("[smoke] attaching smart training memory manager")
    t0 = time.perf_counter()
    # Mirror Krea2Model.load_model(): canonicalize the frozen base into the
    # arena, build residency/plan, and prepare the (unfinalized) runtime. The
    # permanent programs are finalized AFTER LoRA apply, below.
    ignore_modules = [
        module
        for module in transformer.modules()
        if isinstance(module, (SimpleModulation, DoubleSharedModulation))
    ]
    model._attach_immutable_training_memory(transformer, ignore_modules)
    if getattr(transformer, "_memory_manager", None) is not None:
        MemoryManager._attach_prefetch_pool(transformer, device)
    model.model = transformer
    rows.append(
        {
            "event": "attached_training_memory",
            "seconds": time.perf_counter() - t0,
            "arena": _arena_summary(transformer),
            "immutable_arena": _immutable_arena_summary(transformer),
            "cuda": _cuda_snapshot("attached_training_memory", device),
            "dxgi": _dxgi_snapshot("attached_training_memory"),
        }
    )
    _print_json(rows[-1])

    print(
        f"[smoke] applying {args.adapter_variant} network "
        f"(rank={args.lora_rank})"
    )
    t0 = time.perf_counter()
    network = _apply_lora(
        model,
        transformer,
        device,
        args.lora_rank,
        args.lora_alpha,
        **adapter_options(args),
    )
    trainable = [p for p in network.parameters() if p.requires_grad]
    params = network.prepare_optimizer_params(
        text_encoder_lr=args.lr, unet_lr=args.lr, default_lr=args.lr
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    rows.append(
        {
            "event": "applied_adapter",
            "adapter_variant": args.adapter_variant,
            "full_if_contains": adapter_options(args)["full_if_contains"],
            "seconds": time.perf_counter() - t0,
            "trainable_tensors": len(trainable),
            "trainable_params": sum(p.numel() for p in trainable),
            "cuda": _cuda_snapshot("applied_lora", device),
        }
    )
    _print_json(rows[-1])

    compile_cache_key = None
    if args.compile_cache_dir:
        from extensions_built_in.diffusion_models.krea2.krea2 import _train_compile_cache_key
        from toolkit.compile_cache import load_compile_cache

        compile_cache_key = _train_compile_cache_key(model)
        if load_compile_cache(args.compile_cache_dir, compile_cache_key):
            print(f"[smoke] loaded torch.compile mega-cache ({compile_cache_key})")

    # Mirror BaseSDTrainProcess: finalize the permanent programs and activate
    # the TRAIN plan AFTER LoRA apply, so they capture the adapter leaves.
    print("[smoke] finalizing immutable runtime")
    t0 = time.perf_counter()
    training_plan = transformer._mm_immutable_training_plan
    runtime = transformer._immutable_runtime
    if runtime is None:
        raise SystemExit(
            "load_model did not prepare an immutable runtime "
            "(_immutable_runtime is unset)"
        )
    transformer.finalize_immutable_runtime()
    runtime.activate(runtime.TRAIN, training_plan)
    rows.append(
        {
            "event": "immutable_runtime_finalized",
            "seconds": time.perf_counter() - t0,
            "depth": int(runtime.depth),
            "compiled": bool(runtime.compile_blocks),
            "immutable_arena": _immutable_arena_summary(transformer),
            "cuda": _cuda_snapshot("immutable_runtime_finalized", device),
            "dxgi": _dxgi_snapshot("immutable_runtime_finalized"),
        }
    )
    _print_json(rows[-1])

    # --- fp8 backward divergence experiment wiring (ticket fce0b45) ---
    named = _named_trainable(network)
    if args.init_lora:
        init_state = torch.load(args.init_lora, map_location="cpu", weights_only=True)
        loaded = 0
        with torch.no_grad():
            for n, p in named:
                if n not in init_state:
                    raise SystemExit(f"--init-lora missing param {n}")
                p.copy_(init_state[n].to(p.device, p.dtype))
                loaded += 1
        print(f"[smoke] loaded {loaded} LoRA tensors from {args.init_lora}")
    if args.init_optim:
        optim_state = torch.load(args.init_optim, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(optim_state)
        print(f"[smoke] loaded optimizer state from {args.init_optim}")

    if args.blocking_h2d_timing:
        from toolkit.memory_management import ingraph_stream

        ingraph_stream.set_h2d_timing_blocking(True)
        print("[smoke] H2D timing: BLOCKING (old behaviour, A/B control)")

    MemoryManager.set_fp8_grad_input_enabled(bool(args.fp8_grad_input))
    rows.append(
        {
            "event": "fp8_gates",
            "fp8_forward": bool(args.fp8_training_forward),
            "fp8_grad_input": bool(args.fp8_grad_input),
        }
    )
    _print_json(rows[-1])

    dump_dir = None
    horizons = set()
    eval_batch = None
    if args.dump_dir:
        dump_dir = Path(args.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        horizons = {int(h) for h in args.dump_horizons.split(",") if h.strip() != ""}
        meta = {
            "arm": {
                "fp8_forward": bool(args.fp8_training_forward),
                "fp8_grad_input": bool(args.fp8_grad_input),
            },
            "seed": args.seed,
            "eval_seed": args.eval_seed,
            "steps": args.steps,
            "horizons": sorted(horizons),
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lr": args.lr,
            "width": args.width,
            "height": args.height,
            "resolutions": args.resolutions,
            "compile_dynamic": args.compile_dynamic,
            "compile_mark_dynamic": args.compile_mark_dynamic,
            "batch_size": args.batch_size,
            "qtype": args.qtype,
            "init_lora": args.init_lora,
            "init_optim": args.init_optim,
            "param_names": [n for n, _ in named],
        }
        (dump_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        eval_batch = _make_fixed_batch(args.eval_seed, args, device, model.torch_dtype)
        if 0 in horizons:
            eval_outs = _run_fixed_eval(
                model, transformer, network, embeds, *eval_batch, device
            )
            _dump_horizon(dump_dir, 0, named, optimizer, None, eval_outs)

    resolution_buckets = _parse_resolutions(args.resolutions, args.width, args.height)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    print(
        f"[smoke] running {args.steps} fake training steps "
        f"(batch={args.batch_size}, resolution buckets "
        f"{['x'.join(map(str, wh)) for wh in resolution_buckets]} round-robin)"
    )
    step_rows = []
    # Trace only the tail steps: step 0 is the trunk compile, and the first
    # steady step still faults fresh allocator segments, so neither is
    # representative of the steady state we are trying to see inside.
    profiler = None
    trace_from = None
    if args.trace > 0:
        trace_from = max(1, args.steps - args.trace)
        if trace_from >= args.steps:
            raise SystemExit("--trace needs at least one steady step after step 0")
    for step in range(args.steps):
        if args.ab_h2d_parity:
            from toolkit.memory_management import ingraph_stream

            ingraph_stream.set_h2d_timing_blocking(step % 2 == 0)
        if trace_from is not None and step == trace_from:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
            )
            profiler.start()
            print(f"[smoke] tracing steps {trace_from}..{args.steps - 1}")
        bucket_w, bucket_h = resolution_buckets[step % len(resolution_buckets)]
        lat_h, lat_w = bucket_h // 8, bucket_w // 8
        frames_before = torch._dynamo.utils.counters["frames"].get("total", 0)
        latents = torch.randn(
            args.batch_size, 16, lat_h, lat_w, generator=generator
        ).to(device, model.torch_dtype)
        noise = torch.randn(
            args.batch_size, 16, lat_h, lat_w, generator=generator
        ).to(device, model.torch_dtype)
        timestep = (
            torch.rand(args.batch_size, generator=generator) * 1000.0
        ).to(device)
        t_frac = (timestep.float() / 1000.0).view(-1, 1, 1, 1).to(device)
        noisy = ((1.0 - t_frac) * latents.float() + t_frac * noise.float()).to(
            model.torch_dtype
        )
        target = (noise.float() - latents.float()).detach()

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        # Mirror BaseSDTrainProcess: the immutable runtime pins its source table
        # for the duration of a step, and run() fails closed outside one.
        immutable_runtime = getattr(transformer, "_immutable_runtime", None)
        execution_context = (
            immutable_runtime.execution(immutable_runtime.TRAIN)
            if immutable_runtime is not None
            else contextlib.nullcontext()
        )
        # Backward MUST stay inside the network context (multiplier is zeroed on
        # exit; leaving before backward silently kills LoRA grads).
        compile_stance = (
            torch.compiler.set_stance(args.compile_stance)
            if args.compile_stance != "default"
            else contextlib.nullcontext()
        )
        try:
            with execution_context, network, compile_stance:
                pred = model.get_noise_prediction(noisy, timestep, embeds)
                loss = torch.nn.functional.mse_loss(pred.float(), target)
                loss.backward()
        except torch.cuda.OutOfMemoryError:
            # Same policy as BaseSDTrainProcess: an allocator-cap violation
            # widens the cap and skips the batch; strict mode re-raises.
            optimizer.zero_grad(set_to_none=True)
            if not MemoryManager.relieve_wddm_cap_after_oom(
                device, context=f"smoke step {step}"
            ):
                raise
            MemoryManager.recover_cuda_pipeline_after_oom()
            print(f"[smoke] step {step}: OOM at the cap, skipped (cap widened)")
            continue
        grad_norm = torch.sqrt(
            sum(
                p.grad.detach().float().pow(2).sum()
                for p in trainable
                if p.grad is not None
            )
        ).item()
        grads_present = sum(1 for p in trainable if p.grad is not None)
        completed = step + 1
        horizon_grads = None
        if dump_dir is not None and completed in horizons:
            # Grads captured pre-step: the raw fp32 LoRA gradients the fp8
            # grad-input hops contaminated, before Adam integrates them.
            horizon_grads = _clone_grads_cpu(named)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        if dump_dir is not None and completed in horizons:
            eval_outs = _run_fixed_eval(
                model, transformer, network, embeds, *eval_batch, device
            )
            _dump_horizon(
                dump_dir, completed, named, optimizer, horizon_grads, eval_outs
            )

        frames_after = torch._dynamo.utils.counters["frames"].get("total", 0)
        row = {
            "event": "train_step",
            "step": step,
            "bucket": f"{bucket_w}x{bucket_h}",
            "seconds": elapsed,
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "grad_tensors": f"{grads_present}/{len(trainable)}",
            "new_compile_frames": frames_after - frames_before,
            "cuda": _cuda_snapshot(f"step_{step}", device),
            "dxgi": _dxgi_snapshot(f"step_{step}"),
        }
        step_rows.append(row)
        print(
            f"[smoke] step {step} [{row['bucket']}]: {elapsed:.2f}s "
            f"loss={row['loss']:.4f} grad_norm={grad_norm:.4e} "
            f"grads={row['grad_tensors']} new_frames={row['new_compile_frames']}"
        )
        if step == 0 and compile_cache_key is not None:
            from toolkit.compile_cache import save_compile_cache

            if save_compile_cache(args.compile_cache_dir, compile_cache_key):
                print(f"[smoke] saved torch.compile mega-cache ({compile_cache_key})")
        if grads_present == 0:
            raise SystemExit("no LoRA gradients produced -- training path is broken")

    trace_path = None
    if profiler is not None:
        profiler.stop()
        trace_path = Path(args.trace_out)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))
        print(f"[smoke] wrote chrome trace {trace_path}")

    if dump_dir is not None:
        loss_series = [
            {"step": r["step"] + 1, "loss": r["loss"], "grad_norm": r["grad_norm"]}
            for r in step_rows
        ]
        (dump_dir / "loss_series.json").write_text(
            json.dumps(loss_series, indent=2), encoding="utf-8"
        )
        print(f"[smoke] wrote {dump_dir / 'loss_series.json'}")

    rows.extend(step_rows)
    steady = [r["seconds"] for r in step_rows[1:]] or [step_rows[0]["seconds"]]
    ab_summary = None
    if args.ab_h2d_parity:
        # Steady steps only; step 0 carries the compile and would land wholly in
        # one arm and bias it.
        blocking_arm = [
            r["seconds"] for r in step_rows[1:] if r["step"] % 2 == 0
        ]
        lazy_arm = [r["seconds"] for r in step_rows[1:] if r["step"] % 2 == 1]
        ab_summary = {
            "blocking_steps": len(blocking_arm),
            "blocking_mean_s": (
                sum(blocking_arm) / len(blocking_arm) if blocking_arm else None
            ),
            "lazy_steps": len(lazy_arm),
            "lazy_mean_s": sum(lazy_arm) / len(lazy_arm) if lazy_arm else None,
        }
        if blocking_arm and lazy_arm:
            ab_summary["lazy_vs_blocking_pct"] = (
                (ab_summary["lazy_mean_s"] - ab_summary["blocking_mean_s"])
                / ab_summary["blocking_mean_s"]
                * 100.0
            )
    trace_summary = None
    if trace_path is not None:
        # Quantify the observer effect: the same steady state, with and without
        # the profiler attached. A small gap means the timeline's numbers can be
        # trusted; a large one means read it for structure only.
        untraced = [r["seconds"] for r in step_rows[1:trace_from]]
        traced = [r["seconds"] for r in step_rows[trace_from:]]
        trace_summary = {
            "path": str(trace_path),
            "traced_steps": len(traced),
            "traced_step_avg_s": sum(traced) / len(traced),
            "untraced_step_avg_s": (
                sum(untraced) / len(untraced) if untraced else None
            ),
        }
        if untraced:
            base = trace_summary["untraced_step_avg_s"]
            trace_summary["profiler_overhead_pct"] = (
                (trace_summary["traced_step_avg_s"] - base) / base * 100.0
            )
    per_bucket_steady_s = {}
    for r in step_rows[1:]:
        per_bucket_steady_s.setdefault(r["bucket"], []).append(r["seconds"])
    summary = {
        "event": "done",
        "steps": args.steps,
        "resolution_buckets": [f"{w}x{h}" for w, h in resolution_buckets],
        "compile_dynamic": args.compile_dynamic,
        "compile_mark_dynamic": args.compile_mark_dynamic,
        "compile_stance": args.compile_stance,
        "steps_with_new_compile_frames": [
            {"step": r["step"], "bucket": r["bucket"], "new_frames": r["new_compile_frames"]}
            for r in step_rows
            if r["new_compile_frames"] > 0
        ],
        "per_bucket_steady_avg_s": {
            bucket: sum(secs) / len(secs) for bucket, secs in per_bucket_steady_s.items()
        },
        "ab_h2d_parity": ab_summary,
        "trace": trace_summary,
        "first_step_s": step_rows[0]["seconds"],
        "steady_step_avg_s": sum(steady) / len(steady),
        "dynamo": _dynamo_counters(),
        "ingraph_fetch_report": MemoryManager.ingraph_fetch_report(reset=True),
        "offload_profile_report": MemoryManager.offload_profile_report(reset=True),
        "cuda": _cuda_snapshot("done", device),
        "dxgi": _dxgi_snapshot("done"),
    }
    rows.append(summary)
    _print_json(summary)

    if args.pinned_arena:
        # Ticket 534ea49 Phase 2 Slice E: explicit teardown at the very end
        # (model unload, not a training/sampling boundary). The "weights"
        # ledger must return to its pre-arena baseline.
        ledger_before_destroy = pin_manager.pinned_bytes_by_kind().get("weights", 0)
        MemoryManager._destroy_pinned_arena(transformer)
        ledger_after_destroy = pin_manager.pinned_bytes_by_kind().get("weights", 0)
        teardown_row = {
            "event": "pinned_arena_destroyed",
            "ledger_weights_gib_before": _gib(ledger_before_destroy),
            "ledger_weights_gib_after": _gib(ledger_after_destroy),
            "arena_present_after": getattr(transformer, "_mm_weight_arena", None)
            is not None,
        }
        rows.append(teardown_row)
        _print_json(teardown_row)

    # True unload (a genuine model unload, not a phase boundary): the runtime
    # closes and every pinned byte returns to the pre-arena baseline.
    ledger_before_destroy = pin_manager.pinned_bytes_by_kind().get("weights", 0)
    transformer.disable_immutable_runtime()
    ledger_after_destroy = pin_manager.pinned_bytes_by_kind().get("weights", 0)
    teardown_row = {
        "event": "immutable_runtime_closed",
        "ledger_weights_gib_before": _gib(ledger_before_destroy),
        "ledger_weights_gib_after": _gib(ledger_after_destroy),
        "runtime_present_after": getattr(transformer, "_immutable_runtime", None)
        is not None,
    }
    rows.append(teardown_row)
    _print_json(teardown_row)

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[smoke] wrote {json_path}")


if __name__ == "__main__":
    # Serialize GPU scripts: two 11+ GiB smokes on a 12 GB card do not
    # just measure badly, the second OOMs. See scripts/_gpu_lock.py.
    from smoke_runtime import run_locked

    sys.exit(run_locked("smoke_krea2_train_cuda", main))
