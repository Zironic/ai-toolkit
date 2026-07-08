"""Manual CUDA smoke for Krea2 LoRA *training* steps with smart offload.

Training-side counterpart of smoke_krea2_ingraph_cuda.py: loads the full
quantized Krea2 transformer, attaches the real smart MemoryManager exactly the
way load_model() does, applies a fresh LoRA network the way BaseSDTrainProcess
does, then runs a handful of fake training steps (random latents + cached TE
embeddings, flow-matching velocity loss, backward, AdamW step) and quits. No
dataset, no dataloader, no trainer process.

This is the harness for the Phase 4 ladder (INGRAPH_STREAM_PLAN.md): today it
exercises the legacy eager streamed training path as the baseline; compile
knobs get added rung by rung as they land.

Example:

    venv\\Scripts\\python.exe scripts\\smoke_krea2_train_cuda.py ^
      --cond-cache "C:\\GenAI\\ai-toolkit\\output\\LA Jinx Krea the 7th\\.te_cache\\sample_007_cond.safetensors"
"""

from __future__ import annotations

import argparse
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
        layer_offloading_pinned_arena=args.pinned_arena,
        model_kwargs=model_kwargs,
    )


def _attach_training_memory(transformer, model_config, device, *, ingraph_training=False):
    # Mirror Krea2Model.load_model()'s smart-offload attach exactly.
    # Ingraph training pins its own packs; attach pins would double-commit.
    pinned_weight_gib = (
        0.0 if ingraph_training else model_config.layer_offloading_pinned_weight_gb
    )
    ignore_modules = [
        module
        for module in transformer.modules()
        if isinstance(module, (SimpleModulation, DoubleSharedModulation))
    ]
    keep_last = model_config.layer_offloading_checkpoint_keep_last
    pinned_resident_keys = MemoryManager.training_pinned_keys_for_keep_last(
        transformer, max(0, keep_last)
    )
    MemoryManager.attach_smart_training(
        transformer,
        device,
        working_reserve_gib=model_config.layer_offloading_smart_working_reserve_gb,
        wddm_margin_gib=model_config.layer_offloading_smart_wddm_margin_gb,
        wddm_hard_gib=model_config.layer_offloading_smart_wddm_hard_gb,
        ignore_modules=ignore_modules,
        pinned_resident_keys=pinned_resident_keys,
        block_stream_only=model_config.layer_offloading_block_stream_only,
        pinned_weight_gib=pinned_weight_gib,
        wddm_spill_reserve_pct=model_config.layer_offloading_wddm_spill_reserve_pct,
        fp8_training_forward=bool(model_config.layer_offloading_fp8_forward),
        # Mirror krea2.py: the arena and ingraph packs are two independent
        # pin grants until they're merged, so only one may be active.
        use_pinned_arena=(
            bool(model_config.layer_offloading_pinned_arena) and not ingraph_training
        ),
    )
    transformer.enable_gradient_checkpointing(keep_last=max(0, keep_last))
    if getattr(transformer, "_memory_manager", None) is not None:
        MemoryManager._attach_prefetch_pool(transformer, device)


def _apply_lora(model, transformer, device, rank, alpha):
    # Mirror BaseSDTrainProcess's network setup order: freeze the base first
    # (streamed backward computes + stages base grads for any param that still
    # requires grad), then construct, force_to, _update_torch_multiplier,
    # apply_to, prepare_grad_etc, optimizer params.
    transformer.requires_grad_(False)
    transformer.train()
    network_config = NetworkConfig(
        type="lora", linear=rank, linear_alpha=alpha, transformer_only=True
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--working-reserve-gib", default="-1")
    parser.add_argument("--wddm-margin-gib", type=float, default=1.0)
    parser.add_argument("--wddm-hard-gib", type=float, default=1.0)
    parser.add_argument("--spill-reserve-pct", type=float, default=0.20)
    parser.add_argument("--pinned-weight-gib", type=float, default=-1.0)
    parser.add_argument(
        "--pinned-arena", action="store_true",
        help=(
            "Ticket 534ea49: pin offloaded weights once into a persistent "
            "per-block flat arena instead of re-pinning them at every "
            "sampling boundary. Mutually exclusive with --ingraph-training "
            "(both pin the same weights independently until merged)."
        ),
    )
    parser.add_argument("--checkpoint-keep-last", type=int, default=0)
    parser.add_argument("--block-stream-only", action="store_true")
    parser.add_argument("--fp8-training-forward", action="store_true")
    parser.add_argument(
        "--ingraph-training",
        action="store_true",
        help="call enable_ingraph_training() after LoRA apply (Phase 4a): "
        "compiled fully-streamed training trunk, all blocks checkpointed",
    )
    parser.add_argument("--ingraph-depth", type=int, default=2)
    parser.add_argument(
        "--compile-cache-dir",
        default="tmp/torch_compile_cache",
        help="mega-cache dir for torch.compile artifacts (warm start turns "
        "the ~2.5 min cold trunk compile into seconds); empty string disables",
    )
    parser.add_argument(
        "--no-ingraph-compile",
        action="store_true",
        help="with --ingraph-training: run the eager ingraph trunk (no torch.compile)",
    )
    parser.add_argument(
        "--train-compile-blocks",
        action="store_true",
        help="call enable_compiled_training() after LoRA apply, mirroring the "
        "trainer's train_compile_blocks flag (Rung 1 / Phase 4-pre)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("AI_TOOLKIT_MEMORY_DIAGNOSTICS", "1")

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
    _attach_training_memory(
        transformer, config, device, ingraph_training=bool(args.ingraph_training)
    )
    model.model = transformer
    rows.append(
        {
            "event": "attached_training_memory",
            "seconds": time.perf_counter() - t0,
            "arena": _arena_summary(transformer),
            "cuda": _cuda_snapshot("attached_training_memory", device),
            "dxgi": _dxgi_snapshot("attached_training_memory"),
        }
    )
    _print_json(rows[-1])

    print(f"[smoke] applying LoRA network (rank={args.lora_rank})")
    t0 = time.perf_counter()
    network = _apply_lora(
        model, transformer, device, args.lora_rank, args.lora_alpha
    )
    trainable = [p for p in network.parameters() if p.requires_grad]
    params = network.prepare_optimizer_params(
        text_encoder_lr=args.lr, unet_lr=args.lr, default_lr=args.lr
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    rows.append(
        {
            "event": "applied_lora",
            "seconds": time.perf_counter() - t0,
            "trainable_tensors": len(trainable),
            "trainable_params": sum(p.numel() for p in trainable),
            "cuda": _cuda_snapshot("applied_lora", device),
        }
    )
    _print_json(rows[-1])

    compile_cache_key = None
    if args.ingraph_training and args.compile_cache_dir:
        from extensions_built_in.diffusion_models.krea2.krea2 import _compile_cache_key
        from toolkit.compile_cache import load_compile_cache

        compile_cache_key = _compile_cache_key(model) + "_ingraph_train"
        if load_compile_cache(args.compile_cache_dir, compile_cache_key):
            print(f"[smoke] loaded torch.compile mega-cache ({compile_cache_key})")

    if args.ingraph_training:
        # Mirror BaseSDTrainProcess's layer_offloading_ingraph_training hook:
        # enable after LoRA apply so entries see the final module state.
        print("[smoke] enabling in-graph streamed training trunk")
        t0 = time.perf_counter()
        pack_count = transformer.enable_ingraph_training(
            depth=args.ingraph_depth, compile=not args.no_ingraph_compile
        )
        rows.append(
            {
                "event": "ingraph_training_enabled",
                "seconds": time.perf_counter() - t0,
                "packs": pack_count,
                "depth": args.ingraph_depth,
                "compiled": not args.no_ingraph_compile,
                "lora_blocks": len(getattr(transformer, "_ingraph_training_loras", {})),
                "cuda": _cuda_snapshot("ingraph_training_enabled", device),
                "dxgi": _dxgi_snapshot("ingraph_training_enabled"),
            }
        )
        _print_json(rows[-1])

    if args.train_compile_blocks:
        # Mirror BaseSDTrainProcess's train_compile_blocks wiring: compile the
        # pinned-resident blocks for grad-enabled training (Rung 1). Must run
        # after LoRA apply so the readiness audit sees the final module state.
        print("[smoke] enabling resident-block training compile")
        t0 = time.perf_counter()
        mm = getattr(transformer, "_memory_manager", None)
        pinned_keys = set(getattr(mm, "_training_pinned_resident_keys", set()) or set())
        compiled_count, blocked_count = transformer.enable_compiled_training(pinned_keys)
        readiness = transformer.training_compile_readiness(pinned_keys)
        rows.append(
            {
                "event": "train_compile_blocks",
                "seconds": time.perf_counter() - t0,
                "compiled_blocks": compiled_count,
                "blocked_blocks": blocked_count,
                "pinned_keys": len(pinned_keys),
                "blocked_reasons": [
                    {"index": item["index"], "reasons": item.get("reasons")}
                    for item in readiness.get("statuses", [])
                    if item.get("pinned") and not item.get("ready")
                ],
            }
        )
        _print_json(rows[-1])
        if compiled_count == 0:
            raise SystemExit(
                "train-compile-blocks requested but 0 blocks compiled -- "
                "use --checkpoint-keep-last N so pinned resident blocks exist"
            )

    lat_h, lat_w = args.height // 8, args.width // 8
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    print(f"[smoke] running {args.steps} fake training steps "
          f"(batch={args.batch_size}, latents {args.batch_size}x16x{lat_h}x{lat_w})")
    step_rows = []
    for step in range(args.steps):
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
        # Backward MUST stay inside the network context (multiplier is zeroed on
        # exit; leaving before backward silently kills LoRA grads).
        with network:
            pred = model.get_noise_prediction(noisy, timestep, embeds)
            loss = torch.nn.functional.mse_loss(pred.float(), target)
            loss.backward()
        grad_norm = torch.sqrt(
            sum(
                p.grad.detach().float().pow(2).sum()
                for p in trainable
                if p.grad is not None
            )
        ).item()
        grads_present = sum(1 for p in trainable if p.grad is not None)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        row = {
            "event": "train_step",
            "step": step,
            "seconds": elapsed,
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "grad_tensors": f"{grads_present}/{len(trainable)}",
            "cuda": _cuda_snapshot(f"step_{step}", device),
            "dxgi": _dxgi_snapshot(f"step_{step}"),
        }
        step_rows.append(row)
        print(
            f"[smoke] step {step}: {elapsed:.2f}s loss={row['loss']:.4f} "
            f"grad_norm={grad_norm:.4e} grads={row['grad_tensors']}"
        )
        if step == 0 and compile_cache_key is not None:
            from toolkit.compile_cache import save_compile_cache

            if save_compile_cache(args.compile_cache_dir, compile_cache_key):
                print(f"[smoke] saved torch.compile mega-cache ({compile_cache_key})")
        if grads_present == 0:
            raise SystemExit("no LoRA gradients produced -- training path is broken")

    rows.extend(step_rows)
    steady = [r["seconds"] for r in step_rows[1:]] or [step_rows[0]["seconds"]]
    summary = {
        "event": "done",
        "steps": args.steps,
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

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[smoke] wrote {json_path}")


if __name__ == "__main__":
    main()
