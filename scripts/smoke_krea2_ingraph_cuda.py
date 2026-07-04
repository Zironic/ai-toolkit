"""Manual CUDA smoke for Krea2 sampling with smart offload and compile.

This is intentionally not a pytest test. It loads the full Krea2 transformer,
attaches the real smart MemoryManager in training-auto mode, switches through
the sampling resident context, and runs one cached-TE sample with
compile_sample enabled.

Example:

    venv\\Scripts\\python.exe scripts\\smoke_krea2_ingraph_cuda.py ^
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
    Krea2Pipeline,
    SimpleModulation,
)
from toolkit.basic import flush  # noqa: E402
from toolkit.config_modules import GenerateImageConfig, ModelConfig  # noqa: E402
from toolkit.memory_management import MemoryManager, bounce_pool, dxgi_meminfo  # noqa: E402
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


def _dxgi_snapshot(label, device_index=0):
    info = dxgi_meminfo.query_non_local_video_memory_info(
        cuda_device_index=device_index,
        min_interval_s=0.0,
    )
    adapter = dxgi_meminfo.selected_adapter_info()
    row = {
        "label": label,
        "adapter": None
        if adapter is None
        else {
            "index": adapter.index,
            "description": adapter.description,
            "match_method": adapter.match_method,
            "safe_for_control": adapter.safe_for_control,
            "manual_control": adapter.manual_control,
        },
        "budget_gib": None,
        "usage_gib": None,
        "raw_headroom_gib": None,
        "spill_reserve_gib": None,
        "usable_headroom_gib": None,
    }
    if info is not None:
        reserve = bounce_pool.dxgi_spill_reserve_bytes(info.budget_bytes)
        raw_headroom = int(info.budget_bytes) - int(info.current_usage_bytes)
        row.update(
            {
                "budget_gib": _gib(info.budget_bytes),
                "usage_gib": _gib(info.current_usage_bytes),
                "raw_headroom_gib": _gib(raw_headroom),
                "spill_reserve_gib": _gib(reserve),
                "usable_headroom_gib": _gib(raw_headroom - reserve),
            }
        )
    return row


def _print_json(row):
    print(json.dumps(row, indent=2, sort_keys=True))


def _configure_compile_debug(enabled: bool):
    if not enabled:
        return {"enabled": False}
    state = {"enabled": True, "graph_breaks": False, "recompiles": False}
    try:
        torch._logging.set_logs(graph_breaks=True, recompiles=True)
        state.update({"graph_breaks": True, "recompiles": True})
    except Exception as error:
        state["set_logs_error"] = repr(error)
    try:
        torch._dynamo.reset()
        state["dynamo_reset"] = True
    except Exception as error:
        state["dynamo_reset_error"] = repr(error)
    return state


def _dynamo_counters():
    try:
        counters = torch._dynamo.utils.counters
        return {
            "graph_breaks": sum(counters["graph_break"].values()),
            "frames": dict(counters["frames"]),
            "inductor": dict(counters["inductor"]),
            "stats": dict(counters["stats"]),
        }
    except Exception as error:
        return {"error": repr(error)}


def _managed_summary(transformer):
    managed = pinned_layers = 0
    pinned_bytes = 0
    resident_candidates = 0
    for child in transformer.modules():
        if hasattr(child, "_layer_memory_manager"):
            managed += 1
            nbytes = int(getattr(child, "_mm_pinned_bytes", 0) or 0)
            if nbytes > 0:
                pinned_layers += 1
                pinned_bytes += nbytes
        elif child.__class__.__name__ in (
            "Linear",
            "LoRACompatibleLinear",
            "QLinear",
            "Conv2d",
            "LoRACompatibleConv",
            "QConv",
        ):
            resident_candidates += 1
    manager = getattr(transformer, "_memory_manager", None)
    return {
        "managed_layers": managed,
        "resident_candidate_layers": resident_candidates,
        "pinned_layers": pinned_layers,
        "pinned_weight_gib": _gib(pinned_bytes),
        "manager_pinned_gib": _gib(getattr(manager, "pinned_weight_bytes", 0)),
        "ledger_gib": _gib(bounce_pool._pinned_bytes_total),
    }


def _pool_summary(transformer):
    manager = getattr(transformer, "_memory_manager", None)
    pool = getattr(manager, "_prefetch_pool", None) if manager is not None else None
    if pool is None:
        return None
    try:
        return pool.stats(reset=False)
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


def _sibling_uncond_path(cond_path: Path) -> Path | None:
    name = cond_path.name
    if "_cond." not in name:
        return None
    candidate = cond_path.with_name(name.replace("_cond.", "_uncond."))
    return candidate if candidate.exists() else None


def _build_model_config(args):
    model_kwargs = {
        "max_text_length": args.max_text_length,
        "local_files_only": not args.allow_download,
    }
    if args.cache_dir:
        model_kwargs["quantized_transformer_cache_dir"] = args.cache_dir
    if args.disable_quant_cache:
        model_kwargs["quantized_transformer_cache"] = False
    if args.vae_path:
        model_kwargs["vae_path"] = args.vae_path
    if args.checkpoint_filename:
        model_kwargs["checkpoint_filename"] = args.checkpoint_filename
    model_kwargs.update(
        {
            "schedule_y1": args.schedule_y1,
            "schedule_y2": args.schedule_y2,
            "schedule_min_res": args.schedule_min_res,
            "schedule_max_res": args.schedule_max_res,
        }
    )
    if args.schedule_mu is not None:
        model_kwargs["schedule_mu"] = args.schedule_mu

    return ModelConfig(
        name_or_path=args.model_path,
        arch="krea2",
        dtype=args.dtype,
        quantize=True,
        qtype=args.qtype,
        compile_sample=not args.no_compile_sample,
        compile_debug=not args.no_compile_debug,
        layer_offloading=True,
        layer_offloading_smart=True,
        layer_offloading_smart_sampling=True,
        layer_offloading_compile_streamed=not args.disable_compile_streamed,
        layer_offloading_ingraph_sampling=bool(args.strict_ingraph),
        layer_offloading_ingraph_depth=args.ingraph_depth,
        layer_offloading_ingraph_stream_all=bool(args.ingraph_stream_all),
        layer_offloading_pinned_weight_gb=args.pinned_weight_gib,
        layer_offloading_smart_working_reserve_gb=args.working_reserve_gib,
        layer_offloading_smart_wddm_margin_gb=args.wddm_margin_gib,
        layer_offloading_smart_wddm_hard_gb=args.wddm_hard_gib,
        layer_offloading_wddm_spill_reserve_pct=args.spill_reserve_pct,
        layer_offloading_checkpoint_keep_last=args.checkpoint_keep_last,
        layer_offloading_block_stream_only=args.block_stream_only,
        layer_offloading_fp8_forward=args.fp8_training_forward,
        layer_offloading_fp8_sampling=args.fp8_sampling,
        layer_offloading_smart_sampling_working_reserve_gb=args.sampling_working_reserve_gib,
        layer_offloading_smart_sampling_wddm_margin_gb=args.sampling_wddm_margin_gib,
        layer_offloading_smart_sampling_wddm_hard_gb=args.sampling_wddm_hard_gib,
        model_kwargs=model_kwargs,
    )


def _attach_training_memory(transformer, model_config, device):
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
        pinned_weight_gib=model_config.layer_offloading_pinned_weight_gb,
        wddm_spill_reserve_pct=model_config.layer_offloading_wddm_spill_reserve_pct,
        fp8_training_forward=bool(model_config.layer_offloading_fp8_forward),
    )
    transformer.enable_gradient_checkpointing(keep_last=max(0, keep_last))
    if getattr(transformer, "_memory_manager", None) is not None:
        MemoryManager._attach_prefetch_pool(transformer, device)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Manual full-CUDA Krea2 smoke for cached-TE sampling with smart "
            "offload and compile. Do not hook this up to pytest."
        )
    )
    parser.add_argument("--cond-cache", default=DEFAULT_COND_CACHE)
    parser.add_argument("--uncond-cache", default=None)
    parser.add_argument("--model-path", default="krea/Krea-2-Raw")
    parser.add_argument("--checkpoint-filename", default=None)
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--qtype", default="float8")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--disable-quant-cache", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--batch-cfg", action="store_true")
    parser.add_argument("--output", default=".codex/krea2_ingraph_cuda_smoke.png")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--working-reserve-gib", default="-1")
    parser.add_argument("--wddm-margin-gib", type=float, default=1.0)
    parser.add_argument("--wddm-hard-gib", type=float, default=1.0)
    parser.add_argument("--spill-reserve-pct", type=float, default=0.20)
    parser.add_argument("--pinned-weight-gib", type=float, default=-1.0)
    parser.add_argument("--checkpoint-keep-last", type=int, default=-1)
    parser.add_argument("--block-stream-only", action="store_true")
    parser.add_argument("--fp8-training-forward", action="store_true")
    parser.add_argument("--fp8-sampling", action="store_true")
    parser.add_argument("--sampling-working-reserve-gib", default="-1")
    parser.add_argument("--sampling-wddm-margin-gib", type=float, default=-1.0)
    parser.add_argument("--sampling-wddm-hard-gib", type=float, default=1.0)
    parser.add_argument("--disable-compile-streamed", action="store_true")
    parser.add_argument("--no-compile-sample", action="store_true")
    parser.add_argument("--strict-ingraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ingraph-depth", type=int, default=2)
    parser.add_argument("--ingraph-stream-all", action="store_true")
    parser.add_argument("--no-compile-debug", action="store_true")
    parser.add_argument("--schedule-y1", type=float, default=0.5)
    parser.add_argument("--schedule-y2", type=float, default=1.15)
    parser.add_argument("--schedule-min-res", type=int, default=256)
    parser.add_argument("--schedule-max-res", type=int, default=1280)
    parser.add_argument("--schedule-mu", type=float, default=None)
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
    compile_debug_state = _configure_compile_debug(not args.no_compile_debug)

    cond_path = Path(args.cond_cache)
    if not cond_path.exists():
        raise FileNotFoundError(cond_path)
    uncond_path = Path(args.uncond_cache) if args.uncond_cache else _sibling_uncond_path(cond_path)
    conditional_embeds = _load_prompt_cache(str(cond_path))
    unconditional_embeds = _load_prompt_cache(str(uncond_path)) if uncond_path else None
    guidance_scale = args.guidance_scale
    if guidance_scale is None:
        guidance_scale = 4.5 if unconditional_embeds is not None else 0.0

    config = _build_model_config(args)
    rows = []

    print("[smoke] constructing Krea2Model without text encoder")
    model = Krea2Model(device=args.device, model_config=config, dtype=args.dtype)
    model.skip_te = True
    rows.append(
        {
            "event": "start",
            "compile_debug": compile_debug_state,
            "cuda": _cuda_snapshot("start", device),
            "dxgi": _dxgi_snapshot("start"),
        }
    )
    _print_json(rows[-1])

    print("[smoke] loading full Krea2 transformer")
    t0 = time.perf_counter()
    transformer = model._load_transformer()
    rows.append(
        {
            "event": "loaded_transformer",
            "seconds": time.perf_counter() - t0,
            "cuda": _cuda_snapshot("loaded_transformer", device),
            "dxgi": _dxgi_snapshot("loaded_transformer"),
        }
    )
    _print_json(rows[-1])
    flush(garbage_collect=False)

    if config.quantize and not getattr(model, "_transformer_quantized_during_load", False):
        print("[smoke] quantizing transformer")
        t0 = time.perf_counter()
        quantize_model(model, transformer)
        flush()
        rows.append(
            {
                "event": "quantized_transformer",
                "seconds": time.perf_counter() - t0,
                "cuda": _cuda_snapshot("quantized_transformer", device),
                "dxgi": _dxgi_snapshot("quantized_transformer"),
            }
        )
        _print_json(rows[-1])

    print("[smoke] attaching smart training memory manager")
    t0 = time.perf_counter()
    _attach_training_memory(transformer, config, device)
    model.model = transformer
    rows.append(
        {
            "event": "attached_training_memory",
            "seconds": time.perf_counter() - t0,
            "summary": _managed_summary(transformer),
            "pool": _pool_summary(transformer),
            "cuda": _cuda_snapshot("attached_training_memory", device),
            "dxgi": _dxgi_snapshot("attached_training_memory"),
        }
    )
    _print_json(rows[-1])

    print("[smoke] loading VAE")
    t0 = time.perf_counter()
    model.vae = model._load_vae()
    model.vae.to(model.vae_device_torch, dtype=model.vae_torch_dtype)
    model.pipeline = Krea2Pipeline(model)
    rows.append(
        {
            "event": "loaded_vae",
            "seconds": time.perf_counter() - t0,
            "cuda": _cuda_snapshot("loaded_vae", device),
            "dxgi": _dxgi_snapshot("loaded_vae"),
        }
    )
    _print_json(rows[-1])

    gen_config = GenerateImageConfig(
        prompt="cached TE prompt",
        width=args.width,
        height=args.height,
        seed=args.seed,
        guidance_scale=guidance_scale,
        num_inference_steps=args.steps,
        output_path=args.output,
        batch_cfg=bool(args.batch_cfg),
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)

    compile_label = "compile_sample" if config.compile_sample else "compile disabled"
    print(f"[smoke] sampling with MemoryManager.inference_resident and {compile_label}")
    transformer._ingraph_sampling_measure = bool(config.compile_sample)
    t0 = time.perf_counter()
    with MemoryManager.inference_resident(
        transformer,
        device=device,
        fp8_sampling=bool(config.layer_offloading_fp8_sampling),
        working_reserve_gib=config.layer_offloading_smart_sampling_working_reserve_gb,
        wddm_margin_gib=config.layer_offloading_smart_sampling_wddm_margin_gb,
        wddm_hard_gib=config.layer_offloading_smart_sampling_wddm_hard_gb,
        reserve_pin_for_ingraph=bool(
            config.layer_offloading_compile_streamed
            or config.layer_offloading_ingraph_sampling
        ),
    ):
        image = model.generate_single_image(
            model.pipeline,
            gen_config,
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            generator=generator,
            extra={},
        )
        compile_state = getattr(transformer, "_last_ingraph_sampling_compile_state", None)
        if compile_state is None:
            compile_state = {
                "ingraph_compiled": getattr(transformer, "_compiled_ingraph_sampling", None)
                is not None,
                "regional_compiled_blocks": sum(
                    1 for block in (getattr(transformer, "_compiled_blocks", None) or [])
                    if block is not None
                ),
                "ingraph_packs": len(getattr(transformer, "_ingraph_sampling_packs", {}) or {}),
                "unavailable_reasons": tuple(
                    getattr(transformer, "_ingraph_unavailable_reasons", ()) or ()
                ),
            }
        ingraph_timing = getattr(transformer, "_last_ingraph_sampling_timing", None)
        fetch_report = MemoryManager.ingraph_fetch_report(reset=True)
        profile_report = MemoryManager.offload_profile_report(reset=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    rows.append(
        {
            "event": "sampled",
            "seconds": time.perf_counter() - t0,
            "output": str(out),
            "cond_cache": str(cond_path),
            "uncond_cache": None if uncond_path is None else str(uncond_path),
            "guidance_scale": guidance_scale,
            "compile": compile_state,
            "dynamo": _dynamo_counters(),
            "ingraph_fetch_report": fetch_report,
            "ingraph_timing": ingraph_timing,
            "offload_profile_report": profile_report,
            "cuda": _cuda_snapshot("sampled", device),
            "dxgi": _dxgi_snapshot("sampled"),
        }
    )
    _print_json(rows[-1])

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[smoke] wrote {json_path}")

    print(f"[smoke] wrote image {out}")


if __name__ == "__main__":
    main()
