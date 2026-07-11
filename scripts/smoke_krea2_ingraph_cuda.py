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


def _compile_path_snapshot(transformer):
    state = getattr(transformer, "_last_ingraph_sampling_compile_state", None)
    if state is None:
        state = {
            "ingraph_compiled": getattr(transformer, "_compiled_ingraph_sampling", None)
            is not None,
            "ingraph_compiled_blocks": len(
                getattr(transformer, "_compiled_ingraph_sampling_blocks", {}) or {}
            ),
            "regional_compiled_blocks": sum(
                1 for block in (getattr(transformer, "_compiled_blocks", None) or [])
                if block is not None
            ),
            "ingraph_packs": len(getattr(transformer, "_ingraph_sampling_packs", {}) or {}),
            "unavailable_reasons": tuple(
                getattr(transformer, "_ingraph_unavailable_reasons", ()) or ()
            ),
        }
    state = dict(state)
    state["unavailable_reasons"] = list(state.get("unavailable_reasons") or ())
    total_blocks = len(getattr(transformer, "blocks", []) or [])
    state["total_blocks"] = total_blocks
    ingraph_packs = int(state.get("ingraph_packs", 0) or 0)
    ingraph_compiled_blocks = int(state.get("ingraph_compiled_blocks", 0) or 0)
    regional_compiled_blocks = int(state.get("regional_compiled_blocks", 0) or 0)
    stream_expected_blocks = ingraph_packs
    resident_expected_blocks = max(0, total_blocks - ingraph_packs)
    stream_compile_complete = bool(state.get("ingraph_compiled")) or (
        stream_expected_blocks > 0
        and ingraph_compiled_blocks == stream_expected_blocks
    )
    resident_compile_complete = regional_compiled_blocks == resident_expected_blocks
    mixed_compile_complete = bool(
        stream_expected_blocks > 0
        and stream_compile_complete
        and resident_compile_complete
    )
    state["stream_expected_blocks"] = stream_expected_blocks
    state["resident_expected_blocks"] = resident_expected_blocks
    state["stream_compile_complete"] = stream_compile_complete
    state["resident_compile_complete"] = resident_compile_complete
    state["mixed_compile_complete"] = mixed_compile_complete
    if state.get("ingraph_compiled"):
        if total_blocks and ingraph_packs == total_blocks:
            path = "ingraph_fullgraph_all"
        else:
            path = "ingraph_fullgraph_partial"
    elif mixed_compile_complete:
        path = "mixed_resident_ingraph_compile"
    elif ingraph_compiled_blocks:
        path = "ingraph_block_compile"
    elif regional_compiled_blocks:
        path = "regional_compile"
    elif "partial_ingraph_compile" in state["unavailable_reasons"] and ingraph_packs:
        path = "ingraph_partial_eager"
    elif state["unavailable_reasons"]:
        path = "ingraph_unavailable"
    else:
        path = "eager_or_not_compiled"
    state["path"] = path
    return state


def _ingraph_fetch_count(report):
    if not report:
        return None
    marker = "fetches="
    start = str(report).find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = start
    text = str(report)
    while end < len(text) and text[end].isdigit():
        end += 1
    try:
        return int(text[start:end])
    except Exception:
        return None


def _compile_diagnostics(compile_state, dynamo, fetch_report, ingraph_timing):
    stats = (dynamo or {}).get("stats", {}) if isinstance(dynamo, dict) else {}
    graph_breaks = (dynamo or {}).get("graph_breaks") if isinstance(dynamo, dict) else None
    unique_graphs = stats.get("unique_graphs")
    packs = int((compile_state or {}).get("ingraph_packs", 0) or 0)
    calls = None
    if isinstance(ingraph_timing, dict):
        calls = ingraph_timing.get("calls")
    fetches = _ingraph_fetch_count(fetch_report)
    compile_path = (compile_state or {}).get("path")
    ingraph_compiled = bool((compile_state or {}).get("ingraph_compiled"))
    ingraph_compiled_blocks = int((compile_state or {}).get("ingraph_compiled_blocks", 0) or 0)
    regional_compiled_blocks = int((compile_state or {}).get("regional_compiled_blocks", 0) or 0)
    stream_compile_complete = bool((compile_state or {}).get("stream_compile_complete"))
    resident_compile_complete = bool((compile_state or {}).get("resident_compile_complete"))
    mixed_compile_complete = bool((compile_state or {}).get("mixed_compile_complete"))
    expected_fetches = None
    if ingraph_compiled and packs and calls is not None and int(calls) > 0:
        expected_fetches = packs * int(calls)
    fetches_match = None
    if fetches is not None and expected_fetches is not None:
        fetches_match = fetches == expected_fetches
    graph_break_clean = graph_breaks == 0 if graph_breaks is not None else None
    cache_clean = None
    if unique_graphs is not None:
        cache_clean = int(unique_graphs) <= 1
    warning = None
    if not stream_compile_complete and packs:
        warning = "streamed blocks did not compile"
    elif not resident_compile_complete:
        warning = "resident blocks did not compile"
    elif ingraph_compiled and unique_graphs and int(unique_graphs) > 1:
        warning = "ingraph compiled with multiple unique graphs"
    elif compile_path == "ingraph_partial_eager":
        warning = "partial ingraph ran eager"
    return {
        "compile_path": compile_path,
        "graph_break_clean": graph_break_clean,
        "compile_cache_clean": cache_clean,
        "unique_graphs": unique_graphs,
        "calls_captured": stats.get("calls_captured"),
        "ingraph_compiled_blocks": ingraph_compiled_blocks,
        "regional_compiled_blocks": regional_compiled_blocks,
        "stream_compile_complete": stream_compile_complete,
        "resident_compile_complete": resident_compile_complete,
        "mixed_compile_complete": mixed_compile_complete,
        "expected_fetches": expected_fetches,
        "actual_fetches": fetches,
        "fetches_match_packs": fetches_match,
        "warning": warning,
    }

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


def _arena_summary(transformer):
    """Ticket 534ea49 Phase 2 Slice E: arena presence/size + ingraph
    borrowed-vs-owned pack counts, for the full train->sample->train cycle
    instrumentation."""
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
        "ingraph_borrowed_packs": getattr(transformer, "_ingraph_sampling_borrowed_count", 0),
        "ingraph_owned_packs": getattr(transformer, "_ingraph_sampling_owned_count", 0),
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
        compile_cache_dir=args.compile_cache_dir,
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
        layer_offloading_pinned_arena=args.pinned_arena,
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
        # Ticket 534ea49/763bb75: pin the offloaded weights once into a
        # persistent per-block arena; enable_ingraph_sampling's pack build
        # (below, via --strict-ingraph) borrows this same flat instead of
        # allocating its own pinned pack over the same bytes (Slice 4).
        use_pinned_arena=bool(model_config.layer_offloading_pinned_arena),
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
    parser.add_argument(
        "--batch-cfg", action=argparse.BooleanOptionalAction, default=False,
        help="Batched CFG (cond+uncond in one forward). --no-batch-cfg forces sequential.",
    )
    parser.add_argument("--output", default=".codex/krea2_ingraph_cuda_smoke.png")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--working-reserve-gib", default="-1")
    parser.add_argument("--wddm-margin-gib", type=float, default=1.0)
    parser.add_argument("--wddm-hard-gib", type=float, default=1.0)
    parser.add_argument("--spill-reserve-pct", type=float, default=0.15)
    parser.add_argument("--pinned-weight-gib", type=float, default=-1.0)
    parser.add_argument(
        "--pinned-arena", action="store_true",
        help=(
            "Ticket 534ea49: pin offloaded weights once into a persistent "
            "per-block flat arena instead of re-pinning them at every "
            "sampling boundary. With --strict-ingraph, ingraph sampling "
            "packs borrow this same flat (Slice 4) instead of pinning a "
            "second copy of the same weights."
        ),
    )
    parser.add_argument("--checkpoint-keep-last", type=int, default=0)
    parser.add_argument("--block-stream-only", action="store_true")
    parser.add_argument("--fp8-training-forward", action="store_true")
    parser.add_argument("--fp8-sampling", action="store_true")
    parser.add_argument("--sampling-working-reserve-gib", default="-1")
    parser.add_argument("--sampling-wddm-margin-gib", type=float, default=-1.0)
    parser.add_argument("--sampling-wddm-hard-gib", type=float, default=1.0)
    parser.add_argument("--disable-compile-streamed", action="store_true")
    parser.add_argument("--no-compile-sample", action="store_true")
    parser.add_argument(
        "--ab-eager",
        action="store_true",
        help=(
            "Run an eager (compile disabled) baseline sample of --steps steps "
            "before the compile warmup, for an A/B per-forward comparison in "
            "the same process/layout."
        ),
    )
    parser.add_argument(
        "--compile-cache-dir",
        default=None,
        help=(
            "torch.compile mega-cache directory. Loaded before the first "
            "compiled sample, saved after new frames compile (same path the "
            "trainer uses via ModelConfig.compile_cache_dir)."
        ),
    )
    parser.add_argument(
        "--cap-descent", action="store_true",
        help=(
            "After the measured sample, probe torch's true sampling footprint: "
            "lower the allocator fraction cap one notch at a time and re-sample "
            "until GC churn/retries/slowdown appear. Reports the minimum viable "
            "cap (true need incl. fragmentation) vs the unrestricted footprint. "
            "Run with PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95 for "
            "production parity (run.py sets it on real jobs)."
        ),
    )
    parser.add_argument("--cap-descent-step-gib", type=float, default=0.25)
    parser.add_argument("--cap-descent-max-notches", type=int, default=16)
    parser.add_argument(
        "--margin-probe", action="store_true",
        help=(
            "After the measured sample, measure the sampling reserved-pool "
            "overshoot and its run-to-run spread: each pass empties the idle "
            "cache (forcing the transient pool to rebuild), varies the seed, and "
            "records peak_alloc vs peak_reserved. The reserved-over-alloc gap and "
            "its spread are the empirical basis for narrowing wddm_margin now that "
            "the allocator no longer jumps chaotically."
        ),
    )
    parser.add_argument(
        "--resample", type=int, default=0,
        help=(
            "After the first sampling cycle, re-enter inference_resident this "
            "many more times and run one measured sample each, printing where "
            "device_used/free land. This exercises the learn->consume climb: "
            "pass 1 is cold-conservative and records the activation peak; the "
            "next entries plan more residency from the learned reserve and "
            "should settle into the target device_free band."
        ),
    )
    parser.add_argument("--margin-probe-passes", type=int, default=8)
    parser.add_argument(
        "--margin-probe-safety-gib", type=float, default=0.375,
        help="Safety block added to the max measured overshoot for the recommended margin.",
    )
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

    model_setup_started = time.perf_counter()
    print("[smoke] constructing Krea2Model without text encoder")
    t0 = time.perf_counter()
    model = Krea2Model(device=args.device, model_config=config, dtype=args.dtype)
    model_construct_seconds = time.perf_counter() - t0
    model.skip_te = True
    rows.append(
        {
            "event": "start",
            "model_construct_seconds": model_construct_seconds,
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

    # Mirror BaseSDTrainProcess's model-load ordering (unet.requires_grad_(False)
    # before offload attach): this smoke never trains, so the transformer is
    # always frozen. Needed for --pinned-arena, whose arena covers frozen
    # offloaded weights only -- a still-trainable leaf fails attach closed.
    transformer.requires_grad_(False)

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
            "arena": _arena_summary(transformer),
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

    model_setup_seconds = time.perf_counter() - model_setup_started
    rows.append(
        {
            "event": "model_ready",
            "seconds": model_setup_seconds,
            "cuda": _cuda_snapshot("model_ready", device),
            "dxgi": _dxgi_snapshot("model_ready"),
        }
    )
    _print_json(rows[-1])

    def _make_gen_config(num_steps: int, output_path: str):
        return GenerateImageConfig(
            prompt="cached TE prompt",
            width=args.width,
            height=args.height,
            seed=args.seed,
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_steps),
            output_path=output_path,
            batch_cfg=bool(args.batch_cfg),
        )

    def _forward_count_from_fetches(fetch_report, compile_state):
        fetches = _ingraph_fetch_count(fetch_report)
        packs = int((compile_state or {}).get("ingraph_packs", 0) or 0)
        if fetches is None or packs <= 0:
            return None
        return fetches // packs if fetches % packs == 0 else fetches / packs

    def _run_sample(
        label: str,
        num_steps: int,
        output_path: str,
        *,
        keep_ingraph: bool,
        expected_forwards=None,
    ):
        gen_config = _make_gen_config(num_steps, output_path)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        started = time.perf_counter()
        image = model.generate_single_image(
            model.pipeline,
            gen_config,
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            generator=generator,
            extra={
                "keep_ingraph_sampling": keep_ingraph,
                "skip_sampling_guard": True,
            },
        )
        seconds = time.perf_counter() - started
        compile_state = _compile_path_snapshot(transformer)
        ingraph_timing = getattr(transformer, "_last_ingraph_sampling_timing", None)
        fetch_report = MemoryManager.ingraph_fetch_report(reset=True)
        profile_report = MemoryManager.offload_profile_report(reset=True)
        dynamo = _dynamo_counters()
        compile_diagnostics = _compile_diagnostics(
            compile_state,
            dynamo,
            fetch_report,
            ingraph_timing,
        )
        forward_calls = _forward_count_from_fetches(fetch_report, compile_state)
        if forward_calls is None and expected_forwards:
            # Eager pass has no in-graph fetches to count; the forward count
            # is known analytically from steps and CFG mode.
            forward_calls = int(expected_forwards)
        seconds_per_forward = (
            seconds / float(forward_calls)
            if forward_calls not in (None, 0)
            else None
        )
        result = {
            "label": label,
            "steps": int(num_steps),
            "seconds": seconds,
            "transformer_forward_calls": forward_calls,
            "seconds_per_transformer_forward": seconds_per_forward,
            "compile": compile_state,
            "compile_path": compile_state.get("path"),
            "compile_diagnostics": compile_diagnostics,
            "dynamo": dynamo,
            "ingraph_fetch_report": fetch_report,
            "ingraph_timing": ingraph_timing,
            "offload_profile_report": profile_report,
        }
        return image, result

    def _run_cap_descent():
        """Measure true sampling need vs unrestricted use via cap descent.

        Lowers the allocator fraction cap one notch per round and re-samples.
        Each notch runs a settle pass (absorbs the one-time stale-cache sweep
        the tighter cap forces on its first fresh malloc) and then a measured
        pass. A notch is dirty when the measured pass shows OOM-retry
        reclaims, sustained cudaFree churn, or a big slowdown -- the previous
        notch's cap is then the minimum viable footprint (peak live
        allocations + fragmentation slack). The manager's cap is restored
        afterwards.
        """
        gib = 1024 ** 3
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        total_b = torch.cuda.get_device_properties(index).total_memory
        base_fraction = MemoryManager._wddm_hard_cap_applied.get(index)
        if base_fraction is None:
            base_fraction = 1.0
        base_cap_b = base_fraction * total_b
        alloc_conf = os.environ.get(
            "PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        )
        gc_threshold_active = "garbage_collection_threshold" in alloc_conf

        def _gc_counters():
            stats = torch.cuda.memory_stats(index)
            return {
                "retries": int(stats.get("num_alloc_retries", 0)),
                "frees": int(stats.get("num_device_free", 0)),
                "mallocs": int(stats.get("num_device_alloc", 0)),
            }

        def _measured_pass(label):
            torch.cuda.synchronize(index)
            torch.cuda.reset_peak_memory_stats(index)
            before = _gc_counters()
            started = time.perf_counter()
            _image, _res = _run_sample(
                label,
                args.steps,
                ".codex/krea2_cap_descent.png",
                keep_ingraph=False,
            )
            torch.cuda.synchronize(index)
            after = _gc_counters()
            return {
                "seconds": time.perf_counter() - started,
                "retries_delta": after["retries"] - before["retries"],
                "cuda_free_delta": after["frees"] - before["frees"],
                "cuda_malloc_delta": after["mallocs"] - before["mallocs"],
                "peak_alloc_gib": _gib(torch.cuda.max_memory_allocated(index)),
                "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved(index)),
            }

        notches = []
        baseline_seconds = None
        floor_cap_b = None
        stop_reason = "max_notches"
        step_b = int(args.cap_descent_step_gib * gib)
        try:
            for notch in range(int(args.cap_descent_max_notches)):
                cap_b = base_cap_b - notch * step_b
                if cap_b <= 0:
                    stop_reason = "cap_exhausted"
                    break
                torch.cuda.set_per_process_memory_fraction(cap_b / total_b, index)
                try:
                    if notch > 0:
                        # Settle pass: absorb the one-time sweep at the new cap.
                        _run_sample(
                            f"cap_descent_settle_{notch}",
                            args.steps,
                            ".codex/krea2_cap_descent.png",
                            keep_ingraph=False,
                        )
                    measured = _measured_pass(f"cap_descent_measure_{notch}")
                except torch.cuda.OutOfMemoryError as error:
                    notches.append(
                        {"cap_gib": _gib(cap_b), "oom": repr(error)[:200]}
                    )
                    stop_reason = "oom"
                    break
                measured["cap_gib"] = _gib(cap_b)
                notches.append(measured)
                if baseline_seconds is None:
                    baseline_seconds = measured["seconds"]
                dirty = (
                    measured["retries_delta"] > 0
                    or measured["cuda_free_delta"] > 2
                    or measured["seconds"] > 1.5 * baseline_seconds
                )
                measured["dirty"] = dirty
                print(
                    f"[smoke] cap descent notch {notch}: cap={cap_b / gib:.2f} GiB "
                    f"seconds={measured['seconds']:.2f} "
                    f"retries=+{measured['retries_delta']} "
                    f"cudaFree=+{measured['cuda_free_delta']} "
                    f"peak_alloc={measured['peak_alloc_gib']:.2f} "
                    f"peak_reserved={measured['peak_reserved_gib']:.2f} "
                    f"{'DIRTY' if dirty else 'clean'}"
                )
                if dirty:
                    stop_reason = "dirty_notch"
                    break
                floor_cap_b = cap_b
        finally:
            torch.cuda.set_per_process_memory_fraction(base_fraction, index)

        clean = [n for n in notches if n.get("dirty") is False]
        peak_alloc_max = max(
            (n["peak_alloc_gib"] for n in clean), default=None
        )
        summary = {
            "event": "cap_descent",
            "base_cap_gib": _gib(base_cap_b),
            "floor_cap_gib": None if floor_cap_b is None else _gib(floor_cap_b),
            "stop_reason": stop_reason,
            "gc_threshold_active": gc_threshold_active,
            "peak_alloc_max_gib": peak_alloc_max,
            # Fragmentation slack: what the minimum viable footprint carries
            # beyond peak live allocations. With gc_threshold the effective
            # reserve ceiling is threshold*cap, without it the cap itself.
            "slack_gib": (
                None
                if floor_cap_b is None or peak_alloc_max is None
                else max(
                    0.0,
                    _gib(floor_cap_b) * (0.95 if gc_threshold_active else 1.0)
                    - peak_alloc_max,
                )
            ),
            "reclaimable_vs_base_gib": (
                None
                if floor_cap_b is None
                else max(0.0, _gib(base_cap_b) - _gib(floor_cap_b))
            ),
            "notches": notches,
        }
        rows.append(summary)
        _print_json(summary)
        return summary

    def _run_margin_probe():
        """Measure the sampling reserved-pool overshoot + its run-to-run spread.

        wddm_margin was historically wide because the caching allocator's
        reserved pool jumped chaotically above the planned footprint. This probe
        measures the *current* behaviour at the plan's own layout: each pass
        empties the idle cache (forcing the transient ring/activation pool to
        rebuild -- resident weights are live and stay), varies the seed, and
        records peak_alloc vs peak_reserved. The reserved-over-alloc gap is what
        the margin must cover; its spread across passes says how tight we can
        safely make it now.
        """
        import statistics

        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )

        def _gc_counters():
            stats = torch.cuda.memory_stats(index)
            return {
                "retries": int(stats.get("num_alloc_retries", 0)),
                "frees": int(stats.get("num_device_free", 0)),
            }

        passes = []
        base_seed = args.seed
        try:
            for i in range(int(args.margin_probe_passes)):
                args.seed = base_seed + i
                torch.cuda.empty_cache()
                torch.cuda.synchronize(index)
                torch.cuda.reset_peak_memory_stats(index)
                before = _gc_counters()
                started = time.perf_counter()
                _run_sample(
                    f"margin_probe_{i}",
                    args.steps,
                    ".codex/krea2_margin_probe.png",
                    keep_ingraph=False,
                )
                torch.cuda.synchronize(index)
                after = _gc_counters()
                peak_alloc = _gib(torch.cuda.max_memory_allocated(index))
                peak_reserved = _gib(torch.cuda.max_memory_reserved(index))
                row = {
                    "pass": i,
                    "seed": args.seed,
                    "seconds": time.perf_counter() - started,
                    "peak_alloc_gib": peak_alloc,
                    "peak_reserved_gib": peak_reserved,
                    "reserved_over_alloc_gib": peak_reserved - peak_alloc,
                    "retries_delta": after["retries"] - before["retries"],
                    "cuda_free_delta": after["frees"] - before["frees"],
                }
                passes.append(row)
                print(
                    f"[smoke] margin probe pass {i}: seed={args.seed} "
                    f"peak_alloc={peak_alloc:.2f} peak_reserved={peak_reserved:.2f} "
                    f"reserved_over_alloc={row['reserved_over_alloc_gib']:.3f} "
                    f"retries=+{row['retries_delta']} cudaFree=+{row['cuda_free_delta']} "
                    f"seconds={row['seconds']:.2f}"
                )
        finally:
            args.seed = base_seed

        def _stat(key):
            xs = [p[key] for p in passes] or [0.0]
            return {
                "min": min(xs),
                "max": max(xs),
                "mean": statistics.fmean(xs),
                "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            }

        over = _stat("reserved_over_alloc_gib")
        reserved = _stat("peak_reserved_gib")
        alloc = _stat("peak_alloc_gib")
        safety = float(args.margin_probe_safety_gib)
        recommended_margin_gib = over["max"] + safety
        summary = {
            "event": "margin_probe",
            "passes": passes,
            "reserved_over_alloc_gib": over,
            "peak_reserved_gib": reserved,
            "peak_alloc_gib": alloc,
            "safety_block_gib": safety,
            # The margin only has to cover the worst reserved overshoot beyond
            # planned live plus one safety block; the cap backstops the WDDM
            # cliff, so the old fat cushion is no longer the paging guard.
            "recommended_margin_gib": recommended_margin_gib,
            "any_retries": any(p["retries_delta"] > 0 for p in passes),
        }
        rows.append(summary)
        _print_json(summary)
        print(
            "[smoke] margin probe summary: reserved_over_alloc "
            f"min/mean/max/std={over['min']:.3f}/{over['mean']:.3f}/"
            f"{over['max']:.3f}/{over['std']:.3f} GiB; peak_reserved spread="
            f"{reserved['max'] - reserved['min']:.3f} GiB (std {reserved['std']:.3f}); "
            f"recommended margin ~{recommended_margin_gib:.2f} GiB "
            f"(max overshoot {over['max']:.3f} + safety {safety:.2f})"
        )
        return summary

    compile_label = "compile_sample" if config.compile_sample else "compile disabled"
    print(f"[smoke] sampling with MemoryManager.inference_resident and {compile_label}")
    transformer._ingraph_sampling_measure = False

    # Mirrors pipeline.py's CFG decision so the eager pass (no fetch counter)
    # can report a per-forward time.
    do_cfg = bool(guidance_scale and guidance_scale > 0 and unconditional_embeds is not None)
    forwards_per_step = 2 if do_cfg and not args.batch_cfg else 1

    def _run_eager_baseline():
        # Same layout, same process, compile fully off: flip the model-config
        # flags generate_single_image() gates on, restore afterwards.
        model_config = model.model_config
        saved = (
            model_config.compile_sample,
            model_config.layer_offloading_compile_streamed,
            model_config.layer_offloading_ingraph_sampling,
        )
        model_config.compile_sample = False
        model_config.layer_offloading_compile_streamed = False
        model_config.layer_offloading_ingraph_sampling = False
        try:
            return _run_sample(
                "eager_baseline",
                args.steps,
                ".codex/krea2_ingraph_cuda_eager.png",
                keep_ingraph=False,
                expected_forwards=args.steps * forwards_per_step,
            )
        finally:
            (
                model_config.compile_sample,
                model_config.layer_offloading_compile_streamed,
                model_config.layer_offloading_ingraph_sampling,
            ) = saved

    def _compile_cache_snapshot():
        if not args.compile_cache_dir:
            return None
        cache_dir = Path(args.compile_cache_dir)
        blobs = (
            sorted(p.name for p in cache_dir.glob("*.torchcompile_cache"))
            if cache_dir.is_dir()
            else []
        )
        return {"dir": str(cache_dir), "blobs": blobs}

    compile_cache_before = _compile_cache_snapshot()
    # Ticket 534ea49 Phase 2 Slice E: the arena must survive the sampling
    # boundary untouched -- same object, same committed bytes, no unpin/
    # repin churn. Snapshotted here (before) and compared after the
    # `with MemoryManager.inference_resident(...)` block exits below.
    arena_before = _arena_summary(transformer)
    sampling_context_started = time.perf_counter()
    eager_result = None
    warmup_result = None
    measured_result = None
    try:
        cold_start_hint = model.estimate_sampling_working_reserve_bytes(
            [_make_gen_config(args.steps, args.output)]
        )
        if cold_start_hint:
            print(
                f"[smoke] sampling cold-start estimate: "
                f"{cold_start_hint / 1024 ** 3:.2f} GiB "
                f"({args.width}x{args.height}, batch_cfg={bool(args.batch_cfg)}, "
                f"fp8={bool(config.layer_offloading_fp8_sampling)})"
            )
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
            cold_start_hint_bytes=cold_start_hint,
        ):
            if args.ab_eager and config.compile_sample:
                print(f"[smoke] eager baseline sample ({args.steps} steps, compile disabled)")
                _eager_image, eager_result = _run_eager_baseline()
            print("[smoke] warmup compile sample (1 step)")
            _warmup_image, warmup_result = _run_sample(
                "compile_warmup",
                1,
                ".codex/krea2_ingraph_cuda_warmup.png",
                keep_ingraph=True,
            )
            torch.cuda.empty_cache()
            print(f"[smoke] measured post-compile sample ({args.steps} steps)")
            image, measured_result = _run_sample(
                "post_compile_sample",
                args.steps,
                args.output,
                keep_ingraph=False,
            )
            if args.cap_descent:
                print("[smoke] cap descent probe: true need vs unrestricted use")
                _run_cap_descent()
            if args.margin_probe:
                print("[smoke] margin probe: reserved-pool overshoot + spread")
                _run_margin_probe()
        # Re-enter inference_resident to exercise the learn->consume climb: the
        # first cycle above recorded the activation peak, so these passes plan
        # more residency from the learned reserve and should settle into the
        # target device_free band. Each is a fresh plan (re-entry re-runs the
        # smart budget), unlike the margin probe which reuses one fixed layout.
        for extra in range(int(args.resample)):
            index = device.index if device.index is not None else torch.cuda.current_device()
            total_b = torch.cuda.get_device_properties(index).total_memory
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
                cold_start_hint_bytes=cold_start_hint,
            ):
                torch.cuda.synchronize(index)
                torch.cuda.reset_peak_memory_stats(index)
                _img, _res = _run_sample(
                    f"resample_{extra + 1}",
                    args.steps,
                    args.output,
                    keep_ingraph=False,
                )
                torch.cuda.synchronize(index)
                peak_reserved_b = torch.cuda.max_memory_reserved(index)
                peak_alloc_b = torch.cuda.max_memory_allocated(index)
                free_b, _tot = torch.cuda.mem_get_info(index)
                device_used_b = total_b - free_b
                print(
                    f"[smoke] resample {extra + 1}: "
                    f"peak_alloc={_gib(peak_alloc_b):.2f} "
                    f"peak_reserved={_gib(peak_reserved_b):.2f} "
                    f"device_used={_gib(device_used_b):.2f} "
                    f"device_free={_gib(free_b):.2f} GiB "
                    f"seconds={_res['seconds']:.2f}"
                )
        sampling_context_seconds = time.perf_counter() - sampling_context_started
        compile_state = measured_result["compile"]
        ingraph_timing = measured_result["ingraph_timing"]
        fetch_report = measured_result["ingraph_fetch_report"]
        profile_report = measured_result["offload_profile_report"]
        dynamo = measured_result["dynamo"]
        compile_diagnostics = measured_result["compile_diagnostics"]

        arena_after = _arena_summary(transformer)
        rows.append(
            {
                "event": "pinned_arena_sampling_boundary",
                "before": arena_before,
                "after": arena_after,
                # Same object, same committed bytes: the boundary must be
                # pure accounting, never an unpin/repin round trip.
                "arena_identity_unchanged": (
                    arena_before.get("present") == arena_after.get("present")
                    and arena_before.get("id") == arena_after.get("id")
                ),
                "ledger_weights_unchanged": (
                    arena_before.get("ledger_weights_gib")
                    == arena_after.get("ledger_weights_gib")
                ),
            }
        )
        _print_json(rows[-1])
    except Exception as error:
        sampling_context_seconds = time.perf_counter() - sampling_context_started
        compile_state = _compile_path_snapshot(transformer)
        ingraph_timing = getattr(transformer, "_last_ingraph_sampling_timing", None)
        fetch_report = MemoryManager.ingraph_fetch_report(reset=True)
        profile_report = MemoryManager.offload_profile_report(reset=True)
        dynamo = _dynamo_counters()
        compile_diagnostics = _compile_diagnostics(
            compile_state,
            dynamo,
            fetch_report,
            ingraph_timing,
        )
        phase_timing = {
            "model_setup_s": model_setup_seconds,
            "model_construct_s": model_construct_seconds,
            "sampling_context_s": sampling_context_seconds,
        }
        rows.append(
            {
                "event": "sample_failed",
                "seconds": sampling_context_seconds,
                "phase_timing": phase_timing,
                "compile_cache": {
                    "before": compile_cache_before,
                    "after": _compile_cache_snapshot(),
                },
                "eager_sample": eager_result,
                "warmup_sample": warmup_result,
                "measured_sample": measured_result,
                "error": repr(error),
                "cond_cache": str(cond_path),
                "uncond_cache": None if uncond_path is None else str(uncond_path),
                "guidance_scale": guidance_scale,
                "compile": compile_state,
                "compile_path": compile_state.get("path"),
                "compile_diagnostics": compile_diagnostics,
                "dynamo": dynamo,
                "ingraph_fetch_report": fetch_report,
                "ingraph_timing": ingraph_timing,
                "offload_profile_report": profile_report,
                "cuda": _cuda_snapshot("sample_failed", device),
                "dxgi": _dxgi_snapshot("sample_failed"),
            }
        )
        _print_json(rows[-1])
        if args.output_json:
            json_path = Path(args.output_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[smoke] wrote {json_path}")
        raise

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    image_save_started = time.perf_counter()
    image.save(out)
    image_save_seconds = time.perf_counter() - image_save_started
    phase_timing = {
        "model_setup_s": model_setup_seconds,
        "model_construct_s": model_construct_seconds,
        "sampling_context_s": sampling_context_seconds,
        "compile_warmup_sample_s": warmup_result["seconds"],
        "post_compile_sample_s": measured_result["seconds"],
        "post_compile_seconds_per_transformer_forward": measured_result[
            "seconds_per_transformer_forward"
        ],
        "post_compile_transformer_forward_calls": measured_result[
            "transformer_forward_calls"
        ],
        "image_save_s": image_save_seconds,
    }
    if eager_result is not None:
        phase_timing["eager_sample_s"] = eager_result["seconds"]
        phase_timing["eager_seconds_per_transformer_forward"] = eager_result[
            "seconds_per_transformer_forward"
        ]
        eager_spf = eager_result["seconds_per_transformer_forward"]
        compiled_spf = measured_result["seconds_per_transformer_forward"]
        if eager_spf and compiled_spf:
            phase_timing["compiled_vs_eager_forward_speedup"] = eager_spf / compiled_spf
    rows.append(
        {
            "event": "sampled",
            "seconds": sampling_context_seconds,
            "phase_timing": phase_timing,
            "compile_cache": {
                "before": compile_cache_before,
                "after": _compile_cache_snapshot(),
            },
            "eager_sample": eager_result,
            "warmup_sample": warmup_result,
            "measured_sample": measured_result,
            "output": str(out),
            "cond_cache": str(cond_path),
            "uncond_cache": None if uncond_path is None else str(uncond_path),
            "guidance_scale": guidance_scale,
            "compile": compile_state,
            "compile_path": compile_state.get("path"),
            "compile_diagnostics": compile_diagnostics,
            "dynamo": dynamo,
            "ingraph_fetch_report": fetch_report,
            "ingraph_timing": ingraph_timing,
            "offload_profile_report": profile_report,
            "cuda": _cuda_snapshot("sampled", device),
            "dxgi": _dxgi_snapshot("sampled"),
        }
    )
    _print_json(rows[-1])

    if args.pinned_arena:
        # Ticket 534ea49 Phase 2 Slice E: explicit teardown at the very end
        # (model unload, not a sampling boundary -- see _destroy_pinned_arena's
        # docstring). The "weights" ledger must return to its pre-arena
        # baseline; a nonzero remainder means something is still pinned that
        # should have been released.
        ledger_before_destroy = pin_manager.pinned_bytes_by_kind().get("weights", 0)
        MemoryManager._destroy_pinned_arena(transformer)
        ledger_after_destroy = pin_manager.pinned_bytes_by_kind().get("weights", 0)
        rows.append(
            {
                "event": "pinned_arena_destroyed",
                "ledger_weights_gib_before": _gib(ledger_before_destroy),
                "ledger_weights_gib_after": _gib(ledger_after_destroy),
                "arena_present_after": getattr(transformer, "_mm_weight_arena", None)
                is not None,
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
