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
from toolkit.config_modules import ModelConfig  # noqa: E402
from toolkit.memory_management import MemoryManager, bounce_pool, dxgi_meminfo  # noqa: E402
from toolkit.util.quantize import quantize_model  # noqa: E402

GIB = 1024 ** 3


def _gib(value):
    if value is None:
        return None
    return float(value) / GIB


def _dxgi_snapshot(label, device_index=0):
    info = dxgi_meminfo.query_non_local_video_memory_info(
        cuda_device_index=device_index,
        min_interval_s=0.0,
    )
    adapter = dxgi_meminfo.selected_adapter_info()
    row = {
        "label": label,
        "adapter": None if adapter is None else {
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
        row.update({
            "budget_gib": _gib(info.budget_bytes),
            "usage_gib": _gib(info.current_usage_bytes),
            "raw_headroom_gib": _gib(raw_headroom),
            "spill_reserve_gib": _gib(reserve),
            "usable_headroom_gib": _gib(raw_headroom - reserve),
        })
    return row


def _cuda_snapshot(label, device):
    row = {"label": label, "available": torch.cuda.is_available()}
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return row
    free_b, total_b = torch.cuda.mem_get_info(device)
    row.update({
        "free_gib": _gib(free_b),
        "total_gib": _gib(total_b),
        "torch_allocated_gib": _gib(torch.cuda.memory_allocated(device)),
        "torch_reserved_gib": _gib(torch.cuda.memory_reserved(device)),
        "pinned_ledger_gib": _gib(bounce_pool._pinned_bytes_total),
    })
    return row


def _managed_summary(transformer):
    managed = pinned_layers = 0
    pinned_bytes = 0
    resident = 0
    for child in transformer.modules():
        if hasattr(child, "_layer_memory_manager"):
            managed += 1
            n = int(getattr(child, "_mm_pinned_bytes", 0) or 0)
            if n > 0:
                pinned_layers += 1
                pinned_bytes += n
        elif child.__class__.__name__ in ("Linear", "LoRACompatibleLinear", "QLinear", "Conv2d", "LoRACompatibleConv", "QConv"):
            resident += 1
    return {
        "managed_layers": managed,
        "resident_candidate_layers": resident,
        "pinned_layers": pinned_layers,
        "pinned_weight_gib": _gib(pinned_bytes),
        "manager_pinned_gib": _gib(getattr(getattr(transformer, "_memory_manager", None), "pinned_weight_bytes", 0)),
        "ledger_gib": _gib(bounce_pool._pinned_bytes_total),
    }


def _pool_summary(transformer):
    mm = getattr(transformer, "_memory_manager", None)
    pool = getattr(mm, "_prefetch_pool", None) if mm is not None else None
    if pool is None:
        return None
    try:
        return pool.stats(reset=False)
    except Exception as exc:
        return {"error": repr(exc)}


def _print_json(row):
    print(json.dumps(row, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load the real Krea2 FP8 transformer and exercise training memory "
            "management without running a training forward/backward."
        )
    )
    parser.add_argument("--model-path", default="krea/Krea-2-Raw")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--qtype", default="float8")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--disable-quant-cache", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--working-reserve-gib", default="-1")
    parser.add_argument("--wddm-margin-gib", type=float, default=1.0)
    parser.add_argument("--wddm-hard-gib", type=float, default=1.0)
    parser.add_argument("--spill-reserve-pct", type=float, default=0.20)
    parser.add_argument("--pinned-weight-gib", type=float, default=-1.0)
    parser.add_argument("--checkpoint-keep-last", type=int, default=-1)
    parser.add_argument("--block-stream-only", action="store_true")
    parser.add_argument("--fp8-training-forward", action="store_true")
    parser.add_argument("--sampling-transition", action="store_true")
    parser.add_argument("--sampling-working-reserve-gib", default="-1")
    parser.add_argument("--allow-download", action="store_true", help="Permit Hugging Face downloads if the model is not cached locally.")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is false")

    os.environ.setdefault("AI_TOOLKIT_MEMORY_DIAGNOSTICS", "1")

    kwargs = {"max_text_length": args.max_text_length}
    if args.cache_dir:
        kwargs["quantized_transformer_cache_dir"] = args.cache_dir
    if args.disable_quant_cache:
        kwargs["quantized_transformer_cache"] = False

    config = ModelConfig(
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
        model_kwargs=kwargs,
    )

    rows = []
    device = torch.device(args.device)
    print("[probe] constructing Krea2Model")
    model = Krea2Model(device=args.device, model_config=config, dtype=args.dtype)
    model.skip_te = True

    rows.append({"event": "start", "cuda": _cuda_snapshot("start", device), "dxgi": _dxgi_snapshot("start")})
    _print_json(rows[-1])

    print("[probe] loading real Krea2 transformer")
    t0 = time.perf_counter()
    transformer = model._load_transformer()
    rows.append({
        "event": "loaded_transformer",
        "seconds": time.perf_counter() - t0,
        "cuda": _cuda_snapshot("loaded_transformer", device),
        "dxgi": _dxgi_snapshot("loaded_transformer"),
    })
    _print_json(rows[-1])
    flush(garbage_collect=False)

    if config.quantize and not getattr(model, "_transformer_quantized_during_load", False):
        print("[probe] quantizing transformer to FP8")
        t0 = time.perf_counter()
        quantize_model(model, transformer)
        flush()
        rows.append({
            "event": "quantized_transformer",
            "seconds": time.perf_counter() - t0,
            "cuda": _cuda_snapshot("quantized_transformer", device),
            "dxgi": _dxgi_snapshot("quantized_transformer"),
        })
        _print_json(rows[-1])
    else:
        print("[probe] transformer already quantized during load/cache")

    ignore_modules = [
        module for module in transformer.modules()
        if isinstance(module, (SimpleModulation, DoubleSharedModulation))
    ]
    keep_last = config.layer_offloading_checkpoint_keep_last
    pinned_resident_keys = MemoryManager.training_pinned_keys_for_keep_last(
        transformer, max(0, keep_last)
    )

    print("[probe] attaching smart training offload (this is the training memory setup)")
    t0 = time.perf_counter()
    MemoryManager.attach_smart_training(
        transformer,
        device,
        working_reserve_gib=config.layer_offloading_smart_working_reserve_gb,
        wddm_margin_gib=config.layer_offloading_smart_wddm_margin_gb,
        wddm_hard_gib=config.layer_offloading_smart_wddm_hard_gb,
        ignore_modules=ignore_modules,
        pinned_resident_keys=pinned_resident_keys,
        block_stream_only=config.layer_offloading_block_stream_only,
        pinned_weight_gib=config.layer_offloading_pinned_weight_gb,
        wddm_spill_reserve_pct=config.layer_offloading_wddm_spill_reserve_pct,
        fp8_training_forward=bool(config.layer_offloading_fp8_forward),
    )
    transformer.enable_gradient_checkpointing(keep_last=max(0, keep_last))
    mm = getattr(transformer, "_memory_manager", None)
    if mm is not None:
        MemoryManager._attach_prefetch_pool(transformer, device)
    rows.append({
        "event": "attached_training_memory",
        "seconds": time.perf_counter() - t0,
        "summary": _managed_summary(transformer),
        "pool": _pool_summary(transformer),
        "cuda": _cuda_snapshot("attached_training_memory", device),
        "dxgi": _dxgi_snapshot("attached_training_memory"),
    })
    _print_json(rows[-1])

    if args.sampling_transition:
        print("[probe] entering sampling resident context without running a forward")
        t0 = time.perf_counter()
        with MemoryManager.inference_resident(
            transformer,
            device=device,
            fp8_sampling=True,
            working_reserve_gib=args.sampling_working_reserve_gib,
            wddm_margin_gib=config.layer_offloading_smart_wddm_margin_gb,
            wddm_hard_gib=config.layer_offloading_smart_wddm_hard_gb,
        ):
            rows.append({
                "event": "inside_sampling_transition_no_forward",
                "summary": _managed_summary(transformer),
                "cuda": _cuda_snapshot("inside_sampling_transition_no_forward", device),
                "dxgi": _dxgi_snapshot("inside_sampling_transition_no_forward"),
            })
            _print_json(rows[-1])
        restored_mm = getattr(transformer, "_memory_manager", None)
        if restored_mm is not None and getattr(restored_mm, "_prefetch_pool", None) is None:
            MemoryManager._attach_prefetch_pool(transformer, device)
        rows.append({
            "event": "restored_training_memory_after_sampling",
            "seconds": time.perf_counter() - t0,
            "summary": _managed_summary(transformer),
            "pool": _pool_summary(transformer),
            "cuda": _cuda_snapshot("restored_training_memory_after_sampling", device),
            "dxgi": _dxgi_snapshot("restored_training_memory_after_sampling"),
        })
        _print_json(rows[-1])

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[probe] wrote {out}")

    print("[probe] complete; no training forward/backward was run")


if __name__ == "__main__":
    main()
