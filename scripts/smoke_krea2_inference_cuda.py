"""Manual Krea2 arena-offload inference smoke with selectable adapters.

This replaces the retired in-graph sampling smoke with the current immutable
runtime lifecycle: load/quantize, prepare arena, install adapter, finalize,
enter a sampling session, and generate one image from cached text embeddings.
The default `smoke-direct-to-arena` load is the intended harness path.
`production-model-load` mirrors the production generic model-load session and
is only for testing that loading code. The deliberately long alternative
preserves legacy load-then-copy behavior for explicit paging tests.
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
from scripts import smoke_krea2_train_cuda as train_smoke  # noqa: E402
from toolkit.basic import flush  # noqa: E402
from toolkit.config_modules import GenerateImageConfig, ModelConfig  # noqa: E402
from toolkit.memory_management.arena_offload import get_arena_runtime  # noqa: E402
from toolkit.util.quantize import quantize_model  # noqa: E402
from scripts.smoke_runtime import (  # noqa: E402
    add_contention_args,
    add_load_mode_arg,
    add_lock_args,
    assert_smoke_load_mode,
    configure_smoke_load_mode,
    fail_if_vram_contended,
    smoke_model_load_session,
)


def _sibling_uncond_path(cond_path: Path):
    if "_cond." not in cond_path.name:
        return None
    candidate = cond_path.with_name(cond_path.name.replace("_cond.", "_uncond."))
    return candidate if candidate.exists() else None


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Manual full-CUDA Krea2 inference smoke using the immutable arena "
            "runtime and a selectable supported adapter."
        )
    )
    parser.add_argument("--cond-cache", default=train_smoke.DEFAULT_COND_CACHE)
    parser.add_argument("--uncond-cache", default=None)
    parser.add_argument("--model-path", default="krea/Krea-2-Raw")
    parser.add_argument("--checkpoint-filename", default=None)
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--qtype", default="float8")
    parser.add_argument("--cache-dir", default=None)
    add_load_mode_arg(parser)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--batch-cfg", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    train_smoke.add_adapter_args(parser)
    parser.add_argument("--working-reserve-gib", default="-1")
    parser.add_argument("--sampling-working-reserve-gib", default="-1")
    parser.add_argument("--wddm-margin-gib", type=float, default=1.0)
    parser.add_argument("--wddm-hard-gib", type=float, default=1.0)
    parser.add_argument("--spill-reserve-pct", type=float, default=0.20)
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument("--fp8-training-forward", action="store_true")
    parser.add_argument("--fp8-sampling", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--compile-cache-dir", default="tmp/torch_compile_cache")
    parser.add_argument("--output", default=".codex/krea2_inference_smoke.png")
    parser.add_argument("--output-json", default=None)
    add_contention_args(parser)
    add_lock_args(parser)
    return parser.parse_args()


def _model_config(args):
    model_kwargs = {
        "max_text_length": args.max_text_length,
        "local_files_only": not args.allow_download,
    }
    if args.cache_dir:
        model_kwargs["quantized_transformer_cache_dir"] = args.cache_dir
    if args.checkpoint_filename:
        model_kwargs["checkpoint_filename"] = args.checkpoint_filename
    if args.vae_path:
        model_kwargs["vae_path"] = args.vae_path
    return ModelConfig(
        name_or_path=args.model_path,
        arch="krea2",
        dtype=args.dtype,
        quantize=True,
        qtype=args.qtype,
        layer_offloading=True,
        layer_offloading_smart=True,
        layer_offloading_smart_working_reserve_gb=args.working_reserve_gib,
        layer_offloading_smart_sampling_working_reserve_gb=(
            args.sampling_working_reserve_gib
        ),
        layer_offloading_smart_wddm_margin_gb=args.wddm_margin_gib,
        layer_offloading_smart_wddm_hard_gb=args.wddm_hard_gib,
        layer_offloading_smart_sampling_wddm_margin_gb=args.wddm_margin_gib,
        layer_offloading_smart_sampling_wddm_hard_gb=args.wddm_hard_gib,
        layer_offloading_wddm_spill_reserve_pct=args.spill_reserve_pct,
        layer_offloading_prefetch_depth=args.prefetch_depth,
        layer_offloading_fp8_forward=args.fp8_training_forward,
        layer_offloading_fp8_sampling=args.fp8_sampling,
        compile=not args.no_compile,
        compile_sample=not args.no_compile,
        compile_cache_dir=args.compile_cache_dir or None,
        model_kwargs=model_kwargs,
    )


def main():
    args = _parse_args()
    fail_if_vram_contended(
        args.device,
        ignore_contention=args.ignore_contention,
    )
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This smoke requires an available CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cond_path = Path(args.cond_cache)
    if not cond_path.exists():
        raise FileNotFoundError(cond_path)
    uncond_path = (
        Path(args.uncond_cache)
        if args.uncond_cache
        else _sibling_uncond_path(cond_path)
    )
    conditional = train_smoke._load_prompt_cache(str(cond_path))
    unconditional = (
        train_smoke._load_prompt_cache(str(uncond_path))
        if uncond_path is not None
        else None
    )
    guidance = args.guidance_scale
    if guidance is None:
        guidance = 4.5 if unconditional is not None else 0.0

    print("[smoke] loading Krea2 transformer")
    config = _model_config(args)
    model = Krea2Model(device=args.device, model_config=config, dtype=args.dtype)
    model.skip_te = True
    configure_smoke_load_mode(model, args.load_mode)
    with smoke_model_load_session(model, args.load_mode):
        transformer = model._load_transformer()
        assert_smoke_load_mode(model, args.load_mode)
        if config.quantize and not getattr(
            model, "_transformer_quantized_during_load", False
        ):
            print("[smoke] quantizing transformer")
            quantize_model(model, transformer)
            flush()

        print("[smoke] preparing immutable arena offload")
        ignore_modules = [
            module
            for module in transformer.modules()
            if isinstance(module, (SimpleModulation, DoubleSharedModulation))
        ]
        transformer.enable_gradient_checkpointing(
            keep_last=config.layer_offloading_checkpoint_keep_last
        )
        model._attach_immutable_training_memory(transformer, ignore_modules)
        model.model = transformer

    print(f"[smoke] applying {args.adapter_variant} adapter")
    network = train_smoke._apply_lora(
        model,
        transformer,
        device,
        args.lora_rank,
        args.lora_alpha,
        **train_smoke.adapter_options(args),
    )
    transformer.eval()
    runtime = get_arena_runtime(transformer)
    if runtime is None:
        raise RuntimeError("Krea2 did not prepare the arena offload runtime")
    runtime.finalize(network)

    print("[smoke] loading VAE")
    model.vae = model._load_vae()
    model.vae.to(model.vae_device_torch, dtype=model.vae_torch_dtype)
    model.pipeline = Krea2Pipeline(model)

    gen_config = GenerateImageConfig(
        prompt="cached TE prompt",
        width=args.width,
        height=args.height,
        seed=args.seed,
        guidance_scale=guidance,
        num_inference_steps=args.steps,
        output_path=args.output,
        batch_cfg=args.batch_cfg,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    started = time.perf_counter()
    with runtime.sampling_session():
        image = model.generate_single_image(
            model.pipeline,
            gen_config,
            conditional_embeds=conditional,
            unconditional_embeds=unconditional,
            generator=generator,
            extra={"skip_sampling_guard": True},
        )
    seconds = time.perf_counter() - started

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    result = {
        "adapter_variant": args.adapter_variant,
        "full_if_contains": train_smoke.adapter_options(args)["full_if_contains"],
        "load_mode": args.load_mode,
        "seconds": seconds,
        "output": str(output),
        "runtime": runtime.diagnostics(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    # Serialize GPU scripts: two 11+ GiB smokes on a 12 GB card do not
    # just measure badly, the second OOMs. See scripts/_gpu_lock.py.
    from smoke_runtime import run_locked

    sys.exit(run_locked("smoke_krea2_inference_cuda", main))
