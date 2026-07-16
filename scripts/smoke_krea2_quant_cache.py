"""Smoke the Krea2 quantized cache through direct canonical arena loading.

The default commits the restored or freshly quantized payload directly into
the arena. ``--load-mode production-model-load`` remains available when the
production checkpoint-loading lifecycle itself is under test.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extensions_built_in.diffusion_models.krea2.krea2 import (
    DoubleSharedModulation,
    Krea2Model,
    SimpleModulation,
    _save_quantized_transformer_cache,
)
from toolkit.config_modules import ModelConfig
from toolkit.basic import flush
from scripts.smoke_runtime import (
    SMOKE_DIRECT_LOAD_MODE,
    add_contention_args,
    add_load_mode_arg,
    assert_smoke_load_mode,
    configure_smoke_load_mode,
    fail_if_vram_contended,
    smoke_model_load_session,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Krea2 quantized transformer cache load/save.")
    parser.add_argument("--model-path", default="krea/Krea-2-Raw")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--qtype", default="float8")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--disable-cache", action="store_true")
    add_load_mode_arg(parser)
    add_contention_args(parser)
    args = parser.parse_args()
    fail_if_vram_contended(
        args.device,
        ignore_contention=args.ignore_contention,
    )

    kwargs = {"max_text_length": 512}
    if args.cache_dir:
        kwargs["quantized_transformer_cache_dir"] = args.cache_dir
    if args.disable_cache:
        kwargs["quantized_transformer_cache"] = False

    print("constructing Krea2Model")
    flush(garbage_collect=False)
    model = Krea2Model(
        device=args.device,
        model_config=ModelConfig(
            name_or_path=args.model_path,
            arch="krea2",
            dtype=args.dtype,
            quantize=True,
            qtype=args.qtype,
            layer_offloading=args.load_mode == SMOKE_DIRECT_LOAD_MODE,
            layer_offloading_smart=args.load_mode == SMOKE_DIRECT_LOAD_MODE,
            compile=False,
            model_kwargs=kwargs,
        ),
        dtype=args.dtype,
    )
    configure_smoke_load_mode(model, args.load_mode)
    print("calling _load_transformer")
    flush(garbage_collect=False)
    runtime = None
    try:
        with smoke_model_load_session(model, args.load_mode):
            transformer = model._load_transformer()
            assert_smoke_load_mode(model, args.load_mode)
            if args.load_mode == SMOKE_DIRECT_LOAD_MODE:
                transformer.requires_grad_(False)
                transformer.enable_gradient_checkpointing(keep_last=0)
                ignore_modules = [
                    module
                    for module in transformer.modules()
                    if isinstance(
                        module, (SimpleModulation, DoubleSharedModulation)
                    )
                ]
                runtime = model._attach_immutable_training_memory(
                    transformer, ignore_modules
                )
                pending_cache = getattr(
                    model, "_pending_quantized_transformer_cache", None
                )
                if pending_cache is not None:
                    _save_quantized_transformer_cache(
                        model, transformer, pending_cache[0], pending_cache[1]
                    )
                    model._pending_quantized_transformer_cache = None
        print(
            f"loaded transformer type={type(transformer).__name__} "
            f"load_mode={args.load_mode}"
        )
        flush(garbage_collect=False)
    finally:
        if runtime is not None:
            runtime.close()
        model.cleanup_memory_runtime_preparation()


if __name__ == "__main__":
    # Serialize GPU scripts: two 11+ GiB smokes on a 12 GB card do not
    # just measure badly, the second OOMs. See scripts/_gpu_lock.py.
    from smoke_runtime import run_locked

    sys.exit(run_locked("smoke_krea2_quant_cache", main))
