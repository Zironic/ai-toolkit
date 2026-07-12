import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model
from toolkit.config_modules import ModelConfig
from toolkit.basic import flush
from scripts.smoke_runtime import add_contention_args, fail_if_vram_contended


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Krea2 quantized transformer cache load/save.")
    parser.add_argument("--model-path", default="krea/Krea-2-Raw")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--qtype", default="float8")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--disable-cache", action="store_true")
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
            model_kwargs=kwargs,
        ),
        dtype=args.dtype,
    )
    print("calling _load_transformer")
    flush(garbage_collect=False)
    transformer = model._load_transformer()
    print(f"loaded transformer type={type(transformer).__name__}")
    flush(garbage_collect=False)


if __name__ == "__main__":
    # Serialize GPU scripts: two 11+ GiB smokes on a 12 GB card do not
    # just measure badly, the second OOMs. See scripts/_gpu_lock.py.
    from smoke_runtime import run_locked

    sys.exit(run_locked("smoke_krea2_quant_cache", main))
