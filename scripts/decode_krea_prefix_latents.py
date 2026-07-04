import argparse
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkit.basic import flush


DEFAULT_PREFIX_DIR = Path("loras/krea_vector_explore/txtfusion_probe/emotion_prefix_step4_t1")
DEFAULT_VAE_PATH = "D:\\.cache\\huggingface\\hub\\models--Qwen--Qwen-Image\\snapshots\\75e0b4be04f60ec59a75f475837eced720f823b6"


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def decode_latents(vae, latents: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    latents = latents.to(device=device, dtype=dtype).unsqueeze(2)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents = latents * latents_std + latents_mean
    with torch.no_grad():
        images = vae.decode(latents).sample
    return images.squeeze(2)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    image = tensor.float().clamp(-1.0, 1.0)
    image = ((image + 1.0) * 127.5).round().to(torch.uint8)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def iter_prefixes(prefix_dir: Path, prompt_ids: list[str] | None, pattern: str):
    if prompt_ids:
        for prompt_id in prompt_ids:
            path = prefix_dir / prompt_id / pattern
            if not path.exists():
                raise FileNotFoundError(path)
            yield prompt_id, path
    else:
        for path in sorted(prefix_dir.glob(f"*/{pattern}")):
            yield path.parent.name, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode saved Krea step-prefix latents to PNG previews.")
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--prompt-ids", nargs="*", default=None)
    parser.add_argument("--pattern", default="prefix_after_step_4.pt")
    parser.add_argument("--vae-path", default=DEFAULT_VAE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--output-suffix", default="prefix_after_step_4.png")
    parser.add_argument("--enable-tiling", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    print(f"Loading Qwen-Image VAE from {args.vae_path}")
    vae = AutoencoderKLQwenImage.from_pretrained(args.vae_path, subfolder="vae", torch_dtype=dtype, local_files_only=True)
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device=device, dtype=dtype)
    if args.enable_tiling:
        vae.enable_tiling()

    for prompt_id, prefix_path in iter_prefixes(args.prefix_dir, args.prompt_ids, args.pattern):
        payload = torch.load(prefix_path, map_location="cpu", weights_only=False)
        latents = payload["latents"]
        print(f"Decoding {prompt_id}: {list(latents.shape)}")
        image = decode_latents(vae, latents, device, dtype)
        output_path = prefix_path.parent / args.output_suffix
        save_image(image, output_path)
        print(f"Wrote {output_path}")
        flush(garbage_collect=False)


if __name__ == "__main__":
    main()
