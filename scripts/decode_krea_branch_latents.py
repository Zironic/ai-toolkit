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


DEFAULT_INPUT_DIR = Path("loras/krea_vector_explore/txtfusion_probe/emotion_suffix_fork_t1")
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


def iter_latent_files(input_dir: Path, prompt_ids: list[str] | None, branch_names: list[str] | None):
    if prompt_ids and branch_names:
        for prompt_id in prompt_ids:
            for branch in branch_names:
                path = input_dir / prompt_id / f"{branch}.pt"
                if not path.exists():
                    raise FileNotFoundError(path)
                yield path
        return
    for path in sorted(input_dir.glob("*/*.pt")):
        if path.name.startswith("prefix_after_step_"):
            continue
        if prompt_ids and path.parent.name not in prompt_ids:
            continue
        if branch_names and path.stem not in branch_names:
            continue
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode saved Krea branch latents to PNG previews.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--prompt-ids", nargs="*", default=None)
    parser.add_argument("--branches", nargs="*", default=None)
    parser.add_argument("--vae-path", default=DEFAULT_VAE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--enable-tiling", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    latent_files = list(iter_latent_files(args.input_dir, args.prompt_ids, args.branches))
    if not latent_files:
        raise FileNotFoundError(f"No branch .pt files found under {args.input_dir}")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    print(f"Loading Qwen-Image VAE from {args.vae_path}")
    vae = AutoencoderKLQwenImage.from_pretrained(args.vae_path, subfolder="vae", torch_dtype=dtype, local_files_only=True)
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device=device, dtype=dtype)
    if args.enable_tiling:
        vae.enable_tiling()

    decoded = 0
    skipped = 0
    for latent_path in latent_files:
        output_path = latent_path.with_suffix(".png")
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing {output_path}")
            skipped += 1
            continue
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        if "latents" not in payload:
            print(f"Skipping {latent_path}: no latents key")
            skipped += 1
            continue
        latents = payload["latents"]
        print(f"Decoding {latent_path.parent.name}/{latent_path.stem}: {list(latents.shape)}")
        image = decode_latents(vae, latents, device, dtype)
        save_image(image, output_path)
        print(f"Wrote {output_path}")
        decoded += 1
        flush(garbage_collect=False)

    print(f"Decoded {decoded}, skipped {skipped}")


if __name__ == "__main__":
    main()
