import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capture_krea_txtfusion_forward import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_IMAGE,
    DEFAULT_LORA,
    DEFAULT_MODEL,
    build_model,
    load_cached_prompt_features,
    load_image_tensor,
    run_forward,
    tensor_stats,
)
from sweep_skc3vo_blank_vector import extract_projector_vector  # noqa: E402
from toolkit.basic import flush  # noqa: E402


DEFAULT_PROMPTS = Path("loras/krea_vector_explore/txtfusion_probe/prompts.json")
DEFAULT_OUTPUT_DIR = Path("loras/krea_vector_explore/txtfusion_probe/captures/multicapture_testpng_float8")
DEFAULT_PROMPT_IDS = [
    "blank",
    "covered_wide",
    "exposed_wide",
    "covered_close",
    "exposed_close",
    "covered_full_body",
    "exposed_full_body",
    "covered_torso",
    "exposed_torso",
    "fashion_wide",
    "body_focus_wide",
    "clean_no_injury",
    "bloody_injury",
    "no_weapon",
    "holding_weapon",
    "sfw_wide",
    "sfw_close",
    "sfw_full_body",
    "sfw_torso",
    "sfw_fashion_wide",
]


def load_prompt_texts(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: item["prompt"] for item in data.get("phrases", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture multiple Krea2 txtfusion.projector contexts with one model load.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-ids", nargs="+", default=DEFAULT_PROMPT_IDS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--lora-strength", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--noise-t", type=float, default=0.5)
    parser.add_argument("--latent-mode", choices=["image", "random"], default="image")
    parser.add_argument("--skip-vae", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--vae-dtype", default="bf16")
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quantize-te", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qtype", default="float8")
    parser.add_argument("--qtype-te", default="qfloat8")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layer-offloading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layer-offloading-transformer-percent", type=float, default=1.0)
    parser.add_argument("--layer-offloading-text-encoder-percent", type=float, default=1.0)
    parser.add_argument("--max-text-length", type=int, default=2000)
    parser.add_argument("--checkpoint-filename", default=None)
    parser.add_argument("--text-encoder-path", default=None)
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False,
                        help="Skip prompt IDs whose capture .pt file already exists in output-dir.")
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(args.features)
    if not args.prompts.exists():
        raise FileNotFoundError(args.prompts)
    if args.lora and not args.lora.exists():
        raise FileNotFoundError(args.lora)
    if args.latent_mode == "image" and not args.image.exists():
        raise FileNotFoundError(args.image)

    args.cached_prompt = True
    args.negative_prompt = ""

    prompt_texts = load_prompt_texts(args.prompts)
    missing = [prompt_id for prompt_id in args.prompt_ids if prompt_id not in prompt_texts]
    if missing:
        raise ValueError(f"Unknown prompt ids in {args.prompts}: {missing}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Loading model...")
    flush()
    model = build_model(args)
    print("Model loaded")
    flush()
    device = model.device_torch

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t_value = float(args.noise_t)
    print(f"Preparing latent mode={args.latent_mode} noise_t={args.noise_t}")
    flush()
    if args.latent_mode == "image":
        if model.vae is None:
            raise ValueError("--latent-mode image requires VAE; remove --skip-vae or pass a working --vae-path")
        image_tensor = load_image_tensor(args.image, args.width, args.height)
        print("Encoding image through VAE...")
        flush()
        with torch.no_grad():
            clean_latents = model.encode_images([image_tensor], device=model.vae_device_torch, dtype=model.vae_torch_dtype).detach()
        print("VAE encode complete")
        flush()
        noise = torch.randn(clean_latents.shape, generator=generator, dtype=torch.float32).to(clean_latents.device)
        noisy_latents = (1.0 - t_value) * clean_latents.float() + t_value * noise.float()
    else:
        latent_channels = model.model.config.channels
        clean_latents = torch.zeros(1, latent_channels, args.height // model.vae_scale_factor, args.width // model.vae_scale_factor, dtype=torch.float32)
        noise = torch.randn(clean_latents.shape, generator=generator, dtype=torch.float32)
        noisy_latents = noise.float()
    noisy_latents = noisy_latents.to(device, dtype=model.torch_dtype)
    t = torch.full((noisy_latents.shape[0],), t_value, device=device, dtype=model.torch_dtype)

    vector_source = None
    projector_vector = None
    if args.lora:
        vector_source, projector_vector = extract_projector_vector(load_file(str(args.lora)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.output_dir / "manifest.json"
    captures = []
    summaries = []

    for prompt_id in args.prompt_ids:
        capture_path = args.output_dir / f"capture_{prompt_id}.pt"
        if args.skip_existing and capture_path.exists():
            print(f"Skipping {prompt_id} (already captured)")
            flush()
            captures.append(str(capture_path))
            summaries.append(str(args.output_dir / f"capture_{prompt_id}_summary.json"))
            continue
        prompt = prompt_texts[prompt_id]
        print(f"Loading cached features for {prompt_id}...")
        flush()
        context, text_mask = load_cached_prompt_features(args.features, prompt_id, model.device_torch, model.torch_dtype)
        print(f"Running base forward for {prompt_id}...")
        flush()
        base_velocity, base_capture = run_forward(
            model,
            noisy_latents,
            t,
            context,
            text_mask,
            projector_vector=None,
            lora_strength=0.0,
        )
        lora_velocity = None
        lora_capture = None
        if projector_vector is not None and args.lora_strength != 0.0:
            print(f"Running LoRA forward for {prompt_id}...")
            flush()
            lora_velocity, lora_capture = run_forward(
                model,
                noisy_latents,
                t,
                context,
                text_mask,
                projector_vector=projector_vector,
                lora_strength=args.lora_strength,
            )

        summary_path = args.output_dir / f"capture_{prompt_id}_summary.json"
        payload = OrderedDict(
            [
                ("schema", "krea_txtfusion_forward_capture.v1"),
                ("prompt", prompt),
                ("prompt_id", prompt_id),
                ("cached_prompt", True),
                ("features", str(args.features)),
                ("negative_prompt", ""),
                ("model_path", args.model_path),
                ("image", str(args.image)),
                ("width", args.width),
                ("height", args.height),
                ("noise_t", t_value),
                ("latent_mode", args.latent_mode),
                ("skip_vae", args.skip_vae),
                ("seed", args.seed),
                ("dtype", args.dtype),
                ("quantize", args.quantize),
                ("qtype", args.qtype),
                ("lora", str(args.lora) if args.lora else None),
                ("lora_strength", args.lora_strength),
                ("projector_vector_source", vector_source),
                ("projector_vector", projector_vector.cpu() if projector_vector is not None else None),
                ("clean_latents", clean_latents.detach().cpu()),
                ("noise", noise.detach().cpu()),
                ("noisy_latents", noisy_latents.detach().cpu()),
                ("context", context.detach().cpu()),
                ("text_mask", text_mask.detach().cpu()),
                ("uncond_context", None),
                ("uncond_mask", None),
                ("base_projector_input", base_capture["input"]),
                ("base_projector_output", base_capture["output"]),
                ("base_velocity", base_velocity),
                ("lora_projector_input", lora_capture["input"] if lora_capture is not None else None),
                ("lora_projector_output", lora_capture["output"] if lora_capture is not None else None),
                ("lora_velocity", lora_velocity),
            ]
        )
        torch.save(payload, capture_path)

        summary = OrderedDict(
            [
                ("schema", "krea_txtfusion_forward_capture_summary.v1"),
                ("capture", str(capture_path)),
                ("prompt", prompt),
                ("prompt_id", prompt_id),
                ("base_projector_input", tensor_stats(base_capture["input"])),
                ("base_projector_output", tensor_stats(base_capture["output"])),
                ("base_velocity", tensor_stats(base_velocity)),
            ]
        )
        if lora_capture is not None and lora_velocity is not None:
            projector_output_delta = lora_capture["output"].float() - base_capture["output"].float()
            velocity_delta = lora_velocity.float() - base_velocity.float()
            summary["lora_projector_output_delta"] = tensor_stats(projector_output_delta)
            summary["lora_velocity_delta"] = tensor_stats(velocity_delta)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        captures.append(str(capture_path))
        summaries.append(str(summary_path))
        print(f"Captured {prompt_id}: projector_delta_rms={summary.get('lora_projector_output_delta', {}).get('rms', 0.0):.6f}")
        flush()

    manifest = OrderedDict(
        [
            ("schema", "krea_txtfusion_forward_multicapture_manifest.v1"),
            ("output_dir", str(args.output_dir)),
            ("prompt_ids", args.prompt_ids),
            ("captures", captures),
            ("summaries", summaries),
            ("latent_mode", args.latent_mode),
            ("image", str(args.image)),
            ("width", args.width),
            ("height", args.height),
            ("noise_t", t_value),
            ("seed", args.seed),
            ("qtype", args.qtype),
            ("lora", str(args.lora) if args.lora else None),
            ("lora_strength", args.lora_strength),
        ]
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
