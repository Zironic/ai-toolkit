import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capture_krea_txtfusion_forward import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_LORA,
    build_model,
    load_cached_prompt_features,
    projector_delta_forward,
)
from extensions_built_in.diffusion_models.krea2.src.pipeline import predict_velocity, timesteps  # noqa: E402
from sweep_skc3vo_blank_vector import extract_projector_vector, rms  # noqa: E402
from toolkit.basic import flush  # noqa: E402


DEFAULT_MODEL = "krea/Krea-2-Turbo"
DEFAULT_OUTPUT_DIR = Path("loras/krea_vector_explore/txtfusion_probe/emotion_suffix_fork")
DEFAULT_PROMPT_IDS = ["emotion_face_neutral", "emotion_face_crying", "jinx_close_neutral", "jinx_close_crying"]
DEFAULT_VAE_PATH = "D:\\.cache\\huggingface\\hub\\models--Qwen--Qwen-Image\\snapshots\\75e0b4be04f60ec59a75f475837eced720f823b6"
DEFAULT_BRANCHES = [
    "base_suffix:",
    "skc_5_8:5,6,7,8",
    "skc_5_only:5",
    "skc_6_only:6",
    "skc_7_only:7",
    "skc_8_only:8",
    "skc_7_8:7,8",
]


def parse_branch(value: str, fork_after: int, steps: int) -> tuple[str, set[int]]:
    if ":" not in value:
        raise ValueError(f"Branch must be name:steps, got {value!r}")
    name, step_text = value.split(":", 1)
    active = set()
    if step_text.strip():
        for part in step_text.split(","):
            idx = int(part.strip())
            if idx < 1 or idx > steps:
                raise ValueError(f"Step {idx} outside 1..{steps} in {value!r}")
            if idx <= fork_after:
                raise ValueError(f"Branch step {idx} is before/at fork-after {fork_after}: {value!r}")
            active.add(idx)
    return name, active


def tensor_stats(tensor: torch.Tensor) -> OrderedDict:
    t = tensor.detach().float().cpu()
    return OrderedDict(
        [
            ("shape", list(t.shape)),
            ("mean", float(t.mean())),
            ("std", float(t.std(unbiased=False))),
            ("rms", rms(t)),
            ("min", float(t.min())),
            ("max", float(t.max())),
        ]
    )


def build_sampling_args(args):
    # build_model() is shared with the capture script and expects this shape.
    args.cached_prompt = True
    args.negative_prompt = ""
    args.skip_vae = args.no_decode
    args.image = Path("loras/krea_vector_explore/test.png")
    args.prompt = ""
    args.prompt_id = args.prompt_ids[0]
    args.negative_prompt = ""
    args.quantize_te = True
    args.qtype_te = "qfloat8"
    args.low_vram = args.low_vram
    args.layer_offloading = args.layer_offloading
    args.layer_offloading_transformer_percent = args.layer_offloading_transformer_percent
    args.layer_offloading_text_encoder_percent = args.layer_offloading_text_encoder_percent
    args.max_text_length = 512
    args.checkpoint_filename = None
    args.text_encoder_path = None
    return args


def schedule_for(model, width: int, height: int, steps: int) -> list[float]:
    transformer = model.model
    patch = model.patch_size
    ae_scale = model.vae_scale_factor
    mkw = model.model_config.model_kwargs
    y1 = float(mkw.get("schedule_y1", 0.5))
    y2 = float(mkw.get("schedule_y2", 1.15))
    minres = int(mkw.get("schedule_min_res", 256))
    maxres = int(mkw.get("schedule_max_res", 1280))
    mu = mkw.get("schedule_mu", None)
    mu = float(mu) if mu is not None else None
    gh = height // (ae_scale * patch)
    gw = width // (ae_scale * patch)
    align = ae_scale * patch
    x1 = (minres // align) ** 2
    x2 = (maxres // align) ** 2
    return timesteps(gh * gw, steps, x1, x2, y1=y1, y2=y2, mu=mu)


def denoise_step(model, latents, tcurr: float, tprev: float, context, text_mask, vector, strength: float):
    device = model.device_torch
    dtype = model.torch_dtype
    t = torch.full((latents.shape[0],), float(tcurr), dtype=dtype, device=device)
    with projector_delta_forward(model.model.txtfusion.projector, vector, strength):
        with torch.no_grad():
            velocity = predict_velocity(model.model, latents.to(dtype), t, context, text_mask)
    next_latents = latents + (float(tprev) - float(tcurr)) * velocity.to(torch.float32)
    return next_latents, velocity.detach().cpu()


def prefix_path_for(root: Path, prompt_id: str, fork_after: int) -> Path:
    return root / prompt_id / f"prefix_after_step_{fork_after}.pt"


def legacy_prefix_path_for(root: Path, prompt_id: str) -> Path:
    return root / prompt_id / "prefix_after_step_4.pt"


def save_image(model, latents: torch.Tensor, path: Path):
    with torch.no_grad():
        images = model.decode_latents(latents, device=model.device_torch, dtype=model.torch_dtype)
    images = images.float().clamp(-1.0, 1.0)
    images = ((images + 1.0) * 127.5).round().to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    from PIL import Image

    Image.fromarray(images[0]).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an 8-step Krea2 prefix fork for late-step SKC emotion experiments.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--prompt-ids", nargs="+", default=DEFAULT_PROMPT_IDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--lora-strength", type=float, default=0.01)
    parser.add_argument("--branches", nargs="+", default=DEFAULT_BRANCHES)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--fork-after", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--vae-dtype", default="bf16")
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qtype", default="float8")
    parser.add_argument("--vae-path", default=DEFAULT_VAE_PATH)
    parser.add_argument("--schedule-mu", type=float, default=1.15, help="Pinned Krea Turbo time-shift mu. Use none only for dynamic raw schedule experiments.")
    parser.add_argument("--schedule-y1", type=float, default=None)
    parser.add_argument("--schedule-y2", type=float, default=None)
    parser.add_argument("--schedule-min-res", type=int, default=None)
    parser.add_argument("--schedule-max-res", type=int, default=None)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layer-offloading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layer-offloading-transformer-percent", type=float, default=0.0)
    parser.add_argument("--layer-offloading-text-encoder-percent", type=float, default=0.0)
    parser.add_argument("--no-decode", action="store_true", default=False)
    parser.add_argument("--prefix-dir", type=Path, default=None, help="Read saved prefix latents from this directory instead of recomputing them.")
    parser.add_argument("--reuse-prefix", action="store_true", default=False, help="Reuse matching prefix latents already present under --output-dir.")
    parser.add_argument("--prefix-only", action="store_true", default=False, help="Save prefix latents and skip suffix branch generation.")
    parser.add_argument("--skip-existing", action="store_true", default=False, help="Skip branch outputs that already exist on disk.")
    args = build_sampling_args(parser.parse_args())

    if args.fork_after <= 0 or args.fork_after >= args.steps:
        raise ValueError("--fork-after must be between 1 and steps-1")
    branches = [parse_branch(item, args.fork_after, args.steps) for item in args.branches]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Loading Krea2 transformer/VAE for suffix fork...")
    flush()
    model = build_model(args)
    print("Model loaded")
    flush()

    source, projector_vector = extract_projector_vector(load_file(str(args.lora)))
    projector_vector = projector_vector.float()
    ts = schedule_for(model, args.width, args.height, args.steps)
    latent_channels = model.model.config.channels
    shape = (1, latent_channels, args.height // model.vae_scale_factor, args.width // model.vae_scale_factor)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    start_latents = randn_tensor(shape, generator=generator, device=model.device_torch, dtype=torch.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = OrderedDict(
        [
            ("schema", "krea_emotion_suffix_fork.v1"),
            ("prompt_ids", args.prompt_ids),
            ("steps", args.steps),
            ("fork_after", args.fork_after),
            ("timesteps", [float(x) for x in ts]),
            ("seed", args.seed),
            ("width", args.width),
            ("height", args.height),
            ("lora", str(args.lora)),
            ("projector_vector_source", source),
            ("lora_strength", args.lora_strength),
            ("model_path", args.model_path),
            ("schedule_mu", args.schedule_mu),
            ("prefix_dir", str(args.prefix_dir) if args.prefix_dir else None),
            ("reuse_prefix", args.reuse_prefix),
            ("prefix_only", args.prefix_only),
            ("skip_existing", args.skip_existing),
            ("prefixes", []),
            ("branches", []),
        ]
    )

    for prompt_id in args.prompt_ids:
        context, text_mask = load_cached_prompt_features(args.features, prompt_id, model.device_torch, model.torch_dtype)
        prompt_dir = args.output_dir / prompt_id
        prompt_dir.mkdir(parents=True, exist_ok=True)
        output_prefix_path = prefix_path_for(args.output_dir, prompt_id, args.fork_after)
        input_prefix_path = None
        if args.prefix_dir is not None:
            input_prefix_path = prefix_path_for(args.prefix_dir, prompt_id, args.fork_after)
            if not input_prefix_path.exists() and args.fork_after == 4:
                legacy = legacy_prefix_path_for(args.prefix_dir, prompt_id)
                input_prefix_path = legacy if legacy.exists() else input_prefix_path
        elif args.reuse_prefix:
            input_prefix_path = output_prefix_path
            if not input_prefix_path.exists() and args.fork_after == 4:
                legacy = legacy_prefix_path_for(args.output_dir, prompt_id)
                input_prefix_path = legacy if legacy.exists() else input_prefix_path

        prefix_velocities = []
        if input_prefix_path is not None and input_prefix_path.exists():
            print(f"Loading saved prefix for {prompt_id}: {input_prefix_path}")
            flush()
            prefix_payload = torch.load(input_prefix_path, map_location="cpu", weights_only=False)
            if int(prefix_payload.get("step", args.fork_after)) != args.fork_after:
                raise ValueError(f"Prefix {input_prefix_path} was saved at step {prefix_payload.get('step')}, expected {args.fork_after}")
            prefix = prefix_payload["latents"].to(model.device_torch, dtype=torch.float32)
            prefix_velocities = prefix_payload.get("velocities", [])
        else:
            print(f"Prefix base steps for {prompt_id}...")
            flush()
            prefix = start_latents.clone()
            for step_idx, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:]), start=1):
                if step_idx > args.fork_after:
                    break
                prefix, velocity = denoise_step(model, prefix, tcurr, tprev, context, text_mask, None, 0.0)
                prefix_velocities.append(velocity)
            torch.save(
                OrderedDict([
                    ("schema", "krea_emotion_suffix_prefix.v2"),
                    ("prompt_id", prompt_id),
                    ("latents", prefix.detach().cpu()),
                    ("velocities", prefix_velocities),
                    ("step", args.fork_after),
                    ("steps", args.steps),
                    ("timesteps", [float(x) for x in ts]),
                    ("seed", args.seed),
                    ("width", args.width),
                    ("height", args.height),
                ]),
                output_prefix_path,
            )
            if args.fork_after == 4:
                legacy_path = legacy_prefix_path_for(args.output_dir, prompt_id)
                if legacy_path != output_prefix_path:
                    torch.save(torch.load(output_prefix_path, map_location="cpu", weights_only=False), legacy_path)
        manifest["prefixes"].append(
            OrderedDict([
                ("prompt_id", prompt_id),
                ("path", str(output_prefix_path if input_prefix_path is None else input_prefix_path)),
                ("reused", bool(input_prefix_path is not None and input_prefix_path.exists())),
                ("latent_stats", tensor_stats(prefix)),
            ])
        )
        if args.prefix_only:
            continue

        for branch_name, active_steps in branches:
            latent_path = prompt_dir / f"{branch_name}.pt"
            image_path = prompt_dir / f"{branch_name}.png"
            if args.skip_existing and latent_path.exists() and (args.no_decode or image_path.exists()):
                print(f"  Branch {branch_name}: existing output found, skipping")
                flush()
                existing = torch.load(latent_path, map_location="cpu", weights_only=False)
                manifest["branches"].append(
                    OrderedDict([
                        ("prompt_id", prompt_id),
                        ("branch", branch_name),
                        ("active_skc_steps", sorted(active_steps)),
                        ("latent", str(latent_path)),
                        ("image", str(image_path) if not args.no_decode else None),
                        ("latent_stats", existing.get("latent_stats", tensor_stats(existing["latents"]))),
                        ("skipped_existing", True),
                    ])
                )
                continue
            print(f"  Branch {branch_name}: SKC steps {sorted(active_steps) if active_steps else 'none'}")
            flush()
            latents = prefix.clone()
            suffix_velocities = []
            for step_idx, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:]), start=1):
                if step_idx <= args.fork_after:
                    continue
                use_lora = step_idx in active_steps
                latents, velocity = denoise_step(
                    model,
                    latents,
                    tcurr,
                    tprev,
                    context,
                    text_mask,
                    projector_vector if use_lora else None,
                    args.lora_strength if use_lora else 0.0,
                )
                suffix_velocities.append(velocity)
            torch.save(
                OrderedDict([
                    ("schema", "krea_emotion_suffix_branch.v1"),
                    ("prompt_id", prompt_id),
                    ("branch", branch_name),
                    ("active_skc_steps", sorted(active_steps)),
                    ("latents", latents.detach().cpu()),
                    ("suffix_velocities", suffix_velocities),
                    ("latent_stats", tensor_stats(latents)),
                ]),
                latent_path,
            )
            if not args.no_decode:
                save_image(model, latents, image_path)
            manifest["branches"].append(
                OrderedDict([
                    ("prompt_id", prompt_id),
                    ("branch", branch_name),
                    ("active_skc_steps", sorted(active_steps)),
                    ("latent", str(latent_path)),
                    ("image", str(image_path) if not args.no_decode else None),
                    ("latent_stats", tensor_stats(latents)),
                ])
            )
            del latents
            del suffix_velocities
            flush(garbage_collect=False)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
