"""Binary search for the SKC3VO projector-vector flip threshold.

Loads Krea2 once, then bisects between a known-clean strength and a
known-flipped strength by measuring the velocity delta RMS at each midpoint.
The flip manifests as a step-change in that RMS — near zero below, suddenly
large above.
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capture_krea_txtfusion_forward import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_LORA,
    DEFAULT_MODEL,
    build_model,
    load_cached_prompt_features,
    run_forward,
)
from safetensors.torch import load_file
from sweep_skc3vo_blank_vector import extract_projector_vector, rms
from toolkit.basic import flush


DEFAULT_OUTPUT = Path("loras/krea_vector_explore/txtfusion_probe/captures/krea_filter_flip.json")
DEFAULT_PROBE_PROMPTS = ["blank", "exposed_wide"]


def velocity_delta_rms(model, noisy_latents, t, context, text_mask, projector_vector, strength):
    base_vel, _ = run_forward(model, noisy_latents, t, context, text_mask,
                              projector_vector=None, lora_strength=0.0)
    lora_vel, _ = run_forward(model, noisy_latents, t, context, text_mask,
                              projector_vector=projector_vector, lora_strength=strength)
    return rms((lora_vel - base_vel).float()), base_vel, lora_vel


def main():
    parser = argparse.ArgumentParser(description="Binary-search for Krea2 projector filter flip threshold.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--probe-prompts", nargs="+", default=DEFAULT_PROBE_PROMPTS)
    parser.add_argument("--low", type=float, default=0.0, help="Known-clean strength (no flip)")
    parser.add_argument("--high", type=float, default=0.10, help="Known-flipped strength")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--flip-ratio", type=float, default=0.5,
                        help="delta_rms / high_delta_rms threshold to call a point 'flipped'")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--noise-t", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--qtype", default="float8")
    parser.add_argument("--layer-offloading", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layer-offloading-transformer-percent", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.cached_prompt = True
    args.negative_prompt = ""
    args.skip_vae = True
    args.quantize = True
    args.quantize_te = True
    args.qtype_te = "qfloat8"
    args.low_vram = False
    args.layer_offloading_text_encoder_percent = 1.0
    args.max_text_length = 512
    args.checkpoint_filename = None
    args.text_encoder_path = None
    args.vae_path = None
    args.vae_dtype = "bf16"

    print("Loading model...")
    flush()
    model = build_model(args)
    print("Model loaded")
    flush()

    _, projector_vector = extract_projector_vector(load_file(str(args.lora)))
    projector_vector = projector_vector.to(model.device_torch, dtype=model.torch_dtype)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent_channels = model.model.config.channels
    noise = torch.randn(
        1, latent_channels,
        args.height // model.vae_scale_factor,
        args.width // model.vae_scale_factor,
        generator=generator, dtype=torch.float32,
    )
    noisy_latents = noise.to(model.device_torch, dtype=model.torch_dtype)
    t = torch.full((1,), args.noise_t, device=model.device_torch, dtype=model.torch_dtype)

    results = {}
    for prompt_id in args.probe_prompts:
        print(f"\n--- Probe: {prompt_id} ---")
        flush()
        context, text_mask = load_cached_prompt_features(
            args.features, prompt_id, model.device_torch, model.torch_dtype)

        # Measure delta at high to establish the "fully flipped" baseline
        high_rms, _, _ = velocity_delta_rms(
            model, noisy_latents, t, context, text_mask, projector_vector, args.high)
        print(f"  s={args.high:.4f}  delta_rms={high_rms:.6f}  (flipped baseline)")
        flush()

        if high_rms == 0.0:
            print("  WARNING: no delta at high strength — check LoRA/model")
            results[prompt_id] = None
            continue

        flip_threshold = high_rms * args.flip_ratio

        lo, hi = args.low, args.high
        history = []
        for i in range(args.iterations):
            mid = (lo + hi) / 2.0
            mid_rms, _, _ = velocity_delta_rms(
                model, noisy_latents, t, context, text_mask, projector_vector, mid)
            flipped = mid_rms >= flip_threshold
            history.append(OrderedDict([
                ("iteration", i + 1),
                ("strength", mid),
                ("delta_rms", float(mid_rms)),
                ("ratio_to_high", float(mid_rms / high_rms)),
                ("flipped", flipped),
            ]))
            print(f"  [{i+1}/{args.iterations}] s={mid:.5f}  delta_rms={mid_rms:.6f}"
                  f"  ratio={mid_rms/high_rms:.3f}  {'FLIPPED' if flipped else 'clean'}")
            flush()
            if flipped:
                hi = mid
            else:
                lo = mid

        flip_point = (lo + hi) / 2.0
        print(f"  => flip point ~ {flip_point:.5f}  (bracket [{lo:.5f}, {hi:.5f}])")
        flush()
        results[prompt_id] = OrderedDict([
            ("flip_point", flip_point),
            ("bracket_lo", lo),
            ("bracket_hi", hi),
            ("high_delta_rms", float(high_rms)),
            ("flip_threshold_rms", float(flip_threshold)),
            ("history", history),
        ])

    report = OrderedDict([
        ("schema", "krea_filter_flip_search.v1"),
        ("lora", str(args.lora)),
        ("noise_t", args.noise_t),
        ("seed", args.seed),
        ("low", args.low),
        ("high", args.high),
        ("iterations", args.iterations),
        ("flip_ratio", args.flip_ratio),
        ("results", results),
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
