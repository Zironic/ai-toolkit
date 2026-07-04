"""Krea 2 Turbo visual scorer for projector vector candidates.

Runs N-step Turbo denoising for each (candidate, prompt, seed) and saves
images to <output-dir>/<candidate_id>/<prompt_id>_seed<seed>.png for visual
inspection. Also writes scores.json with per-step velocity metrics and
final-latent x0 diff vs base.

Each candidate is a STEP-GATED recipe: a per-step rule ("base" / "full_skc" /
"masked_skc") rather than one blanket mode for the whole trajectory. This
tests whether temporal gating alone (SKC only active after composition is
already committed at step 1) fixes the zoom/framing collapse we observed when
SKC runs the whole trajectory, without needing per-slot coefficient surgery.
See tasks/open/KREA2_FILTER_BYPASS_PLAN.md for the reasoning.

Default candidates (steps = args.steps, "rest" = steps 2..N):
  - base                      : base every step (reference)
  - skc_all                   : full SKC every step (old "original")
  - masked_skc_all            : masked-delta SKC every step (old "masked_skc")
  - base1_skc_rest            : base step 1 (locks composition), full SKC after
  - skc1_base_rest            : full SKC step 1 only, base after
  - base1_skc2_base_rest      : base, full SKC (step 2 only), base, base, ...
  - base1_masked_rest         : base step 1, masked SKC after
  - base1_masked2_base_rest   : base, masked SKC (step 2 only), base, base, ...

--subset-json/--top-k candidates (if given) still run full_skc every step,
using their own decomposed vector instead of the main --lora vector.
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import time

import torch
from PIL import Image
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
from extensions_built_in.diffusion_models.krea2.src.pipeline import (  # noqa: E402
    predict_velocity,
    timesteps,
)
from sweep_skc3vo_blank_vector import extract_projector_vector  # noqa: E402
from toolkit.basic import flush  # noqa: E402


DEFAULT_MODEL = "krea/Krea-2-Turbo"
DEFAULT_OUTPUT = Path("loras/krea_vector_explore/txtfusion_probe/turbo_score")
DEFAULT_SUBSET_JSON = Path(
    "loras/krea_vector_explore/txtfusion_probe/captures/krea_txtfusion_projector_subset_search.json"
)
DEFAULT_VAE = (
    "D:/.cache/huggingface/hub/models--Qwen--Qwen-Image"
    "/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6"
)
DEFAULT_PROBE_PROMPTS = ["blank", "sfw_wide", "exposed_wide"]
DEFAULT_SEEDS = [12345, 54321]

# Turbo schedule constants at 512x512 (patch=2, vae=f8)
# img_tokens = (512//(8*2))^2 = 1024; x1=(256//(8*2))^2=256; x2=(1280//(8*2))^2=6400
_TURBO_SEQ_LEN = 1024
_TURBO_X1 = 256
_TURBO_X2 = 6400
_TURBO_MU = 1.15

# Clothing region in latent pixel coordinates (64x64 latent at 512x512).
# Patch-grid rows 11-21 (clothing) * patch_size 2 = latent rows 22:44.
_CLOTHING_LATENT_ROWS = (22, 44)


def _step_gated_recipes(steps):
    """Return the fixed set of step-gated recipes, as {id: step_rules} for `steps` steps.

    step_rules[i] in {"base", "full_skc", "masked_skc"}. Steps are 0-indexed;
    "rest" means steps 1..steps-1 (i.e. every step after the first).
    """
    recipes = OrderedDict()
    recipes["base"] = ["base"] * steps
    recipes["skc_all"] = ["full_skc"] * steps
    recipes["masked_skc_all"] = ["masked_skc"] * steps
    recipes["base1_skc_rest"] = ["base"] + ["full_skc"] * (steps - 1)
    recipes["skc1_base_rest"] = ["full_skc"] + ["base"] * (steps - 1)
    recipes["base1_masked_rest"] = ["base"] + ["masked_skc"] * (steps - 1)
    if steps >= 3:
        recipes["base1_skc2_base_rest"] = ["base", "full_skc"] + ["base"] * (steps - 2)
        recipes["base1_masked2_base_rest"] = ["base", "masked_skc"] + ["base"] * (steps - 2)
    return recipes


def build_candidates(args, model):
    """Return list of candidate dicts: {id, step_rules, vector, strength, mask_rows}."""
    cands = []

    main_vec = None
    if args.lora and Path(args.lora).exists():
        _, main_vec = extract_projector_vector(load_file(str(args.lora)))
        main_vec = main_vec.to(model.device_torch, dtype=model.torch_dtype)

    for cid, rules in _step_gated_recipes(args.steps).items():
        needs_vector = any(r != "base" for r in rules)
        cands.append({
            "id": cid,
            "step_rules": rules,
            "vector": main_vec if needs_vector else None,
            "strength": args.strength if needs_vector else 0.0,
            "mask_rows": _CLOTHING_LATENT_ROWS if "masked_skc" in rules else None,
        })

    if args.subset_json and Path(args.subset_json).exists():
        data = json.loads(Path(args.subset_json).read_text(encoding="utf-8"))
        for rec in data.get("top_by_score", [])[: args.top_k]:
            vec = torch.tensor(
                rec["vector"], dtype=model.torch_dtype, device=model.device_torch
            )
            cands.append({
                "id": rec["id"],
                "step_rules": ["full_skc"] * args.steps,
                "vector": vec,
                "strength": args.strength,
                "mask_rows": None,
            })

    return cands


def _timed_forward(model, lat_typed, t, context, text_mask, projector, vec, strength, device):
    """Single predict_velocity call with wall-clock timing (synchronizes GPU before+after)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with projector_delta_forward(projector, vec, strength):
        with torch.no_grad():
            v = predict_velocity(model, lat_typed, t, context, text_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return v, time.perf_counter() - t0


def run_denoise(model, starting_noise, context, text_mask, projector_vector, strength, steps,
                step_rules, latent_mask_rows=None):
    """Euler-integrate for `steps` steps, applying a per-step rule.

    step_rules[i] in {"base", "full_skc", "masked_skc"} selects, for step i,
    whether to use the unperturbed velocity, the full-strength SKC velocity, or
    base + a spatially-masked SKC delta (latent_mask_rows=(lo, hi) required).
    Only the forward(s) a given step's rule actually needs are computed (a
    "base"-only step skips the SKC forward entirely, and vice versa).

    Returns (final_latent_f32, step_x0s, step_vel_rms, step_timings).
    final_latent is at t~0 and suitable for VAE decode.
    step_x0s[i] is the one-step x0 estimate at step i (CPU, float32), computed
    from whichever velocity was used that step.
    step_timings[i] is a dict with per-forward wall-clock seconds.
    """
    device = model.device_torch
    dtype = model.torch_dtype
    projector = model.model.txtfusion.projector

    assert len(step_rules) == steps, f"step_rules must have {steps} entries, got {len(step_rules)}"

    ts = timesteps(_TURBO_SEQ_LEN, steps, _TURBO_X1, _TURBO_X2, mu=_TURBO_MU)

    lat = starting_noise.to(device, dtype=torch.float32)
    step_x0s = []
    step_vel_rms = []
    step_timings = []

    for step_i, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
        t = torch.full((lat.shape[0],), tcurr, device=device, dtype=dtype)
        lat_typed = lat.to(dtype)
        timing = {"t": round(float(tcurr), 4), "rule": step_rules[step_i]}
        rule = step_rules[step_i]

        if rule == "base":
            base_v, t_base = _timed_forward(
                model.model, lat_typed, t, context, text_mask, projector, None, 0.0, device)
            v32 = base_v.float()
            timing["base_s"] = round(t_base, 3)
        elif rule == "full_skc":
            skc_v, t_skc = _timed_forward(
                model.model, lat_typed, t, context, text_mask, projector, projector_vector, strength, device)
            v32 = skc_v.float()
            timing["skc_s"] = round(t_skc, 3)
        elif rule == "masked_skc":
            assert latent_mask_rows is not None, "masked_skc rule requires latent_mask_rows"
            base_v, t_base = _timed_forward(
                model.model, lat_typed, t, context, text_mask, projector, None, 0.0, device)
            skc_v, t_skc = _timed_forward(
                model.model, lat_typed, t, context, text_mask, projector, projector_vector, strength, device)
            delta = skc_v.float() - base_v.float()
            masked_delta = torch.zeros_like(delta)
            lo, hi = latent_mask_rows
            masked_delta[:, :, lo:hi, :] = delta[:, :, lo:hi, :]
            v32 = base_v.float() + masked_delta
            timing["base_s"] = round(t_base, 3)
            timing["skc_s"] = round(t_skc, 3)
        else:
            raise ValueError(f"Unknown step rule: {rule!r}")

        step_x0s.append((lat - tcurr * v32).cpu())
        step_vel_rms.append(float(v32.pow(2).mean().sqrt()))
        step_timings.append(timing)
        lat = lat + (tprev - tcurr) * v32

    return lat, step_x0s, step_vel_rms, step_timings


def decode_to_pil(model, latent_f32):
    """Decode a float32 latent tensor to a PIL image. Returns (PIL image, decode_seconds)."""
    device = model.device_torch
    dtype = model.torch_dtype
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        imgs = model.decode_latents(latent_f32.to(device, dtype=dtype), device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_s = time.perf_counter() - t0
    imgs = imgs.float().clamp(-1.0, 1.0)
    imgs = ((imgs + 1.0) * 127.5).round().to(torch.uint8)
    return Image.fromarray(imgs.permute(0, 2, 3, 1).cpu().numpy()[0]), decode_s


def main():
    parser = argparse.ArgumentParser(
        description="Score Krea 2 Turbo projector vector candidates via saved images."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--subset-json", type=Path, default=DEFAULT_SUBSET_JSON)
    parser.add_argument("--top-k", type=int, default=5, help="Number of top subset candidates to include")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--probe-prompts", nargs="+", default=DEFAULT_PROBE_PROMPTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--strength", type=float, default=0.05,
                        help="LoRA strength applied to all non-base candidates")
    parser.add_argument("--steps", type=int, default=4, help="Number of denoising steps")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vae-path", default=DEFAULT_VAE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qtype", default="float8")
    parser.add_argument("--layer-offloading", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layer-offloading-transformer-percent", type=float, default=0.5)
    args = parser.parse_args()

    # Fields expected by build_model but not exposed as CLI args here
    args.cached_prompt = True
    args.negative_prompt = ""
    args.skip_vae = False  # need VAE for decoding
    args.vae_dtype = "bf16"
    args.quantize_te = True
    args.qtype_te = "qfloat8"
    args.low_vram = False
    args.layer_offloading_text_encoder_percent = 1.0
    args.max_text_length = 2000
    args.checkpoint_filename = None
    args.text_encoder_path = None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Krea 2 Turbo...")
    flush()
    model = build_model(args)
    print("Model loaded")
    flush()

    candidates = build_candidates(args, model)
    print(f"Candidates ({len(candidates)}): {[c['id'] for c in candidates]}")
    for c in candidates:
        print(f"  {c['id']:26s} step_rules={c['step_rules']}")
    print(f"Prompts: {args.probe_prompts}")
    print(f"Seeds: {args.seeds}")
    print(f"Steps: {args.steps}  Strength: {args.strength}")
    flush()

    latent_h = args.height // model.vae_scale_factor
    latent_w = args.width // model.vae_scale_factor
    latent_c = model.model.config.channels

    scores = {c["id"]: {} for c in candidates}
    total = len(args.seeds) * len(args.probe_prompts) * len(candidates)
    done = 0

    for seed in args.seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        starting_noise = torch.randn(
            1, latent_c, latent_h, latent_w,
            generator=generator, dtype=torch.float32,
        )

        for prompt_id in args.probe_prompts:
            context, text_mask = load_cached_prompt_features(
                args.features, prompt_id, model.device_torch, model.torch_dtype
            )

            base_final_x0 = None  # set on first (base) pass for diff metric

            for cand in candidates:
                cand_id = cand["id"]
                done += 1
                print(f"[{done}/{total}] {cand_id} [{','.join(cand['step_rules'])}] / {prompt_id} / seed={seed}")
                flush()

                final_lat, step_x0s, step_vel_rms, step_timings = run_denoise(
                    model, starting_noise, context, text_mask,
                    cand["vector"], cand["strength"], args.steps,
                    cand["step_rules"], latent_mask_rows=cand["mask_rows"],
                )

                # Print per-step timing
                for i, tm in enumerate(step_timings):
                    parts = [f"rule={tm['rule']}"]
                    if "base_s" in tm:
                        parts.append(f"base={tm['base_s']:.2f}s")
                    if "skc_s" in tm:
                        parts.append(f"skc={tm['skc_s']:.2f}s")
                    print(f"  step {i+1} t={tm['t']:.3f}: {'  '.join(parts)}")

                if cand_id == "base":
                    base_final_x0 = step_x0s[-1]

                img, decode_s = decode_to_pil(model, final_lat)
                total_fwd_s = sum(
                    (tm.get("base_s", 0) + tm.get("skc_s", 0))
                    for tm in step_timings
                )
                print(f"  decode={decode_s:.2f}s  total_fwd={total_fwd_s:.2f}s  "
                      f"total={total_fwd_s + decode_s:.2f}s")
                flush()
                out_dir = args.output_dir / cand_id
                out_dir.mkdir(parents=True, exist_ok=True)
                img.save(out_dir / f"{prompt_id}_seed{seed}.png")

                x0_diff_rms = None
                if cand_id != "base" and base_final_x0 is not None:
                    x0_diff_rms = float(
                        (step_x0s[-1].float() - base_final_x0.float()).pow(2).mean().sqrt()
                    )

                if prompt_id not in scores[cand_id]:
                    scores[cand_id][prompt_id] = {}
                scores[cand_id][prompt_id][str(seed)] = OrderedDict([
                    ("step_vel_rms", step_vel_rms),
                    ("x0_diff_rms_vs_base", x0_diff_rms),
                    ("step_timings", step_timings),
                    ("decode_s", round(decode_s, 3)),
                ])

    report = OrderedDict([
        ("schema", "krea_turbo_candidate_score.v1"),
        ("model_path", args.model_path),
        ("strength", args.strength),
        ("steps", args.steps),
        ("probe_prompts", args.probe_prompts),
        ("seeds", args.seeds),
        ("candidates", [
            OrderedDict([("id", c["id"]), ("step_rules", c["step_rules"])])
            for c in candidates
        ]),
        ("scores", scores),
    ])
    report_path = args.output_dir / "scores.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nImages saved to: {args.output_dir}")
    print(f"Scores written to: {report_path}")
    flush()


if __name__ == "__main__":
    main()
