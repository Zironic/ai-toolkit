"""Measure which Krea 2 transformer blocks carry the SKC3VO projector signal.

For each of 28 SingleStreamBlocks × N Turbo timesteps, captures the residual-stream
output with and without the SKC projector vector, then computes:

  relative_rms   = rms(skc_act - base_act) / rms(base_act)
  cosine_change  = 1 - cosine(base_act, skc_act)

Separately for text tokens, all image tokens, and three rough spatial regions
of the image token grid (upper / middle / lower thirds of the 32x32 patch grid).

Also computes the final velocity delta in latent spatial coordinates so you can
see WHERE in the image the SKC effect is concentrated, not just which block.

Output: JSON heatmap + a printed table.

Usage:
  venv/Scripts/python.exe scripts/measure_krea_block_heatmap.py
  venv/Scripts/python.exe scripts/measure_krea_block_heatmap.py \\
      --prompt-ids exposed_wide sfw_wide \\
      --strength 0.05 \\
      --timesteps 1.0 0.760 0.513

NOTE: This is Phase 1 (measurement only). Causal patching is Phase 2.
"""
import argparse
import json
import sys
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import torch

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
from safetensors.torch import load_file  # noqa: E402
from sweep_skc3vo_blank_vector import extract_projector_vector  # noqa: E402
from toolkit.basic import flush  # noqa: E402


DEFAULT_MODEL = "krea/Krea-2-Turbo"
DEFAULT_OUTPUT = Path("loras/krea_vector_explore/txtfusion_probe/block_heatmap.json")
DEFAULT_PROMPT_IDS = ["exposed_wide"]
DEFAULT_STRENGTH = 0.05

# Turbo 4-step schedule at 512x512 (mu=1.15): timestep entry points
# from timesteps(1024, 4, 256, 6400, mu=1.15) = [1.0, 0.905, 0.760, 0.513, 0.0]
DEFAULT_TIMESTEPS = [1.0, 0.905, 0.760, 0.513]

# Spatial region boundaries in the 32x32 patch grid (row index)
# For a "full body" shot in portrait orientation: top=face/hair, mid=torso/clothing, bot=legs
_GRID_H = 32
_REGION_UPPER = (0, 11)    # rows 0-10   → "face"
_REGION_MID   = (11, 22)   # rows 11-21  → "clothing"
_REGION_LOWER = (22, 32)   # rows 22-31  → "lower_body"

_MASKS = [
    ("face",       _REGION_UPPER),
    ("clothing",   _REGION_MID),
    ("lower_body", _REGION_LOWER),
]


def rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt())


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    denom = float((a.norm() * b.norm()).clamp_min(1e-12))
    return float((a * b).sum()) / denom


def img_region_slice(img_tokens: torch.Tensor, row_lo: int, row_hi: int) -> torch.Tensor:
    """Slice tokens from a rasterized 32x32 patch grid by row range.

    img_tokens: [B, H*W, features] where H=W=32 (1024 tokens total)
    Returns tokens in rows [row_lo, row_hi).
    """
    indices = [r * _GRID_H + c for r in range(row_lo, row_hi) for c in range(_GRID_H)]
    return img_tokens[:, indices, :]


# ---------------------------------------------------------------------------
# Block capture context manager
# ---------------------------------------------------------------------------


class BlockCapture:
    """Hook all SingleStreamBlocks and capture their combined-sequence output.

    Only the image-token portion is stored (text tokens are a small fraction
    of the value we care about for spatial clothing analysis, but we do store
    a compact txt RMS for reference).
    """

    def __init__(self, blocks: torch.nn.ModuleList, txtlen: int):
        self.blocks = blocks
        self.txtlen = txtlen
        self.img_caps: List[Optional[torch.Tensor]] = [None] * len(blocks)
        self.txt_rms: List[float] = [0.0] * len(blocks)
        self._handles = []

    def __enter__(self):
        for i, block in enumerate(self.blocks):
            self._handles.append(block.register_forward_hook(self._make_hook(i)))
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, idx: int):
        tl = self.txtlen

        def hook(module, inputs, output):
            combined = output.detach()
            # combined: [B, txtlen+imglen(+padding), features]
            img = combined[:, tl : tl + 1024, :].cpu().half()  # always 1024 img tokens
            txt = combined[:, :tl, :]
            self.img_caps[idx] = img
            self.txt_rms[idx] = rms(txt)

        return hook


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def block_metrics(base_img: torch.Tensor, skc_img: torch.Tensor) -> OrderedDict:
    """Per-block activation difference metrics with spatial localization.

    base_img / skc_img: [B, 1024, features] on CPU float16

    For each named mask (face / clothing / lower_body) computes:
      inside_mask_rms  - delta RMS within the mask region
      outside_mask_rms - delta RMS in all OTHER regions combined
      leakage_ratio    - outside / inside  (lower = more localised to this region)
    """
    b = base_img.float()
    s = skc_img.float()
    delta = s - b

    def _region_delta(row_lo, row_hi):
        indices = [r * _GRID_H + c for r in range(row_lo, row_hi) for c in range(_GRID_H)]
        return delta[:, indices, :]

    def _region_stats_full(b_r, s_r, d_r):
        base_r_rms = rms(b_r)
        rel = rms(d_r) / max(base_r_rms, 1e-12)
        cos_chg = 1.0 - cosine_sim(b_r, s_r)
        return OrderedDict([
            ("relative_rms", round(float(rel), 6)),
            ("cosine_change", round(float(cos_chg), 6)),
            ("base_rms", round(float(base_r_rms), 6)),
            ("delta_rms", round(float(rms(d_r)), 6)),
        ])

    # Global (all image tokens)
    all_stats = _region_stats_full(b, s, delta)

    # Per-mask: inside + outside + leakage
    mask_entries = OrderedDict()
    for mask_name, (row_lo, row_hi) in _MASKS:
        inside_d  = _region_delta(row_lo, row_hi)
        # outside = complement rows
        out_rows = [(0, row_lo), (row_hi, _GRID_H)]
        outside_indices = [
            r * _GRID_H + c
            for lo, hi in out_rows
            for r in range(lo, hi)
            for c in range(_GRID_H)
        ]
        outside_d = delta[:, outside_indices, :] if outside_indices else delta[:, :0, :]

        inside_rms  = rms(inside_d)
        outside_rms = rms(outside_d) if outside_d.numel() > 0 else 0.0
        leakage     = outside_rms / max(inside_rms, 1e-12)

        inside_b  = img_region_slice(b, row_lo, row_hi)
        inside_s  = img_region_slice(s, row_lo, row_hi)
        mask_entries[mask_name] = OrderedDict([
            ("inside_mask_rms",  round(inside_rms,  6)),
            ("outside_mask_rms", round(outside_rms, 6)),
            ("leakage_ratio",    round(leakage,     6)),
            ("relative_rms",     round(inside_rms / max(rms(inside_b), 1e-12), 6)),
            ("cosine_change",    round(1.0 - cosine_sim(inside_b, inside_s), 6)),
        ])

    result = OrderedDict([("all_img", all_stats)])
    result.update(mask_entries)
    return result


def velocity_spatial_metrics(base_vel: torch.Tensor, skc_vel: torch.Tensor) -> OrderedDict:
    """Velocity delta in latent spatial coordinates.

    base_vel / skc_vel: [B, C, H, W] latent velocity (C=16, H=W=64 for 512x512)
    Spatial splits (rows): face/hair=top 1/4, clothing=middle 1/2, lower=bottom 1/4
    """
    b = base_vel.float()
    s = skc_vel.float()
    d = s - b
    H = b.shape[2]
    q = H // 4

    def _vstats(bv, sv, dv):
        return OrderedDict([
            ("relative_rms", round(float(rms(dv) / max(rms(bv), 1e-12)), 6)),
            ("delta_rms", round(float(rms(dv)), 6)),
            ("cosine_change", round(float(1.0 - cosine_sim(bv, sv)), 6)),
        ])

    return OrderedDict([
        ("full", _vstats(b, s, d)),
        ("upper", _vstats(b[:, :, :q, :], s[:, :, :q, :], d[:, :, :q, :])),
        ("mid", _vstats(b[:, :, q:3*q, :], s[:, :, q:3*q, :], d[:, :, q:3*q, :])),
        ("lower", _vstats(b[:, :, 3*q:, :], s[:, :, 3*q:, :], d[:, :, 3*q:, :])),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block-level activation heatmap for SKC3VO projector vector effect."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--prompt-ids", nargs="+", default=DEFAULT_PROMPT_IDS)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--timesteps", nargs="+", type=float, default=DEFAULT_TIMESTEPS,
                        help="Turbo timesteps to probe (default: 4-step Turbo schedule)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qtype", default="float8")
    parser.add_argument("--layer-offloading", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layer-offloading-transformer-percent", type=float, default=0.5)
    args = parser.parse_args()

    args.cached_prompt = True
    args.negative_prompt = ""
    args.skip_vae = True
    args.vae_dtype = "bf16"
    args.quantize_te = True
    args.qtype_te = "qfloat8"
    args.low_vram = False
    args.layer_offloading_text_encoder_percent = 1.0
    args.max_text_length = 2000
    args.checkpoint_filename = None
    args.text_encoder_path = None
    args.vae_path = None

    print("Loading model...")
    flush()
    model = build_model(args)
    print("Model loaded")
    flush()

    _, proj_vector = extract_projector_vector(load_file(str(args.lora)))
    proj_vector = proj_vector.to(model.device_torch, dtype=model.torch_dtype)

    latent_h = args.height // model.vae_scale_factor
    latent_w = args.width // model.vae_scale_factor
    latent_c = model.model.config.channels
    n_blocks = len(model.model.blocks)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    starting_noise = torch.randn(
        1, latent_c, latent_h, latent_w, generator=generator, dtype=torch.float32
    )

    all_results = {}

    for prompt_id in args.prompt_ids:
        print(f"\n=== Prompt: {prompt_id} ===")
        flush()
        context, text_mask = load_cached_prompt_features(
            args.features, prompt_id, model.device_torch, model.torch_dtype
        )
        txtlen = text_mask.shape[1]
        print(f"  txtlen={txtlen}, n_blocks={n_blocks}")
        flush()

        prompt_results = []
        blocks = model.model.blocks
        projector = model.model.txtfusion.projector

        # Walk the Turbo trajectory from t=1.0 → later timesteps.
        # Each step advances the base latent so hooks see realistic intermediate states.
        lat = starting_noise.to(model.device_torch, dtype=model.torch_dtype)

        # Sort and process timesteps in descending order (1.0 first)
        probe_ts = sorted(args.timesteps, reverse=True)

        for step_idx, t_val in enumerate(probe_ts):
            t = torch.full((1,), t_val, device=model.device_torch, dtype=model.torch_dtype)
            print(f"  t={t_val:.3f}  capturing base + SKC block activations...")
            flush()

            # --- Base forward ---
            with BlockCapture(blocks, txtlen) as base_cap:
                with torch.no_grad():
                    base_vel = predict_velocity(
                        model.model, lat, t, context, text_mask
                    ).detach().cpu()

            # --- SKC forward ---
            with BlockCapture(blocks, txtlen) as skc_cap:
                with projector_delta_forward(projector, proj_vector, args.strength):
                    with torch.no_grad():
                        skc_vel = predict_velocity(
                            model.model, lat, t, context, text_mask
                        ).detach().cpu()

            # Per-block metrics
            block_rows = []
            for k in range(n_blocks):
                if base_cap.img_caps[k] is None or skc_cap.img_caps[k] is None:
                    block_rows.append(None)
                    continue
                m = block_metrics(base_cap.img_caps[k], skc_cap.img_caps[k])
                m["txt_rms_base"] = round(float(base_cap.txt_rms[k]), 6)
                m["txt_rms_skc"] = round(float(skc_cap.txt_rms[k]), 6)
                block_rows.append(m)

            # Velocity spatial metrics
            vel_m = velocity_spatial_metrics(base_vel, skc_vel)

            t_entry = OrderedDict([
                ("timestep", t_val),
                ("velocity", vel_m),
                ("blocks", block_rows),
            ])
            prompt_results.append(t_entry)

            # Print top 5 blocks by relative_rms (all_img)
            ranked = sorted(
                [(k, r["all_img"]["relative_rms"]) for k, r in enumerate(block_rows) if r],
                key=lambda x: x[1], reverse=True,
            )
            print(f"    Velocity full relative_rms = {vel_m['full']['relative_rms']:.4f}")
            print(f"    Velocity spatial: face={vel_m['upper']['relative_rms']:.4f}  "
                  f"clothing={vel_m['mid']['relative_rms']:.4f}  "
                  f"lower={vel_m['lower']['relative_rms']:.4f}")
            print(f"    Top 5 blocks by relative_rms:")
            for k, rel in ranked[:5]:
                clothing_rel = block_rows[k]["clothing"]["relative_rms"]
                clothing_leak = block_rows[k]["clothing"]["leakage_ratio"]
                face_rel = block_rows[k]["face"]["relative_rms"]
                print(f"      block {k:2d}: all={rel:.4f}  clothing={clothing_rel:.4f}  "
                      f"leak={clothing_leak:.3f}  face={face_rel:.4f}")
            flush()

            # Advance latent with base velocity (Euler step) to next timestep
            if step_idx + 1 < len(probe_ts):
                t_next = probe_ts[step_idx + 1]
                dt = t_next - t_val  # negative (going toward 0)
                lat = lat + dt * base_vel.to(model.device_torch, dtype=model.torch_dtype)

        all_results[prompt_id] = prompt_results

    # Build compact ranked summary + flat rows across all timesteps and prompts
    block_summary = []
    flat_rows = []
    for k in range(n_blocks):
        rel_vals, clothing_rel_vals, clothing_leak_vals, face_rel_vals = [], [], [], []
        for pid, pr in all_results.items():
            for t_entry in pr:
                row = t_entry["blocks"][k]
                if row is None:
                    continue
                t = t_entry["timestep"]
                rel_vals.append(row["all_img"]["relative_rms"])
                clothing_rel_vals.append(row["clothing"]["relative_rms"])
                clothing_leak_vals.append(row["clothing"]["leakage_ratio"])
                face_rel_vals.append(row["face"]["relative_rms"])

                for mask_name in ("face", "clothing", "lower_body"):
                    m = row[mask_name]
                    flat_rows.append(OrderedDict([
                        ("prompt_id",        pid),
                        ("token_span",       "img_tokens"),
                        ("mask_name",        mask_name),
                        ("layer",            k),
                        ("timestep",         t),
                        ("inside_mask_rms",  m["inside_mask_rms"]),
                        ("outside_mask_rms", m["outside_mask_rms"]),
                        ("leakage_ratio",    m["leakage_ratio"]),
                        ("relative_rms",     m["relative_rms"]),
                        ("cosine_change",    m["cosine_change"]),
                    ]))

        if rel_vals:
            block_summary.append(OrderedDict([
                ("block",                    k),
                ("mean_relative_rms",        round(sum(rel_vals) / len(rel_vals), 6)),
                ("mean_clothing_relative",   round(sum(clothing_rel_vals) / len(clothing_rel_vals), 6)),
                ("mean_clothing_leakage",    round(sum(clothing_leak_vals) / len(clothing_leak_vals), 6)),
                ("mean_face_relative",       round(sum(face_rel_vals) / len(face_rel_vals), 6)),
            ]))

    block_summary.sort(key=lambda x: x["mean_relative_rms"], reverse=True)

    print("\n=== Block ranking (mean relative_rms, all image tokens) ===")
    for entry in block_summary[:10]:
        print(f"  block {entry['block']:2d}: all={entry['mean_relative_rms']:.4f}  "
              f"clothing={entry['mean_clothing_relative']:.4f}  "
              f"leak={entry['mean_clothing_leakage']:.3f}  "
              f"face={entry['mean_face_relative']:.4f}")
    flush()

    report = OrderedDict([
        ("schema", "krea_block_heatmap.v2"),
        ("model_path", args.model_path),
        ("lora", str(args.lora)),
        ("strength", args.strength),
        ("seed", args.seed),
        ("prompt_ids", args.prompt_ids),
        ("timesteps", args.timesteps),
        ("n_blocks", n_blocks),
        ("spatial_regions", OrderedDict([
            ("face_rows",        list(_REGION_UPPER)),
            ("clothing_rows",    list(_REGION_MID)),
            ("lower_body_rows",  list(_REGION_LOWER)),
            ("note", "Rows of 32x32 patch grid (face=0-10, clothing=11-21, lower_body=22-31)"),
        ])),
        ("block_summary_by_mean_rms", block_summary),
        ("flat_rows", flat_rows),
        ("per_prompt", all_results),
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    flush()


if __name__ == "__main__":
    main()
