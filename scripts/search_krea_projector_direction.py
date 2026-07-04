"""Subset search for Krea 2 projector vectors guided by directional progress.

For each of the 4095 non-empty subsets of the 12 projector slots:
  1. Analytically compute the minimum scalar s that achieves
       target_progress(s) >= --min-progress on every target prompt.
  2. Require direction cosine vs high-SKC >= --min-cosine.
  3. Score by control-prompt damage + off-axis residual + coefficient norm.

All metrics are in projector-output delta space (the (input @ vector) term
before it enters the transformer). This is a proxy for the x0/velocity space
— use score_krea_turbo_candidates.py to validate the top candidates with real
Turbo denoising passes.

Progress toward high-SKC (per prompt):
  u   = delta_proj(orig_vector) * high_strength    [desired direction]
  v_s = delta_proj(subset_vector) * s              [candidate at scalar s]

  progress(s) = dot(v_s, u) / dot(u, u)            [fraction of desired change]
  cosine       = cos(v_unit, u)                    [scale-invariant direction purity]
  residual(s)  = RMS(v_s - progress(s) * u)        [off-axis component]

Since progress is linear in s, the minimum feasible scalar is:
  s_min = min_progress / progress_unit             [progress_unit = progress at s=1]
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sweep_skc3vo_blank_vector import extract_projector_vector  # noqa: E402
from toolkit.basic import flush  # noqa: E402


DEFAULT_CAPTURE_DIR = Path(
    "loras/krea_vector_explore/txtfusion_probe/captures/multicapture_random_t100_float8"
)
DEFAULT_LORA = Path("loras/krea_vector_explore/skc3vo.safetensors")
DEFAULT_OUTPUT = Path(
    "loras/krea_vector_explore/txtfusion_probe/captures/krea_projector_direction_search.json"
)

# Prompts where we want high progress toward high-SKC behavior
DEFAULT_TARGET_PROMPTS = [
    "exposed_wide",
    "exposed_full_body",
    "body_focus_wide",
]

# Prompts where we want low drift (damage metric)
DEFAULT_CONTROL_PROMPTS = [
    "blank",
    "sfw_wide",
    "sfw_full_body",
    "sfw_fashion_wide",
    "fashion_wide",
    "covered_wide",
    "covered_full_body",
]


# ---------------------------------------------------------------------------
# Projector delta helpers
# ---------------------------------------------------------------------------


def proj_delta_flat(proj_input: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Compute projector output delta for a 12-element vector, flattened to 1D.

    proj_input: [..., 2560, 12]  (CPU float32)
    vector:     [12]
    returns:    1D float32 tensor of length prod(input.shape[:-1])
    """
    v = vector.view(*([1] * (proj_input.dim() - 1)), 12)
    delta = (proj_input * v).sum(dim=-1)  # [..., 2560]
    return delta.reshape(-1).contiguous()


def rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt())


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a * b).sum() / denom)


def subset_vector(base: torch.Tensor, mask: int) -> torch.Tensor:
    """Return base with slots zeroed according to mask bits (bit i → slot i, 0-indexed)."""
    v = base.clone()
    for i in range(12):
        if not (mask >> i) & 1:
            v[i] = 0.0
    return v


# ---------------------------------------------------------------------------
# Capture loading
# ---------------------------------------------------------------------------


def load_proj_inputs(capture_dir: Path, prompt_ids: List[str]) -> Dict[str, torch.Tensor]:
    """Load base_projector_input tensors from capture_*.pt files.

    Returns {prompt_id: float32 tensor}, skipping any missing captures.
    """
    result = {}
    for pid in prompt_ids:
        p = capture_dir / f"capture_{pid}.pt"
        if not p.exists():
            print(f"  WARNING: no capture for {pid!r}: {p}")
            continue
        data = torch.load(p, map_location="cpu", weights_only=False)
        proj_in = data.get("base_projector_input")
        if proj_in is None:
            print(f"  WARNING: no base_projector_input in {p}")
            continue
        result[pid] = proj_in.float()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subset search using directional progress toward high-SKC reference."
    )
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR,
                        help="Directory containing capture_*.pt files")
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA,
                        help="Path to SKC3VO .safetensors (original vector)")
    parser.add_argument("--high-strength", type=float, default=0.10,
                        help="Strength at which original SKC is known-working; defines the target direction u")
    parser.add_argument("--target-prompts", nargs="+", default=DEFAULT_TARGET_PROMPTS)
    parser.add_argument("--control-prompts", nargs="+", default=DEFAULT_CONTROL_PROMPTS)
    parser.add_argument("--min-progress", type=float, default=0.80,
                        help="Minimum progress on every target prompt (0.8 = 80%% of desired change)")
    parser.add_argument("--min-cosine", type=float, default=0.90,
                        help="Minimum cosine alignment with high-SKC direction")
    parser.add_argument("--scale-max", type=float, default=2.0,
                        help="Maximum allowed scalar s_min (rejects subsets that need too large a boost)")
    parser.add_argument("--damage-weight", type=float, default=1.0,
                        help="Weight of mean control-prompt damage in score")
    parser.add_argument("--residual-weight", type=float, default=0.25,
                        help="Weight of mean target off-axis residual in score")
    parser.add_argument("--norm-weight", type=float, default=0.05,
                        help="Weight of coefficient norm (relative to original) in score")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of top results to include in output JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    all_prompt_ids = list(dict.fromkeys(args.target_prompts + args.control_prompts))

    print(f"Loading captures from {args.capture_dir}")
    print(f"  Target prompts:  {args.target_prompts}")
    print(f"  Control prompts: {args.control_prompts}")
    flush()

    proj_inputs = load_proj_inputs(args.capture_dir, all_prompt_ids)
    target_ids = [p for p in args.target_prompts if p in proj_inputs]
    control_ids = [p for p in args.control_prompts if p in proj_inputs]
    missing = [p for p in all_prompt_ids if p not in proj_inputs]
    if missing:
        print(f"  Missing captures (skipped): {missing}")
    if not target_ids:
        raise RuntimeError("No target captures found — check --capture-dir and --target-prompts")
    print(f"  Loaded: {len(proj_inputs)} prompts ({len(target_ids)} target, {len(control_ids)} control)")
    flush()

    _, orig_vector = extract_projector_vector(load_file(str(args.lora)))
    orig_vector = orig_vector.float().cpu()
    orig_norm = float(orig_vector.norm())
    print(f"Original vector norm: {orig_norm:.3f}")
    flush()

    # Reference direction u per target prompt: proj_delta(input, orig_vec) * high_strength
    u_refs: Dict[str, torch.Tensor] = {
        pid: proj_delta_flat(proj_inputs[pid], orig_vector) * args.high_strength
        for pid in target_ids
    }
    u_norms_sq: Dict[str, float] = {
        pid: float((u_refs[pid] * u_refs[pid]).sum().clamp_min(1e-12))
        for pid in target_ids
    }

    # Pre-compute original reference metrics
    orig_blank_damage: Optional[float] = None
    if "blank" in proj_inputs:
        orig_blank_damage = rms(proj_delta_flat(proj_inputs["blank"], orig_vector) * args.high_strength)
    orig_control_damages = [
        rms(proj_delta_flat(proj_inputs[pid], orig_vector) * args.high_strength)
        for pid in control_ids
    ]
    orig_control_damage = sum(orig_control_damages) / len(orig_control_damages) if orig_control_damages else None

    print(f"Original @ high_strength={args.high_strength}:")
    print(f"  blank_damage={orig_blank_damage:.4f}" if orig_blank_damage is not None else "  blank_damage=N/A")
    print(f"  control_damage_mean={orig_control_damage:.4f}" if orig_control_damage is not None else "  control_damage_mean=N/A")
    flush()

    # -------------------------------------------------------------------
    # Enumerate all 4095 non-empty subsets
    # -------------------------------------------------------------------
    results = []
    n_wrong_dir = 0
    n_low_cosine = 0
    n_too_large_s = 0

    for mask in range(1, 1 << 12):
        unit_vec = subset_vector(orig_vector, mask)  # scale=1 (not s_min yet)

        # Per-target: compute progress_unit and cosine
        min_cosine = float("inf")
        min_progress_unit = float("inf")
        all_positive = True
        per_target_unit: Dict[str, dict] = {}

        for pid in target_ids:
            v_unit = proj_delta_flat(proj_inputs[pid], unit_vec)
            u = u_refs[pid]
            u_sq = u_norms_sq[pid]

            progress_unit = float((v_unit * u).sum()) / u_sq  # at s=1
            if progress_unit <= 0.0:
                all_positive = False
                break

            cos = cosine_sim(v_unit, u)
            per_target_unit[pid] = {
                "progress_unit": progress_unit,
                "cosine": cos,
                "v_unit": v_unit,
            }
            if cos < min_cosine:
                min_cosine = cos
            if progress_unit < min_progress_unit:
                min_progress_unit = progress_unit

        if not all_positive:
            n_wrong_dir += 1
            continue
        if min_cosine < args.min_cosine:
            n_low_cosine += 1
            continue

        # s_min: smallest scalar achieving min_progress on every target prompt
        s_min = max(
            args.min_progress / r["progress_unit"]
            for r in per_target_unit.values()
        )
        if s_min > args.scale_max:
            n_too_large_s += 1
            continue

        # Metrics at s_min
        target_progresses = []
        target_residuals = []
        for pid, r in per_target_unit.items():
            v_s = r["v_unit"] * s_min
            prog = r["progress_unit"] * s_min
            target_progresses.append(prog)
            res = rms(v_s - prog * u_refs[pid])
            target_residuals.append(res)
        mean_progress = sum(target_progresses) / len(target_progresses)
        mean_residual = sum(target_residuals) / len(target_residuals)

        # Control damage at s_min
        control_damages = [
            rms(proj_delta_flat(proj_inputs[pid], unit_vec) * s_min)
            for pid in control_ids
        ]
        mean_control_damage = sum(control_damages) / len(control_damages) if control_damages else 0.0

        # Blank damage separately
        blank_damage: Optional[float] = None
        if "blank" in proj_inputs:
            blank_damage = rms(proj_delta_flat(proj_inputs["blank"], unit_vec) * s_min)

        # Coefficient norm relative to original
        coeff_norm = float((unit_vec * s_min).norm())
        coeff_norm_rel = coeff_norm / orig_norm if orig_norm > 0 else coeff_norm

        score = (
            args.damage_weight * mean_control_damage
            + args.residual_weight * mean_residual
            + args.norm_weight * coeff_norm_rel
        )

        active_slots = [i + 1 for i in range(12) if (mask >> i) & 1]
        final_vec = (unit_vec * s_min).tolist()

        results.append(OrderedDict([
            ("mask", mask),
            ("slots", active_slots),
            ("scalar", round(float(s_min), 6)),
            ("vector", [round(x, 6) for x in final_vec]),
            ("score", round(float(score), 8)),
            ("target_progress_mean", round(float(mean_progress), 6)),
            ("target_cosine_min", round(float(min_cosine), 6)),
            ("target_residual_mean", round(float(mean_residual), 6)),
            ("control_damage_mean", round(float(mean_control_damage), 6)),
            ("blank_damage", round(float(blank_damage), 6) if blank_damage is not None else None),
            ("coeff_norm_relative", round(float(coeff_norm_rel), 6)),
            ("per_target", OrderedDict([
                (pid, OrderedDict([
                    ("progress", round(float(per_target_unit[pid]["progress_unit"] * s_min), 6)),
                    ("cosine", round(float(per_target_unit[pid]["cosine"]), 6)),
                ]))
                for pid in per_target_unit
            ])),
        ]))

    n_feasible = len(results)
    print(
        f"\nSearch complete: {n_feasible}/4095 feasible"
        f" ({n_wrong_dir} wrong-dir, {n_low_cosine} low-cosine, {n_too_large_s} s-too-large)"
    )
    flush()

    results.sort(key=lambda r: r["score"])
    top = results[: args.top_k]

    print(f"\nTop 5 by score:")
    for r in top[:5]:
        bd = f"{r['blank_damage']:.4f}" if r["blank_damage"] is not None else "N/A"
        print(
            f"  slots={r['slots']}"
            f"  s={r['scalar']:.4f}"
            f"  progress={r['target_progress_mean']:.3f}"
            f"  cosine={r['target_cosine_min']:.3f}"
            f"  ctrl={r['control_damage_mean']:.4f}"
            f"  blank={bd}"
            f"  score={r['score']:.5f}"
        )
    flush()

    report = OrderedDict([
        ("schema", "krea_projector_direction_search.v1"),
        ("capture_dir", str(args.capture_dir)),
        ("lora", str(args.lora)),
        ("high_strength", args.high_strength),
        ("min_progress", args.min_progress),
        ("min_cosine", args.min_cosine),
        ("scale_max", args.scale_max),
        ("target_prompts", args.target_prompts),
        ("control_prompts", args.control_prompts),
        ("weights", OrderedDict([
            ("damage", args.damage_weight),
            ("residual", args.residual_weight),
            ("norm", args.norm_weight),
        ])),
        ("original", OrderedDict([
            ("vector", orig_vector.tolist()),
            ("norm", round(orig_norm, 4)),
            ("high_strength", args.high_strength),
            ("blank_damage", round(orig_blank_damage, 6) if orig_blank_damage is not None else None),
            ("control_damage_mean", round(orig_control_damage, 6) if orig_control_damage is not None else None),
        ])),
        ("count_total", 4095),
        ("count_feasible", n_feasible),
        ("rejections", OrderedDict([
            ("wrong_direction", n_wrong_dir),
            ("low_cosine", n_low_cosine),
            ("scalar_too_large", n_too_large_s),
        ])),
        ("top_by_score", top),
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
