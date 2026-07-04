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

from replay_krea_txtfusion_projector import cosine, projector_delta  # noqa: E402
from sweep_skc3vo_blank_vector import rms  # noqa: E402


DEFAULT_MINIMIZE_CAPTURE_DIR = Path("loras/krea_vector_explore/txtfusion_probe/captures/multicapture_testpng_t050_float8")
DEFAULT_TARGET_CAPTURE_DIR = Path("loras/krea_vector_explore/txtfusion_probe/captures/multicapture_random_t100_float8")
DEFAULT_PROMPTS = Path("loras/krea_vector_explore/txtfusion_probe/prompts.json")
DEFAULT_OUTPUT = Path("loras/krea_vector_explore/txtfusion_probe/captures/krea_txtfusion_projector_subset_search.json")
DEFAULT_TARGET_PAIRS = [
    "covered_to_exposed_wide",
    "covered_to_exposed_full_body",
    "sfw_to_exposed_wide",
    "sfw_to_exposed_close",
    "sfw_to_exposed_full_body",
    "sfw_to_exposed_torso",
    "sfw_fashion_to_body_focus",
    "fashion_to_body_focus",
]
DEFAULT_DAMAGE_PAIRS = [
    "wide_to_close_covered",
    "wide_to_close_exposed",
    "full_body_to_torso_covered",
    "full_body_to_torso_exposed",
    "clean_to_bloody",
    "no_weapon_to_weapon",
]


def load_prompt_pairs(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in data.get("contrast_pairs", [])}


def load_captures(capture_dir: Path, capture_glob: str) -> dict[str, dict]:
    captures = {}
    for path in sorted(capture_dir.glob(capture_glob)):
        if path.name.endswith("_summary.pt"):
            continue
        capture = torch.load(path, map_location="cpu", weights_only=False)
        prompt_id = capture.get("prompt_id")
        if prompt_id:
            captures[prompt_id] = capture
    return captures


def vector_norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector.float()))


def pool_seq(delta: torch.Tensor) -> torch.Tensor:
    # delta: [seq, hidden, 1] — mean over token dim so shapes align across prompts
    return delta.mean(dim=0, keepdim=True)


def pair_delta(deltas: dict[str, torch.Tensor], pair: dict[str, str]) -> torch.Tensor | None:
    if pair["negative"] not in deltas or pair["positive"] not in deltas:
        return None
    return pool_seq(deltas[pair["positive"]]) - pool_seq(deltas[pair["negative"]])


def flatten_concat(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat([tensor.float().flatten() for tensor in tensors], dim=0)


def candidate_deltas(captures: dict[str, dict], vector: torch.Tensor, strength: float) -> dict[str, torch.Tensor]:
    return {
        prompt_id: projector_delta(capture["base_projector_input"].float(), vector, strength)
        for prompt_id, capture in captures.items()
    }


def contrast_vector(deltas: dict[str, torch.Tensor], pairs: dict[str, dict[str, str]], pair_ids: list[str]) -> torch.Tensor:
    parts = []
    for pair_id in pair_ids:
        pair = pairs.get(pair_id)
        if pair is None:
            continue
        delta = pair_delta(deltas, pair)
        if delta is not None:
            parts.append(delta)
    return flatten_concat(parts)


def covered_pairs(pairs: dict[str, dict[str, str]], pair_ids: list[str], capture_ids: set[str]) -> list[str]:
    return [
        pair_id
        for pair_id in pair_ids
        if pair_id in pairs and {pairs[pair_id]["negative"], pairs[pair_id]["positive"]} <= capture_ids
    ]


def subset_vector(base_vector: torch.Tensor, mask: int, scale: float) -> torch.Tensor:
    out = torch.zeros_like(base_vector)
    for idx in range(12):
        if mask & (1 << idx):
            out[idx] = base_vector[idx] * float(scale)
    return out


def subset_slots(mask: int) -> list[int]:
    return [idx + 1 for idx in range(12) if mask & (1 << idx)]


def prompt_delta_rms(deltas: dict[str, torch.Tensor], prompt_ids: list[str]) -> float:
    parts = [deltas[prompt_id] for prompt_id in prompt_ids if prompt_id in deltas]
    return rms(flatten_concat(parts))


def pair_metrics(
    deltas: dict[str, torch.Tensor],
    original_deltas: dict[str, torch.Tensor],
    pairs: dict[str, dict[str, str]],
    pair_ids: list[str],
) -> list[OrderedDict]:
    out = []
    for pair_id in pair_ids:
        pair = pairs[pair_id]
        delta = pair_delta(deltas, pair)
        original = pair_delta(original_deltas, pair)
        if delta is None or original is None:
            continue
        original_r = rms(original)
        out.append(
            OrderedDict(
                [
                    ("id", pair_id),
                    ("rms", rms(delta)),
                    ("ratio_to_original", rms(delta) / original_r if original_r else 0.0),
                    ("cosine_to_original", cosine(delta, original)),
                ]
            )
        )
    return out


def dominates(a: OrderedDict, b: OrderedDict) -> bool:
    a_values = (
        a["minimize_delta_rms"],
        a["damage_contrast_rms"],
        a["vector_norm"],
        -a["target_contrast_ratio_to_original"],
        -a["target_contrast_cosine_to_original"],
    )
    b_values = (
        b["minimize_delta_rms"],
        b["damage_contrast_rms"],
        b["vector_norm"],
        -b["target_contrast_ratio_to_original"],
        -b["target_contrast_cosine_to_original"],
    )
    return all(x <= y for x, y in zip(a_values, b_values)) and any(x < y for x, y in zip(a_values, b_values))


def pareto_frontier(records: list[OrderedDict], limit: int) -> list[OrderedDict]:
    frontier = []
    for record in records:
        if any(dominates(other, record) for other in records):
            continue
        frontier.append(record)
    frontier.sort(key=lambda row: (row["minimize_delta_rms"], row["damage_contrast_rms"], -row["target_contrast_ratio_to_original"]))
    return frontier[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate Krea txtfusion projector vector subsets across separate minimize and target captures.")
    parser.add_argument("--minimize-capture-dir", type=Path, default=DEFAULT_MINIMIZE_CAPTURE_DIR)
    parser.add_argument("--target-capture-dir", type=Path, default=DEFAULT_TARGET_CAPTURE_DIR)
    parser.add_argument("--capture-dir", type=Path, default=None, help="Legacy shorthand: use one directory for both minimize and target captures.")
    parser.add_argument("--capture-glob", default="capture_*.pt")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strength", type=float, default=None, help="Override capture lora_strength.")
    parser.add_argument("--blank-id", default="blank")
    parser.add_argument("--minimize-prompt-ids", nargs="*", default=None, help="Prompt deltas to minimize in the low-noise context. Defaults to all minimize captures.")
    parser.add_argument("--target-pairs", nargs="+", default=DEFAULT_TARGET_PAIRS)
    parser.add_argument("--damage-pairs", nargs="+", default=DEFAULT_DAMAGE_PAIRS)
    parser.add_argument("--min-target-ratio", type=float, default=0.80)
    parser.add_argument("--min-target-cosine", type=float, default=0.95)
    parser.add_argument("--scale-min", type=float, default=0.0)
    parser.add_argument("--scale-max", type=float, default=1.25)
    parser.add_argument("--scale-step", type=float, default=0.025)
    parser.add_argument("--w-minimize", type=float, default=1.0)
    parser.add_argument("--w-damage", type=float, default=1.0)
    parser.add_argument("--w-norm", type=float, default=0.0001)
    parser.add_argument("--w-change", type=float, default=0.0001)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--pareto-limit", type=int, default=80)
    args = parser.parse_args()

    if args.capture_dir is not None:
        args.minimize_capture_dir = args.capture_dir
        args.target_capture_dir = args.capture_dir

    minimize_captures = load_captures(args.minimize_capture_dir, args.capture_glob)
    target_captures = load_captures(args.target_capture_dir, args.capture_glob)
    if not minimize_captures:
        raise FileNotFoundError(f"No minimize captures matching {args.capture_glob!r} in {args.minimize_capture_dir}")
    if not target_captures:
        raise FileNotFoundError(f"No target captures matching {args.capture_glob!r} in {args.target_capture_dir}")
    if args.blank_id not in minimize_captures:
        raise ValueError(f"Blank minimize capture {args.blank_id!r} missing. Available: {sorted(minimize_captures)}")

    pairs = load_prompt_pairs(args.prompts)
    target_pairs = covered_pairs(pairs, args.target_pairs, set(target_captures))
    damage_pairs = covered_pairs(pairs, args.damage_pairs, set(minimize_captures))
    if not target_pairs:
        raise ValueError("No target pairs are fully covered by the target capture set")

    first_capture = next(iter(target_captures.values()))
    base_vector = first_capture.get("projector_vector")
    if base_vector is None:
        raise ValueError("Captures do not contain projector_vector")
    base_vector = base_vector.float()
    strength = float(args.strength if args.strength is not None else first_capture.get("lora_strength", 0.01))

    minimize_prompt_ids = args.minimize_prompt_ids or sorted(minimize_captures)
    minimize_prompt_ids = [prompt_id for prompt_id in minimize_prompt_ids if prompt_id in minimize_captures]

    original_minimize_deltas = candidate_deltas(minimize_captures, base_vector, strength)
    original_target_deltas = candidate_deltas(target_captures, base_vector, strength)
    original_minimize_rms = prompt_delta_rms(original_minimize_deltas, minimize_prompt_ids)
    original_blank_rms = rms(original_minimize_deltas[args.blank_id])
    original_target = contrast_vector(original_target_deltas, pairs, target_pairs)
    original_damage = contrast_vector(original_minimize_deltas, pairs, damage_pairs)
    original_target_rms = rms(original_target)
    original_damage_rms = rms(original_damage)

    records = []
    feasible = []
    for mask in range(1, 1 << 12):
        # Compute unit deltas at scale=1.0 (projector_delta is linear in the vector,
        # so delta(s) = unit_delta * s for any scalar s)
        unit_vector = subset_vector(base_vector, mask, 1.0)
        unit_min_deltas = candidate_deltas(minimize_captures, unit_vector, strength)
        unit_tgt_deltas = candidate_deltas(target_captures, unit_vector, strength)

        minimize_r_unit = prompt_delta_rms(unit_min_deltas, minimize_prompt_ids)
        blank_r_unit = rms(unit_min_deltas[args.blank_id])
        tgt_unit = contrast_vector(unit_tgt_deltas, pairs, target_pairs)
        dmg_unit = contrast_vector(unit_min_deltas, pairs, damage_pairs)
        tgt_r_unit = rms(tgt_unit)
        dmg_r_unit = rms(dmg_unit)

        # Cosine is scale-invariant: compute once
        tgt_cos = cosine(tgt_unit, original_target) if original_target_rms > 0.0 else 0.0

        # Minimum scale satisfying target_ratio >= min_target_ratio (linear in s)
        if tgt_r_unit > 0.0 and original_target_rms > 0.0:
            s_min = args.min_target_ratio * original_target_rms / tgt_r_unit
            optimal_s = float(max(args.scale_min, min(args.scale_max, s_min)))
        else:
            optimal_s = float(args.scale_max)

        # Scale unit metrics to optimal_s
        s = optimal_s
        minimize_r = minimize_r_unit * s
        blank_r = blank_r_unit * s
        tgt_r = tgt_r_unit * s
        dmg_r = dmg_r_unit * s
        tgt_ratio = tgt_r / original_target_rms if original_target_rms > 0.0 else 0.0

        vec_s = subset_vector(base_vector, mask, s)
        norm = vector_norm(vec_s)
        change_norm = vector_norm(vec_s - base_vector)
        score = (
            args.w_minimize * minimize_r
            + args.w_damage * dmg_r
            + args.w_norm * norm * norm
            + args.w_change * change_norm * change_norm
        )

        # Scale deltas for pair_metrics comparison
        scaled_min_deltas = {p: d * s for p, d in unit_min_deltas.items()}
        scaled_tgt_deltas = {p: d * s for p, d in unit_tgt_deltas.items()}

        is_feasible = (
            tgt_r_unit > 0.0
            and tgt_ratio >= args.min_target_ratio
            and tgt_cos >= args.min_target_cosine
            and args.scale_min <= s <= args.scale_max
        )

        record = OrderedDict(
            [
                ("id", f"subset_{mask:03x}_scale_{s:.4f}"),
                ("mask", mask),
                ("slots", subset_slots(mask)),
                ("scale", s),
                ("vector", [float(x) for x in vec_s.tolist()]),
                ("vector_norm", norm),
                ("change_norm_from_original", change_norm),
                ("score", float(score)),
                ("minimize_delta_rms", minimize_r),
                ("minimize_delta_ratio_to_original", minimize_r / original_minimize_rms if original_minimize_rms else 0.0),
                ("blank_delta_rms", blank_r),
                ("blank_delta_ratio_to_original", blank_r / original_blank_rms if original_blank_rms else 0.0),
                ("target_contrast_rms", tgt_r),
                ("target_contrast_ratio_to_original", tgt_ratio),
                ("target_contrast_cosine_to_original", tgt_cos),
                ("damage_contrast_rms", dmg_r),
                ("damage_contrast_ratio_to_original", dmg_r / original_damage_rms if original_damage_rms else 0.0),
                ("target_pairs", pair_metrics(scaled_tgt_deltas, original_target_deltas, pairs, target_pairs)),
                ("damage_pairs", pair_metrics(scaled_min_deltas, original_minimize_deltas, pairs, damage_pairs)),
            ]
        )
        records.append(record)
        if is_feasible:
            feasible.append(record)

    feasible_sorted = sorted(feasible, key=lambda row: row["score"])
    records_sorted = sorted(records, key=lambda row: row["score"])
    frontier = pareto_frontier(feasible_sorted, args.pareto_limit)

    report = OrderedDict(
        [
            ("schema", "krea_txtfusion_projector_subset_search.v3"),
            ("minimize_capture_dir", str(args.minimize_capture_dir)),
            ("target_capture_dir", str(args.target_capture_dir)),
            ("capture_glob", args.capture_glob),
            ("minimize_captures", sorted(minimize_captures)),
            ("target_captures", sorted(target_captures)),
            ("strength", strength),
            ("blank_id", args.blank_id),
            ("minimize_prompt_ids", minimize_prompt_ids),
            ("target_pairs", target_pairs),
            ("damage_pairs", damage_pairs),
            ("scale_min", args.scale_min),
            ("scale_max", args.scale_max),
            ("min_target_ratio", args.min_target_ratio),
            ("min_target_cosine", args.min_target_cosine),
            ("original", OrderedDict([
                ("vector", [float(x) for x in base_vector.tolist()]),
                ("vector_norm", vector_norm(base_vector)),
                ("minimize_delta_rms", original_minimize_rms),
                ("blank_delta_rms", original_blank_rms),
                ("target_contrast_rms", original_target_rms),
                ("damage_contrast_rms", original_damage_rms),
            ])),
            ("count_records", len(records)),
            ("count_feasible", len(feasible)),
            ("top_by_score", feasible_sorted[: args.top_k]),
            ("pareto_frontier", frontier),
            ("best_per_subset_by_score", records_sorted[: args.top_k]),
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Minimize captures: {', '.join(sorted(minimize_captures))}")
    print(f"Target captures: {', '.join(sorted(target_captures))}")
    print(f"Target pairs: {', '.join(target_pairs)}")
    print(f"Damage pairs: {', '.join(damage_pairs) if damage_pairs else '(none covered)'}")
    print(
        f"Original minimize_rms={original_minimize_rms:.6f} blank_rms={original_blank_rms:.6f} "
        f"target_rms={original_target_rms:.6f} damage_rms={original_damage_rms:.6f}"
    )
    print(f"Feasible candidates: {len(feasible)}")
    print("Top feasible candidates:")
    for row in feasible_sorted[:10]:
        print(
            f"  {row['id']:<26} slots={row['slots']} minimize={row['minimize_delta_rms']:.6f} "
            f"blank={row['blank_delta_rms']:.6f} target={row['target_contrast_ratio_to_original']:.3f}/"
            f"{row['target_contrast_cosine_to_original']:.3f} damage={row['damage_contrast_rms']:.6f} score={row['score']:.6f}"
        )


if __name__ == "__main__":
    main()
