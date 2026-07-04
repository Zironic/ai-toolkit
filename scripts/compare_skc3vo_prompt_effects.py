import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file

from sweep_skc3vo_blank_vector import DEFAULT_CACHE, DEFAULT_LORA, extract_projector_vector, summarize_delta


DEFAULT_SUMMARY = Path("loras/krea_vector_explore/txtfusion_probe/cache/summary.json")
DEFAULT_OPTIMIZED = [
    Path("loras/krea_vector_explore/txtfusion_probe/cache/skc3vo_blank_vector_optimize.json"),
    Path("loras/krea_vector_explore/txtfusion_probe/cache/skc3vo_blank_vector_optimize_200.json"),
]
DEFAULT_OUTPUT = Path("loras/krea_vector_explore/txtfusion_probe/cache/skc3vo_prompt_effect_comparison.json")


def load_vector_from_optimize_report(path: Path) -> tuple[str, torch.Tensor]:
    data = json.loads(path.read_text(encoding="utf-8"))
    label = path.stem.replace("skc3vo_blank_vector_", "")
    return label, torch.tensor(data["final_vector"], dtype=torch.float32)


def ratio(value: float, base: float) -> float:
    if base == 0.0:
        return 0.0
    return value / base


def prompt_rows(features_by_id: dict[str, torch.Tensor], vector: torch.Tensor, strength: float) -> list[OrderedDict]:
    blank_rms = summarize_delta(features_by_id["blank"], vector, strength)["rms_delta"]
    rows = []
    for prompt_id, features in features_by_id.items():
        metrics = summarize_delta(features, vector, strength)
        rms = metrics["rms_delta"]
        rows.append(
            OrderedDict(
                [
                    ("prompt_id", prompt_id),
                    ("rms_delta", rms),
                    ("blank_ratio", ratio(rms, blank_rms)),
                    ("mean_abs_delta", metrics["mean_abs_delta"]),
                    ("max_abs_delta", metrics["max_abs_delta"]),
                ]
            )
        )
    return rows


def pair_rows(rows_by_prompt: dict[str, OrderedDict], contrast_pairs: list[dict]) -> list[OrderedDict]:
    out = []
    for pair in contrast_pairs:
        negative = rows_by_prompt[pair["negative"]]
        positive = rows_by_prompt[pair["positive"]]
        neg_rms = negative["rms_delta"]
        pos_rms = positive["rms_delta"]
        out.append(
            OrderedDict(
                [
                    ("id", pair["id"]),
                    ("label", pair.get("label", f"{pair['negative']} -> {pair['positive']}")),
                    ("negative", pair["negative"]),
                    ("positive", pair["positive"]),
                    ("negative_rms", neg_rms),
                    ("positive_rms", pos_rms),
                    ("rms_difference", pos_rms - neg_rms),
                    ("relative_difference_percent", ((pos_rms / neg_rms) - 1.0) * 100.0 if neg_rms else 0.0),
                    ("negative_blank_ratio", negative["blank_ratio"]),
                    ("positive_blank_ratio", positive["blank_ratio"]),
                ]
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SKC3VO vector variants across cached Krea txtfusion probe prompts.")
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--features", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--optimized", type=Path, nargs="*", default=DEFAULT_OPTIMIZED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strength", type=float, default=0.01)
    args = parser.parse_args()

    features_by_id = torch.load(args.features, map_location="cpu")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    state_dict = load_file(str(args.lora))
    source, original_vector = extract_projector_vector(state_dict)

    vectors = [("original", original_vector)]
    for path in args.optimized:
        if path.exists():
            vectors.append(load_vector_from_optimize_report(path))

    reports = []
    for label, vector in vectors:
        rows = prompt_rows(features_by_id, vector, args.strength)
        rows_by_prompt = {row["prompt_id"]: row for row in rows}
        pairs = pair_rows(rows_by_prompt, summary.get("contrast_pairs", []))
        reports.append(
            OrderedDict(
                [
                    ("id", label),
                    ("vector", [float(x) for x in vector.tolist()]),
                    ("vector_norm", float(torch.linalg.vector_norm(vector.float()))),
                    ("blank_rms_delta", rows_by_prompt["blank"]["rms_delta"]),
                    ("prompts", rows),
                    ("contrast_pairs", pairs),
                    ("top_prompt_blank_ratios", sorted(rows, key=lambda row: row["blank_ratio"], reverse=True)[:8]),
                    ("bottom_prompt_blank_ratios", sorted(rows, key=lambda row: row["blank_ratio"])[:8]),
                    ("top_pair_relative_differences", sorted(pairs, key=lambda row: abs(row["relative_difference_percent"]), reverse=True)[:8]),
                ]
            )
        )

    report = OrderedDict(
        [
            ("schema", "skc3vo_prompt_effect_comparison.v1"),
            ("lora", str(args.lora)),
            ("vector_source", source),
            ("features", str(args.features)),
            ("summary", str(args.summary)),
            ("strength", args.strength),
            ("vectors", reports),
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    for item in reports:
        print(f"\n{item['id']} norm={item['vector_norm']:.6f} blank={item['blank_rms_delta']:.6f}")
        print("  Top prompt / blank ratios:")
        for row in item["top_prompt_blank_ratios"][:6]:
            print(f"    {row['prompt_id']:<24} rms={row['rms_delta']:.6f} ratio={row['blank_ratio']:.4f}")
        print("  Contrast relative differences:")
        for row in item["top_pair_relative_differences"][:6]:
            print(
                f"    {row['id']:<28} {row['relative_difference_percent']:+.3f}% "
                f"({row['negative_rms']:.6f}->{row['positive_rms']:.6f})"
            )


if __name__ == "__main__":
    main()
