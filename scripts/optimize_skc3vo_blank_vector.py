import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file

from sweep_skc3vo_blank_vector import (
    DEFAULT_CACHE,
    DEFAULT_LORA,
    extract_projector_vector,
    summarize_delta,
)


DEFAULT_OUTPUT = Path("loras/krea_vector_explore/txtfusion_probe/cache/skc3vo_blank_vector_optimize.json")


def score(features: torch.Tensor, vector: torch.Tensor, strength: float) -> float:
    return summarize_delta(features, vector, strength)["rms_delta"]


def vector_metrics(features_by_id: dict[str, torch.Tensor], vector: torch.Tensor, strength: float, prompt_ids: list[str]) -> OrderedDict:
    out = OrderedDict()
    for prompt_id in prompt_ids:
        out[prompt_id] = summarize_delta(features_by_id[prompt_id], vector, strength)
    return out


def candidate_record(
    features: torch.Tensor,
    current_vector: torch.Tensor,
    base_score: float,
    slot: int,
    direction: str,
    factor: float,
    strength: float,
) -> OrderedDict:
    candidate = current_vector.clone()
    candidate[slot] = candidate[slot] * factor
    metrics = summarize_delta(features, candidate, strength)
    candidate_score = metrics["rms_delta"]
    return OrderedDict(
        [
            ("slot", slot),
            ("slot_1based", slot + 1),
            ("direction", direction),
            ("factor", factor),
            ("old_value", float(current_vector[slot])),
            ("new_value", float(candidate[slot])),
            ("score", candidate_score),
            ("score_change", candidate_score - base_score),
            ("score_change_percent", ((candidate_score / base_score) - 1.0) * 100.0 if base_score else 0.0),
            ("metrics", metrics),
            ("vector", [float(x) for x in candidate.tolist()]),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedily optimize SKC3VO txtfusion projector slot scales against cached blank prompt RMS."
    )
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--features", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-id", default="blank")
    parser.add_argument("--track-prompts", nargs="*", default=["blank", "sfw_wide", "exposed_wide", "sfw_close", "exposed_close"])
    parser.add_argument("--strength", type=float, default=0.01)
    parser.add_argument("--percent", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--min-improvement", type=float, default=1e-6)
    args = parser.parse_args()

    state_dict = load_file(str(args.lora))
    source, base_vector = extract_projector_vector(state_dict)

    features_by_id = torch.load(args.features, map_location="cpu")
    if args.prompt_id not in features_by_id:
        known = ", ".join(sorted(features_by_id))
        raise ValueError(f"Prompt id {args.prompt_id!r} not found in {args.features}. Known ids: {known}")

    track_prompts = []
    for prompt_id in args.track_prompts:
        if prompt_id not in features_by_id:
            known = ", ".join(sorted(features_by_id))
            raise ValueError(f"Tracked prompt id {prompt_id!r} not found in {args.features}. Known ids: {known}")
        if prompt_id not in track_prompts:
            track_prompts.append(prompt_id)
    if args.prompt_id not in track_prompts:
        track_prompts.insert(0, args.prompt_id)

    optimize_features = features_by_id[args.prompt_id]
    current_vector = base_vector.clone()
    factor_down = 1.0 - (args.percent / 100.0)
    factor_up = 1.0 + (args.percent / 100.0)

    initial_score = score(optimize_features, current_vector, args.strength)
    current_score = initial_score
    iterations = []

    print(f"Vector source: {source}")
    print(f"Prompt: {args.prompt_id}")
    print(f"Strength: {args.strength:g}")
    print(f"Initial blank RMS: {initial_score:.6f}")

    for iteration in range(1, args.iterations + 1):
        candidates = []
        for slot in range(12):
            candidates.append(
                candidate_record(
                    optimize_features,
                    current_vector,
                    current_score,
                    slot,
                    "minus",
                    factor_down,
                    args.strength,
                )
            )
            candidates.append(
                candidate_record(
                    optimize_features,
                    current_vector,
                    current_score,
                    slot,
                    "plus",
                    factor_up,
                    args.strength,
                )
            )

        ranked = sorted(candidates, key=lambda item: item["score"])
        best = ranked[0]
        improvement = current_score - best["score"]
        accepted = improvement > args.min_improvement
        step = OrderedDict(
            [
                ("iteration", iteration),
                ("start_score", current_score),
                ("best_candidate", best),
                ("accepted", accepted),
                ("improvement", improvement),
                ("top_candidates", ranked[:8]),
            ]
        )
        iterations.append(step)

        if not accepted:
            print(f"Iter {iteration:02d}: stop, best improvement {improvement:.9f} <= {args.min_improvement:g}")
            break

        current_vector = torch.tensor(best["vector"], dtype=torch.float32)
        current_score = best["score"]
        print(
            f"Iter {iteration:02d}: V{best['slot_1based']:02d} {best['direction']:5s} "
            f"score={current_score:.6f} improvement={improvement:.6f} "
            f"total={(current_score / initial_score - 1.0) * 100.0:.3f}%"
        )

    final_metrics = vector_metrics(features_by_id, current_vector, args.strength, track_prompts)
    report = OrderedDict(
        [
            ("schema", "skc3vo_blank_vector_optimize.v1"),
            ("lora", str(args.lora)),
            ("vector_source", source),
            ("features", str(args.features)),
            ("prompt_id", args.prompt_id),
            ("track_prompts", track_prompts),
            ("strength", args.strength),
            ("percent", args.percent),
            ("iterations_requested", args.iterations),
            ("min_improvement", args.min_improvement),
            ("initial_vector", [float(x) for x in base_vector.tolist()]),
            ("initial_score", initial_score),
            ("final_vector", [float(x) for x in current_vector.tolist()]),
            ("final_score", current_score),
            ("final_score_change", current_score - initial_score),
            ("final_score_change_percent", ((current_score / initial_score) - 1.0) * 100.0 if initial_score else 0.0),
            ("final_metrics", final_metrics),
            ("iterations", iterations),
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Final blank RMS: {current_score:.6f}")
    print(f"Final change: {current_score - initial_score:.6f} ({((current_score / initial_score) - 1.0) * 100.0:.3f}%)")
    print(f"Final vector: {[round(float(x), 6) for x in current_vector.tolist()]}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
