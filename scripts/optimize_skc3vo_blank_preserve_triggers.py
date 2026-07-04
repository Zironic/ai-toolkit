import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file

from sweep_skc3vo_blank_vector import DEFAULT_CACHE, DEFAULT_LORA, extract_projector_vector, summarize_delta


DEFAULT_OUTPUT = Path("loras/krea_vector_explore/txtfusion_probe/cache/skc3vo_blank_preserve_triggers_optimize.json")
DEFAULT_TRIGGER_PROMPTS = [
    "exposed_wide",
    "exposed_close",
    "exposed_full_body",
    "exposed_torso",
    "body_focus_wide",
]
DEFAULT_TRACK_PROMPTS = [
    "blank",
    "sfw_wide",
    "covered_wide",
    "exposed_wide",
    "sfw_close",
    "exposed_close",
    "sfw_full_body",
    "exposed_full_body",
    "sfw_torso",
    "exposed_torso",
    "body_focus_wide",
]


def rms_for(features_by_id: dict[str, torch.Tensor], vector: torch.Tensor, strength: float, prompt_id: str) -> float:
    return summarize_delta(features_by_id[prompt_id], vector, strength)["rms_delta"]


def prompt_metrics(features_by_id: dict[str, torch.Tensor], vector: torch.Tensor, strength: float, prompt_ids: list[str]) -> OrderedDict:
    out = OrderedDict()
    for prompt_id in prompt_ids:
        out[prompt_id] = summarize_delta(features_by_id[prompt_id], vector, strength)
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def candidate_record(
    features_by_id: dict[str, torch.Tensor],
    current_vector: torch.Tensor,
    slot: int,
    direction: str,
    factor: float,
    strength: float,
    blank_prompt: str,
    trigger_prompts: list[str],
    current_blank_rms: float,
    current_trigger_rms: dict[str, float],
) -> OrderedDict:
    candidate = current_vector.clone()
    candidate[slot] = candidate[slot] * factor

    candidate_blank_rms = rms_for(features_by_id, candidate, strength, blank_prompt)
    candidate_trigger_rms = {
        prompt_id: rms_for(features_by_id, candidate, strength, prompt_id)
        for prompt_id in trigger_prompts
    }

    blank_reduction = 1.0 - (candidate_blank_rms / current_blank_rms) if current_blank_rms else 0.0
    trigger_reductions = [
        1.0 - (candidate_trigger_rms[prompt_id] / current_trigger_rms[prompt_id])
        if current_trigger_rms[prompt_id]
        else 0.0
        for prompt_id in trigger_prompts
    ]
    avg_trigger_reduction = mean(trigger_reductions)
    max_trigger_reduction = max(trigger_reductions) if trigger_reductions else 0.0
    preservation_gap = blank_reduction - avg_trigger_reduction

    return OrderedDict(
        [
            ("slot", slot),
            ("slot_1based", slot + 1),
            ("direction", direction),
            ("factor", factor),
            ("old_value", float(current_vector[slot])),
            ("new_value", float(candidate[slot])),
            ("blank_rms", candidate_blank_rms),
            ("blank_reduction", blank_reduction),
            ("blank_reduction_percent", blank_reduction * 100.0),
            ("trigger_rms", candidate_trigger_rms),
            ("trigger_reductions", dict(zip(trigger_prompts, trigger_reductions))),
            ("avg_trigger_reduction", avg_trigger_reduction),
            ("avg_trigger_reduction_percent", avg_trigger_reduction * 100.0),
            ("max_trigger_reduction", max_trigger_reduction),
            ("max_trigger_reduction_percent", max_trigger_reduction * 100.0),
            ("preservation_gap", preservation_gap),
            ("preservation_gap_percent", preservation_gap * 100.0),
            ("vector", [float(x) for x in candidate.tolist()]),
        ]
    )


def validate_prompt_ids(features_by_id: dict[str, torch.Tensor], prompt_ids: list[str], label: str) -> list[str]:
    out = []
    known = ", ".join(sorted(features_by_id))
    for prompt_id in prompt_ids:
        if prompt_id not in features_by_id:
            raise ValueError(f"{label} prompt id {prompt_id!r} not found. Known ids: {known}")
        if prompt_id not in out:
            out.append(prompt_id)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedily reduce SKC3VO blank RMS only when blank falls more than trigger prompts."
    )
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--features", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blank-prompt", default="blank")
    parser.add_argument("--trigger-prompts", nargs="*", default=DEFAULT_TRIGGER_PROMPTS)
    parser.add_argument("--track-prompts", nargs="*", default=DEFAULT_TRACK_PROMPTS)
    parser.add_argument("--strength", type=float, default=0.01)
    parser.add_argument("--percent", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--min-blank-improvement", type=float, default=1e-6)
    parser.add_argument(
        "--require-beats-max-trigger",
        action="store_true",
        help="Require blank reduction to beat every trigger prompt reduction, not just the average.",
    )
    args = parser.parse_args()

    features_by_id = torch.load(args.features, map_location="cpu")
    blank_prompt = validate_prompt_ids(features_by_id, [args.blank_prompt], "blank")[0]
    trigger_prompts = validate_prompt_ids(features_by_id, args.trigger_prompts, "trigger")
    track_prompts = validate_prompt_ids(features_by_id, args.track_prompts, "track")
    if blank_prompt not in track_prompts:
        track_prompts.insert(0, blank_prompt)

    state_dict = load_file(str(args.lora))
    source, initial_vector = extract_projector_vector(state_dict)
    current_vector = initial_vector.clone()
    factor_down = 1.0 - (args.percent / 100.0)
    factor_up = 1.0 + (args.percent / 100.0)

    initial_blank_rms = rms_for(features_by_id, initial_vector, args.strength, blank_prompt)
    initial_trigger_rms = {
        prompt_id: rms_for(features_by_id, initial_vector, args.strength, prompt_id)
        for prompt_id in trigger_prompts
    }
    current_blank_rms = initial_blank_rms
    current_trigger_rms = dict(initial_trigger_rms)
    iterations = []

    print(f"Vector source: {source}")
    print(f"Blank prompt: {blank_prompt}")
    print(f"Trigger prompts: {', '.join(trigger_prompts)}")
    print(f"Strength: {args.strength:g}")
    print(f"Initial blank RMS: {initial_blank_rms:.6f}")
    print(f"Initial trigger RMS avg: {mean(list(initial_trigger_rms.values())):.6f}")

    for iteration in range(1, args.iterations + 1):
        candidates = []
        for slot in range(12):
            candidates.append(
                candidate_record(
                    features_by_id,
                    current_vector,
                    slot,
                    "minus",
                    factor_down,
                    args.strength,
                    blank_prompt,
                    trigger_prompts,
                    current_blank_rms,
                    current_trigger_rms,
                )
            )
            candidates.append(
                candidate_record(
                    features_by_id,
                    current_vector,
                    slot,
                    "plus",
                    factor_up,
                    args.strength,
                    blank_prompt,
                    trigger_prompts,
                    current_blank_rms,
                    current_trigger_rms,
                )
            )

        eligible = []
        for candidate in candidates:
            blank_improvement = current_blank_rms - candidate["blank_rms"]
            beats_trigger = candidate["blank_reduction"] > candidate["avg_trigger_reduction"]
            if args.require_beats_max_trigger:
                beats_trigger = candidate["blank_reduction"] > candidate["max_trigger_reduction"]
            if blank_improvement > args.min_blank_improvement and beats_trigger:
                eligible.append(candidate)

        ranked_all = sorted(
            candidates,
            key=lambda item: (item["blank_rms"], -item["preservation_gap"]),
        )
        ranked_eligible = sorted(
            eligible,
            key=lambda item: (item["blank_rms"], -item["preservation_gap"]),
        )
        accepted = bool(ranked_eligible)
        step = OrderedDict(
            [
                ("iteration", iteration),
                ("start_blank_rms", current_blank_rms),
                ("start_trigger_rms", current_trigger_rms),
                ("accepted", accepted),
                ("best_eligible", ranked_eligible[0] if ranked_eligible else None),
                ("top_eligible", ranked_eligible[:8]),
                ("top_all", ranked_all[:8]),
            ]
        )
        iterations.append(step)

        if not accepted:
            print(f"Iter {iteration:03d}: stop, no candidate reduced blank more than trigger prompts")
            break

        best = ranked_eligible[0]
        current_vector = torch.tensor(best["vector"], dtype=torch.float32)
        current_blank_rms = best["blank_rms"]
        current_trigger_rms = dict(best["trigger_rms"])
        print(
            f"Iter {iteration:03d}: V{best['slot_1based']:02d} {best['direction']:5s} "
            f"blank={current_blank_rms:.6f} "
            f"blank_red={best['blank_reduction_percent']:+.3f}% "
            f"trigger_red={best['avg_trigger_reduction_percent']:+.3f}% "
            f"gap={best['preservation_gap_percent']:+.3f}%"
        )

    final_trigger_avg = mean(list(current_trigger_rms.values()))
    report = OrderedDict(
        [
            ("schema", "skc3vo_blank_preserve_triggers_optimize.v1"),
            ("lora", str(args.lora)),
            ("vector_source", source),
            ("features", str(args.features)),
            ("blank_prompt", blank_prompt),
            ("trigger_prompts", trigger_prompts),
            ("track_prompts", track_prompts),
            ("strength", args.strength),
            ("percent", args.percent),
            ("iterations_requested", args.iterations),
            ("min_blank_improvement", args.min_blank_improvement),
            ("require_beats_max_trigger", args.require_beats_max_trigger),
            ("initial_vector", [float(x) for x in initial_vector.tolist()]),
            ("initial_blank_rms", initial_blank_rms),
            ("initial_trigger_rms", initial_trigger_rms),
            ("initial_trigger_avg_rms", mean(list(initial_trigger_rms.values()))),
            ("final_vector", [float(x) for x in current_vector.tolist()]),
            ("final_blank_rms", current_blank_rms),
            ("final_trigger_rms", current_trigger_rms),
            ("final_trigger_avg_rms", final_trigger_avg),
            ("final_blank_reduction_percent", (1.0 - current_blank_rms / initial_blank_rms) * 100.0 if initial_blank_rms else 0.0),
            ("final_trigger_avg_reduction_percent", (1.0 - final_trigger_avg / mean(list(initial_trigger_rms.values()))) * 100.0 if initial_trigger_rms else 0.0),
            ("final_metrics", prompt_metrics(features_by_id, current_vector, args.strength, track_prompts)),
            ("iterations", iterations),
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Final blank RMS: {current_blank_rms:.6f}")
    print(f"Final trigger RMS avg: {final_trigger_avg:.6f}")
    print(f"Final blank reduction: {report['final_blank_reduction_percent']:.3f}%")
    print(f"Final trigger avg reduction: {report['final_trigger_avg_reduction_percent']:.3f}%")
    print(f"Final vector: {[round(float(x), 6) for x in current_vector.tolist()]}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
