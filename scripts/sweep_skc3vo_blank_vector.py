import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file


DEFAULT_LORA = Path("loras/krea_vector_explore/skc3vo.safetensors")
DEFAULT_CACHE = Path("loras/krea_vector_explore/txtfusion_probe/cache/features.pt")
DEFAULT_OUTPUT = Path("loras/krea_vector_explore/txtfusion_probe/cache/skc3vo_blank_vector_sweep.json")


def extract_projector_vector(state_dict: dict[str, torch.Tensor]) -> tuple[str, torch.Tensor]:
    diff_keys = [
        "diffusion_model.txtfusion.projector.diff",
        "transformer.text_fusion.projector.diff",
        "txtfusion.projector.diff",
    ]
    for key in diff_keys:
        if key in state_dict:
            vector = state_dict[key].detach().float().reshape(-1)
            if vector.numel() != 12:
                raise ValueError(f"{key} must contain 12 projector values, got {tuple(state_dict[key].shape)}")
            return key, vector

    pairs = [
        (
            "transformer.text_fusion.projector.lora_A.weight",
            "transformer.text_fusion.projector.lora_B.weight",
        ),
        (
            "diffusion_model.txtfusion.projector.lora_A.weight",
            "diffusion_model.txtfusion.projector.lora_B.weight",
        ),
        (
            "txtfusion.projector.lora_A.weight",
            "txtfusion.projector.lora_B.weight",
        ),
    ]
    for a_key, b_key in pairs:
        if a_key in state_dict and b_key in state_dict:
            a = state_dict[a_key].detach().float()
            b = state_dict[b_key].detach().float()
            delta = torch.matmul(b, a).reshape(-1)
            if delta.numel() != 12:
                raise ValueError(
                    f"{b_key} @ {a_key} must produce 12 projector values, got {tuple(delta.shape)}"
                )
            return f"{b_key} @ {a_key}", delta

    raise ValueError("No txtfusion.projector diff or LoRA A/B pair found")


def projector_delta(features: torch.Tensor, vector: torch.Tensor, strength: float) -> torch.Tensor:
    f = features.float()
    v = vector.float().view(1, 12, 1)
    return (f * v).sum(dim=1) * float(strength)


def rms(tensor: torch.Tensor) -> float:
    t = tensor.float()
    return float(torch.sqrt(torch.mean(t * t)))


def mean_abs(tensor: torch.Tensor) -> float:
    return float(tensor.float().abs().mean())


def summarize_delta(features: torch.Tensor, vector: torch.Tensor, strength: float) -> OrderedDict:
    delta = projector_delta(features, vector, strength)
    return OrderedDict(
        [
            ("rms_delta", rms(delta)),
            ("mean_abs_delta", mean_abs(delta)),
            ("max_abs_delta", float(delta.float().abs().max())),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep each SKC3VO txtfusion projector slot by +/-N percent against cached blank prompt features."
    )
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--features", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-id", default="blank")
    parser.add_argument("--strength", type=float, default=0.01)
    parser.add_argument("--percent", type=float, default=10.0)
    args = parser.parse_args()

    state_dict = load_file(str(args.lora))
    source, base_vector = extract_projector_vector(state_dict)

    features_by_id = torch.load(args.features, map_location="cpu")
    if args.prompt_id not in features_by_id:
        known = ", ".join(sorted(features_by_id))
        raise ValueError(f"Prompt id {args.prompt_id!r} not found in {args.features}. Known ids: {known}")
    features = features_by_id[args.prompt_id]

    base_summary = summarize_delta(features, base_vector, args.strength)
    base_rms = base_summary["rms_delta"]
    pct = args.percent / 100.0

    candidates = []
    for slot in range(12):
        for direction, factor in (("minus", 1.0 - pct), ("plus", 1.0 + pct)):
            candidate = base_vector.clone()
            candidate[slot] = candidate[slot] * factor
            metrics = summarize_delta(features, candidate, args.strength)
            rms_delta = metrics["rms_delta"]
            candidates.append(
                OrderedDict(
                    [
                        ("slot", slot),
                        ("slot_1based", slot + 1),
                        ("direction", direction),
                        ("factor", factor),
                        ("old_value", float(base_vector[slot])),
                        ("new_value", float(candidate[slot])),
                        ("rms_delta", rms_delta),
                        ("rms_change", rms_delta - base_rms),
                        ("rms_change_percent", ((rms_delta / base_rms) - 1.0) * 100.0 if base_rms else 0.0),
                        ("mean_abs_delta", metrics["mean_abs_delta"]),
                        ("max_abs_delta", metrics["max_abs_delta"]),
                    ]
                )
            )

    ranked = sorted(candidates, key=lambda item: item["rms_delta"])
    report = OrderedDict(
        [
            ("schema", "skc3vo_blank_vector_sweep.v1"),
            ("lora", str(args.lora)),
            ("vector_source", source),
            ("features", str(args.features)),
            ("prompt_id", args.prompt_id),
            ("strength", args.strength),
            ("percent", args.percent),
            ("base_vector", [float(x) for x in base_vector.tolist()]),
            ("base", base_summary),
            ("best", ranked[:8]),
            ("worst", list(reversed(ranked[-8:]))),
            ("candidates", candidates),
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Vector source: {source}")
    print(f"Prompt: {args.prompt_id}")
    print(f"Strength: {args.strength:g}")
    print(f"Base blank RMS: {base_rms:.6f}")
    print(f"Wrote {args.output}")
    print("\nBest single-slot +/- changes:")
    for item in ranked[:8]:
        print(
            f"  V{item['slot_1based']:02d} {item['direction']:5s} "
            f"{item['old_value']:.6g}->{item['new_value']:.6g} "
            f"rms={item['rms_delta']:.6f} "
            f"change={item['rms_change']:.6f} ({item['rms_change_percent']:.3f}%)"
        )


if __name__ == "__main__":
    main()
