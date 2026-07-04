import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

from replay_krea_txtfusion_projector import cosine
from sweep_skc3vo_blank_vector import rms, mean_abs


DEFAULT_OUTPUT = Path("loras/krea_vector_explore/txtfusion_probe/captures/krea_emotion_probe_analysis.json")
DEFAULT_PAIRS = [
    "emotion_neutral_to_distressed:emotion_face_neutral:emotion_face_distressed",
    "emotion_neutral_to_tears:emotion_face_neutral:emotion_face_tears",
    "emotion_distressed_to_crying:emotion_face_distressed:emotion_face_crying",
    "emotion_neutral_to_crying:emotion_face_neutral:emotion_face_crying",
    "emotion_tears_to_crying:emotion_face_tears:emotion_face_crying",
    "jinx_close_neutral_to_crying:jinx_close_neutral:jinx_close_crying",
]


def tensor_stats(tensor: torch.Tensor) -> OrderedDict:
    t = tensor.detach().float().cpu()
    return OrderedDict(
        [
            ("shape", list(t.shape)),
            ("mean", float(t.mean())),
            ("std", float(t.std(unbiased=False))),
            ("rms", rms(t)),
            ("mean_abs", mean_abs(t)),
            ("min", float(t.min())),
            ("max", float(t.max())),
        ]
    )


def spatial_stats(delta: torch.Tensor, top_k: int) -> OrderedDict:
    d = delta.detach().float()
    if d.dim() == 4:
        heat = torch.sqrt(torch.mean(d * d, dim=1))[0]
    elif d.dim() == 3:
        heat = torch.sqrt(torch.mean(d * d, dim=-1))
    else:
        return OrderedDict([("available", False), ("reason", f"unsupported shape {list(d.shape)}")])
    flat = heat.flatten()
    k = min(top_k, flat.numel())
    values, indices = torch.topk(flat, k=k)
    width = heat.shape[-1]
    coords = []
    for value, index in zip(values.tolist(), indices.tolist()):
        y = int(index // width)
        x = int(index % width)
        coords.append(OrderedDict([("y", y), ("x", x), ("value", float(value))]))
    return OrderedDict(
        [
            ("available", True),
            ("shape", list(heat.shape)),
            ("rms", rms(heat)),
            ("mean", float(heat.mean())),
            ("max", float(heat.max())),
            ("top", coords),
        ]
    )


def load_captures(capture_dir: Path) -> dict[str, dict]:
    captures = {}
    for path in sorted(capture_dir.glob("capture_*.pt")):
        capture = torch.load(path, map_location="cpu", weights_only=False)
        prompt_id = capture.get("prompt_id")
        if prompt_id:
            captures[prompt_id] = capture
    return captures


def parse_pair(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Pairs must be id:negative:positive, got {value!r}")
    return parts[0], parts[1], parts[2]


def delta(capture: dict, key: str) -> torch.Tensor | None:
    if key == "projector":
        if capture.get("lora_projector_output") is None:
            return None
        return capture["lora_projector_output"].float() - capture["base_projector_output"].float()
    if key == "velocity":
        if capture.get("lora_velocity") is None:
            return None
        return capture["lora_velocity"].float() - capture["base_velocity"].float()
    raise ValueError(key)


def clause_delta(captures: dict[str, dict], negative: str, positive: str, key: str, lora: bool) -> torch.Tensor | None:
    neg = captures[negative]
    pos = captures[positive]
    if key == "projector":
        neg_key = "lora_projector_output" if lora else "base_projector_output"
        pos_key = "lora_projector_output" if lora else "base_projector_output"
    elif key == "velocity":
        neg_key = "lora_velocity" if lora else "base_velocity"
        pos_key = "lora_velocity" if lora else "base_velocity"
    else:
        raise ValueError(key)
    if neg.get(neg_key) is None or pos.get(pos_key) is None:
        return None
    return pos[pos_key].float() - neg[neg_key].float()


def analyze_pair(captures: dict[str, dict], pair_id: str, negative: str, positive: str, top_k: int) -> OrderedDict:
    row = OrderedDict([("id", pair_id), ("negative", negative), ("positive", positive)])
    for key in ("projector", "velocity"):
        base_clause = clause_delta(captures, negative, positive, key, lora=False)
        skc_clause = clause_delta(captures, negative, positive, key, lora=True)
        neg_skc = delta(captures[negative], key)
        pos_skc = delta(captures[positive], key)
        if base_clause is None or skc_clause is None or neg_skc is None or pos_skc is None:
            row[key] = OrderedDict([("available", False)])
            continue
        skc_clause_change = skc_clause - base_clause
        skc_emotion_specific = pos_skc - neg_skc
        base_r = rms(base_clause)
        skc_r = rms(skc_clause)
        row[key] = OrderedDict(
            [
                ("available", True),
                ("base_clause_delta", tensor_stats(base_clause)),
                ("skc_clause_delta", tensor_stats(skc_clause)),
                ("skc_clause_change", tensor_stats(skc_clause_change)),
                ("skc_emotion_specific_delta", tensor_stats(skc_emotion_specific)),
                ("skc_clause_rms_ratio_to_base", skc_r / base_r if base_r else 0.0),
                ("skc_clause_cosine_to_base", cosine(skc_clause, base_clause)),
                ("skc_emotion_specific_cosine_to_base_clause", cosine(skc_emotion_specific, base_clause)),
                ("positive_skc_delta", tensor_stats(pos_skc)),
                ("negative_skc_delta", tensor_stats(neg_skc)),
                ("positive_skc_delta_cosine_to_base_clause", cosine(pos_skc, base_clause)),
                ("negative_skc_delta_cosine_to_base_clause", cosine(neg_skc, base_clause)),
                ("base_clause_spatial", spatial_stats(base_clause, top_k) if key == "velocity" else None),
                ("skc_clause_change_spatial", spatial_stats(skc_clause_change, top_k) if key == "velocity" else None),
                ("skc_emotion_specific_spatial", spatial_stats(skc_emotion_specific, top_k) if key == "velocity" else None),
            ]
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze neutral-vs-emotion Krea captures and SKC-specific emotion effects.")
    parser.add_argument("--capture-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair", dest="pairs", action="append", default=None, help="id:negative:positive. Can be repeated.")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    pair_specs = [parse_pair(item) for item in (args.pairs or DEFAULT_PAIRS)]
    reports = []
    for capture_dir in args.capture_dir:
        captures = load_captures(capture_dir)
        pair_reports = []
        for pair_id, negative, positive in pair_specs:
            if negative not in captures or positive not in captures:
                continue
            pair_reports.append(analyze_pair(captures, pair_id, negative, positive, args.top_k))
        reports.append(
            OrderedDict(
                [
                    ("capture_dir", str(capture_dir)),
                    ("captures", sorted(captures)),
                    ("pairs", pair_reports),
                ]
            )
        )

    report = OrderedDict([("schema", "krea_emotion_probe_analysis.v1"), ("reports", reports)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    for capture_report in reports:
        print(f"Capture dir: {capture_report['capture_dir']}")
        for pair in capture_report["pairs"]:
            velocity = pair["velocity"]
            if not velocity.get("available"):
                continue
            print(
                f"  {pair['id']:<34} base={velocity['base_clause_delta']['rms']:.6f} "
                f"skc/base={velocity['skc_clause_rms_ratio_to_base']:.3f} "
                f"skc_cos={velocity['skc_clause_cosine_to_base']:.3f} "
                f"skc_specific_cos={velocity['skc_emotion_specific_cosine_to_base_clause']:.3f}"
            )


if __name__ == "__main__":
    main()
