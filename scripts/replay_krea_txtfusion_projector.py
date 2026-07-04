import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file

from sweep_skc3vo_blank_vector import extract_projector_vector, rms, mean_abs


DEFAULT_CAPTURE = Path("loras/krea_vector_explore/txtfusion_probe/captures/krea_txtfusion_forward_capture_testpng_float8.pt")
DEFAULT_OUTPUT = Path("loras/krea_vector_explore/txtfusion_probe/captures/krea_txtfusion_projector_replay.json")
DEFAULT_LORA_DIR = Path("loras/krea_vector_explore")
DEFAULT_OPTIMIZED = [
    Path("loras/krea_vector_explore/txtfusion_probe/cache/skc3vo_blank_vector_optimize.json"),
    Path("loras/krea_vector_explore/txtfusion_probe/cache/skc3vo_blank_vector_optimize_200.json"),
    Path("loras/krea_vector_explore/txtfusion_probe/cache/skc3vo_blank_preserve_triggers_optimize.json"),
    Path("loras/krea_vector_explore/txtfusion_probe/cache/skc3vo_blank_preserve_triggers_optimize_strict.json"),
]


def tensor_stats(tensor: torch.Tensor) -> OrderedDict:
    t = tensor.detach().float().cpu()
    return OrderedDict(
        [
            ("shape", list(t.shape)),
            ("dtype", str(tensor.dtype).replace("torch.", "")),
            ("mean", float(t.mean())),
            ("std", float(t.std(unbiased=False))),
            ("rms", rms(t)),
            ("mean_abs", mean_abs(t)),
            ("min", float(t.min())),
            ("max", float(t.max())),
        ]
    )


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().float().flatten()
    bv = b.detach().float().flatten()
    denom = torch.linalg.vector_norm(av) * torch.linalg.vector_norm(bv)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(av, bv) / denom)


def projector_delta(projector_input: torch.Tensor, vector: torch.Tensor, strength: float, *, exact_dtype: bool = False) -> torch.Tensor:
    if exact_dtype:
        x = projector_input
        v = vector.to(device=x.device, dtype=x.dtype).view(1, 1, 12)
        return (x * v).sum(dim=-1, keepdim=True) * float(strength)
    x = projector_input.float()
    v = vector.float().view(1, 1, 12)
    return (x * v).sum(dim=-1, keepdim=True) * float(strength)


def replay_output(
    base_output: torch.Tensor,
    projector_input: torch.Tensor,
    vector: torch.Tensor | None,
    strength: float,
    *,
    exact_dtype: bool = False,
) -> torch.Tensor:
    if vector is None or strength == 0.0:
        return base_output if exact_dtype else base_output.float()
    if exact_dtype:
        return base_output + projector_delta(projector_input, vector, strength, exact_dtype=True)
    return base_output.float() + projector_delta(projector_input, vector, strength)


def compare_tensors(actual: torch.Tensor, replayed: torch.Tensor) -> OrderedDict:
    diff = replayed.float() - actual.float()
    return OrderedDict(
        [
            ("actual", tensor_stats(actual)),
            ("replayed", tensor_stats(replayed)),
            ("diff", tensor_stats(diff)),
            ("cosine", cosine(actual, replayed)),
            ("max_abs_error", float(diff.abs().max())),
            ("rms_error", rms(diff)),
        ]
    )


def load_optimized_vectors(paths: list[Path]) -> list[tuple[str, torch.Tensor, str]]:
    out = []
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        label = path.stem.replace("skc3vo_", "")
        out.append((label, torch.tensor(data["final_vector"], dtype=torch.float32), str(path)))
    return out


def vector_recipes(base_vector: torch.Tensor) -> list[tuple[str, torch.Tensor, str]]:
    recipes = []
    recipes.append(("skc3vo_original", base_vector.clone(), "capture projector_vector"))

    for idx in range(12):
        v = torch.zeros_like(base_vector)
        v[idx] = base_vector[idx]
        recipes.append((f"v{idx + 1:02d}_only", v, "single original slot"))

    for name, slots in [
        ("v08_v09_v10_v11_only", [7, 8, 9, 10]),
        ("v09_v10_v11_only", [8, 9, 10]),
        ("v10_v11_only", [9, 10]),
        ("without_v08", [i for i in range(12) if i != 7]),
        ("without_v09", [i for i in range(12) if i != 8]),
        ("without_v10", [i for i in range(12) if i != 9]),
        ("without_v11", [i for i in range(12) if i != 10]),
        ("without_v08_v09_v10_v11", [i for i in range(12) if i not in (7, 8, 9, 10)]),
    ]:
        v = torch.zeros_like(base_vector)
        v[slots] = base_vector[slots]
        recipes.append((name, v, "slot mask recipe"))

    return recipes


def load_lora_vectors(lora_dir: Path) -> list[tuple[str, torch.Tensor, str]]:
    out = []
    for path in sorted(lora_dir.glob("*.safetensors")):
        source, vector = extract_projector_vector(load_file(str(path)))
        out.append((path.stem, vector.float(), source))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Krea txtfusion.projector deltas against captured real pre-projector tensors.")
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--optimized", type=Path, nargs="*", default=DEFAULT_OPTIMIZED)
    parser.add_argument("--strength", type=float, default=None, help="Override capture lora_strength.")
    parser.add_argument(
        "--normalize",
        choices=["none", "vector_norm", "projector_delta_rms"],
        default="projector_delta_rms",
        help="Normalize candidate vectors before comparison. projector_delta_rms equalizes effect on the captured projector input.",
    )
    parser.add_argument("--target-id", default="skc3vo", help="Reference id used for labeling the normalization target.")
    parser.add_argument("--drop-collinear", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collinear-cosine", type=float, default=0.9999)
    args = parser.parse_args()

    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    projector_input = capture["base_projector_input"].float()
    base_output = capture["base_projector_output"].float()
    captured_lora_output = capture.get("lora_projector_output")
    capture_vector = capture.get("projector_vector")
    if capture_vector is None:
        raise ValueError("Capture does not contain projector_vector")
    capture_vector = capture_vector.float()
    strength = float(args.strength if args.strength is not None else capture.get("lora_strength", 0.01))

    base_replay = replay_output(base_output, projector_input, None, 0.0)
    skc_replay = replay_output(base_output, projector_input, capture_vector, strength)
    base_replay_exact = replay_output(capture["base_projector_output"], capture["base_projector_input"], None, 0.0, exact_dtype=True)
    skc_replay_exact = replay_output(capture["base_projector_output"], capture["base_projector_input"], capture_vector, strength, exact_dtype=True)

    vectors = []
    vectors.extend(load_lora_vectors(args.lora_dir))
    existing = {name for name, _, _ in vectors}
    for name, vector, source in vector_recipes(capture_vector):
        if name not in existing:
            vectors.append((name, vector, source))
    vectors.extend(load_optimized_vectors(args.optimized))

    target_delta = projector_delta(projector_input, capture_vector, strength)
    target_delta_rms = rms(target_delta)
    target_vector_norm = float(torch.linalg.vector_norm(capture_vector.float()))

    records = []
    dropped_records = []
    kept_delta_directions = []
    for name, vector, source in vectors:
        vector = vector.float()
        raw_delta = projector_delta(projector_input, vector, strength)
        raw_delta_rms = rms(raw_delta)
        raw_vector_norm = float(torch.linalg.vector_norm(vector))

        normalization_scale = 1.0
        if args.normalize == "vector_norm":
            normalization_scale = target_vector_norm / raw_vector_norm if raw_vector_norm else 0.0
        elif args.normalize == "projector_delta_rms":
            normalization_scale = target_delta_rms / raw_delta_rms if raw_delta_rms else 0.0

        normalized_vector = vector * float(normalization_scale)
        out = replay_output(base_output, projector_input, normalized_vector, strength)
        delta = out.float() - base_output.float()
        delta_rms = rms(delta)
        duplicate_of = None
        duplicate_cosine = None
        if args.drop_collinear and delta_rms > 0.0:
            for prior_name, prior_delta in kept_delta_directions:
                c = cosine(delta, prior_delta)
                if c >= args.collinear_cosine:
                    duplicate_of = prior_name
                    duplicate_cosine = c
                    break

        row = OrderedDict(
            [
                ("id", name),
                ("source", source),
                ("strength", strength),
                ("raw_vector", [float(x) for x in vector.tolist()]),
                ("raw_vector_norm", raw_vector_norm),
                ("raw_delta", tensor_stats(raw_delta)),
                ("raw_delta_cosine_to_skc3vo", cosine(raw_delta, target_delta)),
                ("raw_delta_rms_ratio_to_skc3vo", raw_delta_rms / target_delta_rms if target_delta_rms else 0.0),
                ("normalization_scale", float(normalization_scale)),
                ("normalized_vector", [float(x) for x in normalized_vector.tolist()]),
                ("normalized_vector_norm", float(torch.linalg.vector_norm(normalized_vector))),
                ("output", tensor_stats(out)),
                ("delta", tensor_stats(delta)),
                ("delta_cosine_to_skc3vo", cosine(delta, target_delta)),
                ("delta_rms_ratio_to_skc3vo", delta_rms / target_delta_rms if target_delta_rms else 0.0),
            ]
        )
        if duplicate_of is not None:
            row["dropped_duplicate_of"] = duplicate_of
            row["dropped_duplicate_cosine"] = duplicate_cosine
            dropped_records.append(row)
            continue

        records.append(row)
        if delta_rms > 0.0:
            kept_delta_directions.append((name, delta.detach().clone()))

    records_sorted = sorted(records, key=lambda row: row["delta_cosine_to_skc3vo"], reverse=True)
    report = OrderedDict(
        [
            ("schema", "krea_txtfusion_projector_replay.v2"),
            ("capture", str(args.capture)),
            ("prompt", capture.get("prompt")),
            ("prompt_id", capture.get("prompt_id")),
            ("latent_mode", capture.get("latent_mode")),
            ("noise_t", capture.get("noise_t")),
            ("lora_strength", strength),
            ("normalize", args.normalize),
            ("normalization_target_id", args.target_id),
            ("normalization_target_delta_rms", target_delta_rms),
            ("normalization_target_vector_norm", target_vector_norm),
            ("drop_collinear", args.drop_collinear),
            ("collinear_cosine", args.collinear_cosine),
            ("metadata", OrderedDict([
                ("stage", "resample"),
                ("tensor_name", "txtfusion.projector"),
                ("pre_projector_shape", list(projector_input.shape)),
                ("pre_projector_dtype", str(capture["base_projector_input"].dtype).replace("torch.", "")),
                ("post_projector_shape", list(base_output.shape)),
                ("post_projector_dtype", str(capture["base_projector_output"].dtype).replace("torch.", "")),
                ("branch", "cond"),
                ("cfg", 1.0),
            ])),
            ("sanity", OrderedDict([
                ("base_replay_vs_capture", compare_tensors(base_output, base_replay)),
                ("skc_replay_vs_capture", compare_tensors(captured_lora_output, skc_replay) if captured_lora_output is not None else None),
                ("base_replay_exact_vs_capture", compare_tensors(capture["base_projector_output"], base_replay_exact)),
                ("skc_replay_exact_vs_capture", compare_tensors(captured_lora_output, skc_replay_exact) if captured_lora_output is not None else None),
            ])),
            ("records", records),
            ("dropped_collinear_records", dropped_records),
            ("top_by_delta_cosine_to_skc3vo", records_sorted[:12]),
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    print("Sanity:")
    print(f"  base replay rms error: {report['sanity']['base_replay_vs_capture']['rms_error']:.9f}")
    if report["sanity"]["skc_replay_vs_capture"] is not None:
        s = report["sanity"]["skc_replay_vs_capture"]
        print(f"  skc replay rms error:  {s['rms_error']:.9f}")
        print(f"  skc replay max error:  {s['max_abs_error']:.9f}")
        sx = report["sanity"].get("skc_replay_exact_vs_capture")
        if sx is not None:
            print(f"  skc exact rms error:   {sx['rms_error']:.9f}")
            print(f"  skc exact max error:   {sx['max_abs_error']:.9f}")
    print(f"Normalization: {args.normalize} target_rms={target_delta_rms:.6f}")
    if dropped_records:
        print("Dropped collinear records:")
        for row in dropped_records:
            print(f"  {row['id']} duplicate_of={row['dropped_duplicate_of']} cos={row['dropped_duplicate_cosine']:.6f}")
    print("Top normalized projector delta shapes:")
    for row in records_sorted[:10]:
        print(
            f"  {row['id']:<36} rms={row['delta']['rms']:.6f} "
            f"raw_ratio={row['raw_delta_rms_ratio_to_skc3vo']:.4f} "
            f"scale={row['normalization_scale']:.4f} cos={row['delta_cosine_to_skc3vo']:.4f}"
        )


if __name__ == "__main__":
    main()
