"""Run the controlled fresh-process full-model MegaCache acceptance matrix.

The four required arms separate a cold compile, an unrelated empty-cache
control, TorchInductor's ordinary shared disk cache, and a serialized
MegaCache load into a third empty disk cache. Pass the full-model smoke
arguments after ``--``. Nothing is launched unless ``--execute`` is present.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = REPO_ROOT / "scripts" / "smoke_transformer_train_cuda.py"


@dataclass(frozen=True)
class Arm:
    name: str
    mode: str
    expected: str
    cache_dir: Path
    compile_dynamic: str
    allow_populated: bool = False


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-variants", type=int, default=3)
    parser.add_argument("--compile-dynamic", choices=("true", "false", "none"), default="true")
    parser.add_argument(
        "--compile-coordinate-descent",
        choices=("true", "false", "none"),
        default="false",
    )
    parser.add_argument("--include-invalidation", action="store_true")
    parser.add_argument("--torch-trace", action="store_true")
    parser.add_argument(
        "smoke_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to smoke_transformer_train_cuda.py after --",
    )
    args = parser.parse_args(argv)
    if args.expected_variants < 1:
        parser.error("--expected-variants must be positive")
    if args.smoke_args[:1] == ["--"]:
        args.smoke_args = args.smoke_args[1:]
    if not args.smoke_args:
        parser.error("full-model smoke arguments are required after --")
    reserved = (
        "--output-json",
        "--megacache-",
        "--compile-dynamic",
        "--compile-coordinate-descent",
        "--compile-fullgraph",
    )
    for value in args.smoke_args:
        if any(value == item or value.startswith(item) for item in reserved):
            parser.error(f"matrix-owned smoke option cannot be forwarded: {value}")
    return args


def build_arms(out_dir: Path, compile_dynamic: str, include_invalidation: bool):
    arms = [
        Arm("cold", "produce", "cold", out_dir / "inductor_cold", compile_dynamic),
        Arm(
            "empty_control",
            "control",
            "empty-control",
            out_dir / "inductor_empty_control",
            compile_dynamic,
        ),
        Arm(
            "shared_disk",
            "control",
            "shared-disk",
            out_dir / "inductor_cold",
            compile_dynamic,
            allow_populated=True,
        ),
        Arm(
            "megacache",
            "consume",
            "megacache",
            out_dir / "inductor_megacache",
            compile_dynamic,
        ),
    ]
    if include_invalidation:
        changed = "false" if compile_dynamic in ("true", "none") else "true"
        arms.append(
            Arm(
                "invalidation",
                "consume",
                "invalidation",
                out_dir / "inductor_invalidation",
                changed,
            )
        )
    return arms


def _command(args, arm: Arm, artifact: Path, result: Path) -> list[str]:
    command = [sys.executable, str(SMOKE), *args.smoke_args]
    command.extend(
        [
            "--output-json",
            str(result),
            "--compile-dynamic",
            arm.compile_dynamic,
            "--compile-coordinate-descent",
            args.compile_coordinate_descent,
            "--compile-fullgraph",
            "--megacache-mode",
            arm.mode,
            "--megacache-expected-arm",
            arm.expected,
            "--megacache-expected-variants",
            str(args.expected_variants),
        ]
    )
    if arm.mode in ("produce", "consume"):
        command.extend(["--megacache-artifact", str(artifact)])
    if arm.allow_populated:
        command.append("--megacache-allow-populated-inductor-cache")
    return command


def _phase_signature(rows: list[dict]) -> list[dict]:
    signature = []
    for row in rows:
        if row.get("event") != "phase":
            continue
        signature.append(
            {
                "phase": row["phase"],
                "pred_checksum": row.get("pred_checksum"),
                "grad_checksum": row.get("grad_checksum"),
                "loss": row.get("loss"),
            }
        )
    return signature


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv=None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    artifact = out_dir / "full_model.torchcompile_cache"
    arms = build_arms(out_dir, args.compile_dynamic, args.include_invalidation)
    plan = {
        "out_dir": str(out_dir),
        "artifact": str(artifact),
        "arms": [
            {
                "name": arm.name,
                "mode": arm.mode,
                "expected": arm.expected,
                "cache_dir": str(arm.cache_dir),
                "compile_dynamic": arm.compile_dynamic,
            }
            for arm in arms
        ],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("[megacache-matrix] dry run only; add --execute to launch full models")
        return 0
    if out_dir.exists():
        raise SystemExit(f"refusing to reuse matrix output directory: {out_dir}")
    out_dir.mkdir(parents=True)

    results = {}
    failures = []
    for arm in arms:
        arm.cache_dir.mkdir(parents=True, exist_ok=True)
        result_path = out_dir / f"{arm.name}.json"
        command = _command(args, arm, artifact, result_path)
        env = os.environ.copy()
        env["TORCHINDUCTOR_CACHE_DIR"] = str(arm.cache_dir)
        env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        if args.torch_trace:
            trace_dir = out_dir / "torch_trace" / arm.name
            trace_dir.mkdir(parents=True, exist_ok=True)
            env["TORCH_TRACE"] = str(trace_dir)
        print(f"[megacache-matrix] starting {arm.name}")
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        (out_dir / f"{arm.name}.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (out_dir / f"{arm.name}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        row = {"returncode": completed.returncode, "result": str(result_path)}
        if result_path.is_file():
            rows = json.loads(result_path.read_text(encoding="utf-8"))
            done = next(
                (item for item in reversed(rows) if item.get("event") == "done"),
                None,
            )
            row.update({"summary": done, "phase_signature": _phase_signature(rows)})
        results[arm.name] = row
        if completed.returncode:
            failures.append(f"{arm.name} exited with {completed.returncode}")
            break

    if "cold" in results and results["cold"].get("phase_signature"):
        baseline = results["cold"]["phase_signature"]
        for name, result in results.items():
            if name == "cold" or "phase_signature" not in result:
                continue
            if result["phase_signature"] != baseline:
                failures.append(
                    f"{name} predictions, gradients, or losses differ from cold"
                )
    else:
        failures.append("cold arm produced no comparable phase signature")

    summary = {**plan, "results": results, "failures": failures}
    _write_json_atomic(out_dir / "matrix_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    if failures:
        raise SystemExit("\n".join(["MEGACACHE MATRIX FAILURES:", *failures]))
    print("[megacache-matrix] all full-model MegaCache arms passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
