"""Benchmark MegaCache while Arena residency changes across fresh processes.

The five-run sequence models a persistent cache that learns new variants:

    mixed cold -> mixed warm -> full transition -> full warm -> mixed return

Cache misses are measurements, not failures. Model correctness, fullgraph
execution, residency assertions, transfer accounting, and numerical parity
remain hard gates. Pass the common full-model smoke arguments after ``--``.
Nothing is launched unless ``--execute`` is present.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.run_full_model_megacache_matrix import (
        _git_head,
        _phase_parity_failures,
        _phase_signature,
        _write_json_atomic,
    )
except ModuleNotFoundError:
    from run_full_model_megacache_matrix import (
        _git_head,
        _phase_parity_failures,
        _phase_signature,
        _write_json_atomic,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = REPO_ROOT / "scripts" / "smoke_transformer_train_cuda.py"


@dataclass(frozen=True)
class ResidencyRun:
    name: str
    transition: str
    residency: str
    mode: str
    input_artifact: Path
    update_artifact: Path | None = None


def build_sequence(out_dir: Path) -> list[ResidencyRun]:
    mixed = out_dir / "mixed.torchcompile_cache"
    expanded = out_dir / "mixed_full.torchcompile_cache"
    return [
        ResidencyRun("mixed_cold", "cold->mixed", "mixed", "produce", mixed),
        ResidencyRun("mixed_warm", "mixed->mixed", "mixed", "consume", mixed),
        ResidencyRun(
            "full_transition",
            "mixed->full",
            "full",
            "consume",
            mixed,
            expanded,
        ),
        ResidencyRun("full_warm", "full->full", "full", "consume", expanded),
        ResidencyRun("mixed_return", "full->mixed", "mixed", "consume", expanded),
    ]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--mixed-simulated-vram-gib",
        type=float,
        default=10.0,
        help="simulated card size used to force the mixed-residency arms",
    )
    parser.add_argument(
        "--full-working-reserve-gib",
        type=float,
        default=3.0,
        help="real-card execution reserve used for all-resident arms",
    )
    parser.add_argument(
        "--compile-dynamic", choices=("true", "false", "none"), default="true"
    )
    parser.add_argument(
        "--compile-coordinate-descent",
        choices=("true", "false", "none"),
        default="false",
    )
    parser.add_argument(
        "smoke_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to smoke_transformer_train_cuda.py after --",
    )
    args = parser.parse_args(argv)
    if args.mixed_simulated_vram_gib <= 0:
        parser.error("--mixed-simulated-vram-gib must be positive")
    if args.full_working_reserve_gib < 0:
        parser.error("--full-working-reserve-gib must be non-negative")
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
        "--freeze-arena-residency",
        "--simulated-vram-gib",
        "--working-reserve-gib",
        "--expected-residency",
    )
    for value in args.smoke_args:
        if any(value == item or value.startswith(item) for item in reserved):
            parser.error(f"benchmark-owned smoke option cannot be forwarded: {value}")
    return args


def _command(args, run: ResidencyRun, result: Path) -> list[str]:
    command = [sys.executable, str(SMOKE), *args.smoke_args]
    command.extend(
        [
            "--output-json",
            str(result),
            "--compile-dynamic",
            args.compile_dynamic,
            "--compile-coordinate-descent",
            args.compile_coordinate_descent,
            "--compile-fullgraph",
            "--freeze-arena-residency",
            "--expected-residency",
            run.residency,
            "--megacache-mode",
            run.mode,
            "--megacache-expected-arm",
            "cold" if run.mode == "produce" else "megacache",
            "--megacache-artifact",
            str(run.input_artifact),
        ]
    )
    if run.residency == "mixed":
        command.extend(
            [
                "--simulated-vram-gib",
                str(args.mixed_simulated_vram_gib),
                "--working-reserve-gib",
                "-1",
            ]
        )
    else:
        command.extend(
            [
                "--simulated-vram-gib",
                "0",
                "--working-reserve-gib",
                str(args.full_working_reserve_gib),
            ]
        )
    if run.mode == "consume":
        command.append("--megacache-measure-only")
    if run.update_artifact is not None:
        command.extend(
            ["--megacache-update-artifact", str(run.update_artifact)]
        )
    return command


def _compact_result(rows: list[dict], wall_seconds: float) -> dict:
    done = next(
        (item for item in reversed(rows) if item.get("event") == "done"), {}
    )
    runtime = next(
        (
            item
            for item in reversed(rows)
            if item.get("event") == "runtime_finalized"
        ),
        {},
    )
    phases = [item for item in rows if item.get("event") == "phase"]
    megacache = done.get("megacache") or {}
    return {
        "wall_seconds": wall_seconds,
        "first_phase_seconds": phases[0].get("seconds") if phases else None,
        "phase_seconds": [item.get("seconds") for item in phases],
        "cache_load_seconds": megacache.get("load_seconds"),
        "cache_save_seconds": megacache.get("save_seconds"),
        "loaded_artifact_bytes": megacache.get("artifact_bytes_loaded"),
        "saved_artifact_bytes": megacache.get("artifact_bytes_saved"),
        "loaded_artifacts": megacache.get("loaded_artifacts"),
        "saved_artifacts": megacache.get("saved_artifacts"),
        "manifest_matches": megacache.get("manifest_matches"),
        "evidence": megacache.get("evidence"),
        "accounting": runtime.get("accounting"),
        "phase_signature": _phase_signature(rows),
    }


def main(argv=None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    sequence = build_sequence(out_dir)
    matrix_git_head = _git_head()
    plan = {
        "out_dir": str(out_dir),
        "toolkit_git_head": matrix_git_head,
        "compile_dynamic": args.compile_dynamic,
        "mixed_simulated_vram_gib": args.mixed_simulated_vram_gib,
        "full_working_reserve_gib": args.full_working_reserve_gib,
        "runs": [
            {
                "name": run.name,
                "transition": run.transition,
                "residency": run.residency,
                "mode": run.mode,
                "input_artifact": str(run.input_artifact),
                "update_artifact": (
                    None if run.update_artifact is None else str(run.update_artifact)
                ),
            }
            for run in sequence
        ],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("[residency-bench] dry run only; add --execute to launch full models")
        return 0
    if out_dir.exists():
        raise SystemExit(f"refusing to reuse benchmark output directory: {out_dir}")
    out_dir.mkdir(parents=True)

    results = {}
    failures = []
    baseline = None
    for index, run in enumerate(sequence):
        current_head = _git_head()
        if current_head != matrix_git_head:
            failures.append(
                "repository HEAD changed during benchmark: "
                f"started={matrix_git_head}, current={current_head}"
            )
            break
        cache_dir = out_dir / f"inductor_{index}_{run.name}"
        cache_dir.mkdir()
        result_path = out_dir / f"{run.name}.json"
        env = os.environ.copy()
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
        env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        command = _command(args, run, result_path)
        print(f"[residency-bench] starting {run.name} ({run.transition})")
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        wall_seconds = time.perf_counter() - started
        (out_dir / f"{run.name}.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (out_dir / f"{run.name}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        row = {
            "transition": run.transition,
            "residency": run.residency,
            "returncode": completed.returncode,
            "result": str(result_path),
            "wall_seconds": wall_seconds,
        }
        if result_path.is_file():
            rows = json.loads(result_path.read_text(encoding="utf-8"))
            row.update(_compact_result(rows, wall_seconds))
            if baseline is None:
                baseline = row["phase_signature"]
                row["parity_failures"] = []
            else:
                row["parity_failures"] = _phase_parity_failures(
                    baseline, row["phase_signature"], run.name
                )
                failures.extend(row["parity_failures"])
        results[run.name] = row
        if completed.returncode:
            failures.append(f"{run.name} exited with {completed.returncode}")
            break
        current_head = _git_head()
        if current_head != matrix_git_head:
            failures.append(
                "repository HEAD changed during benchmark: "
                f"started={matrix_git_head}, current={current_head}"
            )
            break

    if baseline is None:
        failures.append("mixed cold run produced no comparable phase signature")
    summary = {**plan, "results": results, "failures": failures}
    _write_json_atomic(out_dir / "residency_benchmark_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    if failures:
        raise SystemExit("\n".join(["RESIDENCY BENCHMARK FAILURES:", *failures]))
    print("[residency-bench] all five residency measurements passed correctness gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
