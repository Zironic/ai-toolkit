"""Run and summarize the Krea2 quantization training benchmark matrix.

The controller deliberately does not run a GPU benchmark unless ``--execute``
is present.  Every child smoke writes its native JSON plus a complete stdout
log.  The controller then produces normalized JSON, CSV, and Markdown reports,
so comparisons do not depend on reading console output by hand.

Examples:

    # Inspect the exact 15-run schedule without touching the GPU.
    venv\\Scripts\\python.exe scripts\\bench_quantization_matrix.py ^
        --out-dir output\\quantization_benchmarks\\krea2_512 ^
        --repeats 3

    # Execute it. Full-model runs are intentionally explicit.
    venv\\Scripts\\python.exe scripts\\bench_quantization_matrix.py ^
        --out-dir output\\quantization_benchmarks\\krea2_512 ^
        --repeats 3 --execute

    # Rebuild reports from already captured native JSON files.
    venv\\Scripts\\python.exe scripts\\bench_quantization_matrix.py ^
        --out-dir output\\quantization_benchmarks\\krea2_512 ^
        --report-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from smoke_runtime import GpuBusy, gpu_lock


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_krea2_train_cuda.py"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Arm:
    name: str
    qtype: str
    fp8_forward: bool = False
    fp8_backward: bool = False


ARMS = (
    Arm("fp8", "float8"),
    Arm("fp8_forward", "float8", fp8_forward=True),
    Arm(
        "fp8_forward_backward",
        "float8",
        fp8_forward=True,
        fp8_backward=True,
    ),
    Arm("convrot8", "convrot8"),
    Arm("convrot4", "convrot4"),
)
ARM_BY_NAME = {arm.name: arm for arm in ARMS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _median(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(finite) if finite else None


def _mean(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def _fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def _git_metadata():
    def run(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def arm_order(repeat_index: int):
    """Rotate and alternate direction to reduce fixed thermal/order bias."""
    arms = list(ARMS)
    shift = (repeat_index // 2) % len(arms)
    arms = arms[shift:] + arms[:shift]
    return list(reversed(arms)) if repeat_index % 2 else arms


def _reject_reserved_smoke_args(values):
    reserved = {
        "--output-json",
        "--qtype",
        "--fp8-training-forward",
        "--fp8-grad-input",
        "--steps",
        "--warmup-steps",
        "--seed",
    }
    for value in values:
        flag = value.split("=", 1)[0]
        if flag in reserved:
            raise SystemExit(
                f"{flag} is controlled by the matrix; use its controller option"
            )


def smoke_command(args, arm: Arm, raw_path: Path):
    command = [
        sys.executable,
        str(SMOKE_SCRIPT),
        "--qtype",
        arm.qtype,
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--seed",
        str(args.seed),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--batch-size",
        str(args.batch_size),
        "--load-mode",
        args.load_mode,
        "--output-json",
        str(raw_path),
        # The matrix controller holds the shared smoke lock around the child.
        # Acquiring it again in the child would deadlock in --wait-for-gpu mode.
        "--no-gpu-lock",
    ]
    if args.cond_cache:
        command.extend(("--cond-cache", args.cond_cache))
    if args.model_path:
        command.extend(("--model-path", args.model_path))
    if args.resolutions:
        command.extend(("--resolutions", args.resolutions))
    if args.no_compile:
        command.append("--no-compile")
    if arm.fp8_forward:
        command.append("--fp8-training-forward")
    if arm.fp8_backward:
        command.append("--fp8-grad-input")
    command.extend(args.smoke_arg)
    return command


def make_manifest(args, out_dir: Path):
    runs = []
    position = 0
    for repeat_index in range(args.repeats):
        for order_index, arm in enumerate(arm_order(repeat_index)):
            position += 1
            run_id = f"r{repeat_index + 1:02d}_p{order_index + 1:02d}_{arm.name}"
            raw_path = out_dir / "runs" / f"{run_id}.json"
            log_path = out_dir / "logs" / f"{run_id}.log"
            runs.append(
                {
                    "run_id": run_id,
                    "repeat": repeat_index + 1,
                    "position": position,
                    "position_in_repeat": order_index + 1,
                    "arm": arm.name,
                    "raw_json": str(raw_path.relative_to(out_dir)),
                    "stdout_log": str(log_path.relative_to(out_dir)),
                    "command": smoke_command(args, arm, raw_path),
                    "status": "planned",
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "repository": _git_metadata(),
        "arms": [asdict(arm) for arm in ARMS],
        "settings": {
            "repeats": args.repeats,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "resolutions": args.resolutions,
            "batch_size": args.batch_size,
            "load_mode": args.load_mode,
            "no_compile": args.no_compile,
            "cond_cache": args.cond_cache,
            "model_path": args.model_path,
            "extra_smoke_args": args.smoke_arg,
        },
        "runs": runs,
    }


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _has_done_event(path: Path):
    try:
        return any(row.get("event") == "done" for row in _load_json(path))
    except (OSError, ValueError, TypeError):
        return False


def _display_command(command):
    return subprocess.list2cmdline([str(part) for part in command])


def print_schedule(manifest):
    settings = manifest["settings"]
    print(
        f"Quantization matrix: {len(manifest['runs'])} runs, "
        f"{settings['repeats']} repeats, {settings['steps']} steps/run "
        f"({settings['warmup_steps']} warmup)"
    )
    for run in manifest["runs"]:
        print(f"\n[{run['run_id']}] {run['arm']}")
        print(_display_command(run["command"]))


def _run_one(command, log_path: Path, *, run_id: str, wait_for_gpu: bool):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with gpu_lock(
        "bench_quantization_matrix",
        detail=run_id,
        wait=wait_for_gpu,
    ):
        with log_path.open("w", encoding="utf-8", newline="") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            return process.wait()


def execute(
    manifest,
    out_dir: Path,
    *,
    resume: bool,
    keep_going: bool,
    wait_for_gpu: bool,
):
    manifest_path = out_dir / "manifest.json"
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)

    for run in manifest["runs"]:
        raw_path = out_dir / run["raw_json"]
        log_path = out_dir / run["stdout_log"]
        if resume and _has_done_event(raw_path):
            run["status"] = "complete"
            run["resumed_utc"] = _utc_now()
            print(f"[matrix] skip complete {run['run_id']}")
            continue
        print(f"[matrix] start {run['run_id']} ({run['arm']})")
        run["status"] = "running"
        run["started_utc"] = _utc_now()
        manifest["updated_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        try:
            returncode = _run_one(
                run["command"],
                log_path,
                run_id=run["run_id"],
                wait_for_gpu=wait_for_gpu,
            )
        except GpuBusy as error:
            print(f"[matrix] {error}", file=sys.stderr)
            returncode = 2
        run["returncode"] = returncode
        run["finished_utc"] = _utc_now()
        run["status"] = (
            "complete" if returncode == 0 and _has_done_event(raw_path) else "failed"
        )
        manifest["updated_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        if run["status"] == "failed" and not keep_going:
            print(f"[matrix] failed {run['run_id']}; see {log_path}", file=sys.stderr)
            return returncode or 1
    return 0


def _event(rows, name):
    return next((row for row in rows if row.get("event") == name), None)


def normalize_run(run, out_dir: Path):
    path = out_dir / run["raw_json"]
    base = {
        "run_id": run["run_id"],
        "arm": run["arm"],
        "repeat": run["repeat"],
        "position": run["position"],
        "status": run.get("status", "unknown"),
        "raw_json": run["raw_json"],
        "stdout_log": run["stdout_log"],
    }
    try:
        rows = _load_json(path)
    except (OSError, ValueError, TypeError) as error:
        return {**base, "status": "failed", "parse_error": str(error)}
    done = _event(rows, "done")
    steps = [row for row in rows if row.get("event") == "train_step"]
    if done is None or not steps:
        return {**base, "status": "failed", "parse_error": "missing done/train_step event"}
    warmup = int(done.get("warmup_steps", 0))
    steady = [row for row in steps if int(row.get("step", -1)) >= warmup] or steps[-1:]
    phases = done.get("phase_medians_ms", {})
    attached = _event(rows, "attached_training_memory") or {}
    finalized = _event(rows, "immutable_runtime_finalized") or {}
    loaded = _event(rows, "loaded_transformer") or {}
    quantized = _event(rows, "quantized_transformer") or {}
    cuda_rows = [row.get("cuda", {}) for row in steps]
    dxgi_rows = [row.get("dxgi", {}) for row in steps]
    immutable = finalized.get("immutable_arena") or attached.get("immutable_arena") or {}
    return {
        **base,
        "status": "complete",
        "qtype": ARM_BY_NAME[run["arm"]].qtype,
        "fp8_forward": ARM_BY_NAME[run["arm"]].fp8_forward,
        "fp8_backward": ARM_BY_NAME[run["arm"]].fp8_backward,
        "steps": len(steps),
        "steady_steps": len(steady),
        "steady_step_mean_s": _mean(row.get("seconds") for row in steady),
        "steady_step_median_s": _median(row.get("seconds") for row in steady),
        "forward_median_ms": phases.get("forward_cuda_ms"),
        "backward_median_ms": phases.get("backward_cuda_ms"),
        "optimizer_median_ms": phases.get("optimizer_cuda_ms"),
        "loss_mean": _mean(row.get("loss") for row in steady),
        "grad_norm_mean": _mean(row.get("grad_norm") for row in steady),
        "peak_torch_allocated_gib": max(
            (row.get("torch_max_allocated_gib") for row in cuda_rows if row.get("torch_max_allocated_gib") is not None),
            default=None,
        ),
        "peak_torch_reserved_gib": max(
            (row.get("torch_reserved_gib") for row in cuda_rows if row.get("torch_reserved_gib") is not None),
            default=None,
        ),
        "min_cuda_free_gib": min(
            (row.get("free_gib") for row in cuda_rows if row.get("free_gib") is not None),
            default=None,
        ),
        "peak_dxgi_usage_gib": max(
            (row.get("usage_gib") for row in dxgi_rows if row.get("usage_gib") is not None),
            default=None,
        ),
        "loaded_transformer_s": loaded.get("seconds"),
        "quantize_s": quantized.get("seconds"),
        "attach_s": attached.get("seconds"),
        "finalize_s": finalized.get("seconds"),
        "resident_sidecar_gib": immutable.get("resident_sidecar_gib"),
        "streamed_blocks": immutable.get("streamed_blocks"),
        "new_compile_frames_after_warmup": sum(
            int(row.get("new_compile_frames", 0)) for row in steady
        ),
    }


CSV_FIELDS = (
    "run_id",
    "arm",
    "repeat",
    "position",
    "status",
    "qtype",
    "fp8_forward",
    "fp8_backward",
    "steps",
    "steady_steps",
    "steady_step_mean_s",
    "steady_step_median_s",
    "forward_median_ms",
    "backward_median_ms",
    "optimizer_median_ms",
    "loss_mean",
    "grad_norm_mean",
    "peak_torch_allocated_gib",
    "peak_torch_reserved_gib",
    "min_cuda_free_gib",
    "peak_dxgi_usage_gib",
    "loaded_transformer_s",
    "quantize_s",
    "attach_s",
    "finalize_s",
    "resident_sidecar_gib",
    "streamed_blocks",
    "new_compile_frames_after_warmup",
    "raw_json",
    "stdout_log",
    "parse_error",
)


def aggregate_runs(runs):
    aggregates = {}
    for arm in ARMS:
        selected = [run for run in runs if run["arm"] == arm.name and run["status"] == "complete"]
        aggregates[arm.name] = {
            "successful_runs": len(selected),
            "steady_step_median_s": _median(run.get("steady_step_mean_s") for run in selected),
            "forward_median_ms": _median(run.get("forward_median_ms") for run in selected),
            "backward_median_ms": _median(run.get("backward_median_ms") for run in selected),
            "peak_torch_allocated_gib": _median(run.get("peak_torch_allocated_gib") for run in selected),
            "peak_torch_reserved_gib": _median(run.get("peak_torch_reserved_gib") for run in selected),
            "min_cuda_free_gib": _median(run.get("min_cuda_free_gib") for run in selected),
            "peak_dxgi_usage_gib": _median(run.get("peak_dxgi_usage_gib") for run in selected),
        }
    baseline = aggregates["fp8"]["steady_step_median_s"]
    for aggregate in aggregates.values():
        seconds = aggregate["steady_step_median_s"]
        aggregate["speedup_vs_fp8"] = baseline / seconds if baseline and seconds else None
    return aggregates


def render_markdown(manifest, runs, aggregates):
    settings = manifest["settings"]
    lines = [
        "# Krea2 quantization benchmark",
        "",
        f"Commit: `{manifest['repository'].get('commit') or 'unknown'}` "
        f"(dirty: {manifest['repository'].get('dirty')})",
        "",
        f"Protocol: {settings['repeats']} repeats, {settings['steps']} steps/run, "
        f"{settings['warmup_steps']} warmup, {settings['width']}x{settings['height']}, "
        f"batch {settings['batch_size']}, compile={'off' if settings['no_compile'] else 'on'}.",
        "",
        "Aggregate values are medians across independent runs. Step time uses each "
        "run's steady-step mean.",
        "",
        "| arm | runs | step s | speedup vs fp8 | fwd ms | bwd ms | peak alloc GiB | peak reserved GiB | min CUDA free GiB | DXGI usage GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = aggregates[arm.name]
        lines.append(
            f"| {arm.name} | {row['successful_runs']} | "
            f"{_fmt(row['steady_step_median_s'])} | {_fmt(row['speedup_vs_fp8'])}x | "
            f"{_fmt(row['forward_median_ms'], 1)} | {_fmt(row['backward_median_ms'], 1)} | "
            f"{_fmt(row['peak_torch_allocated_gib'])} | {_fmt(row['peak_torch_reserved_gib'])} | "
            f"{_fmt(row['min_cuda_free_gib'])} | {_fmt(row['peak_dxgi_usage_gib'])} |"
        )
    failures = [run for run in runs if run["status"] != "complete"]
    if failures:
        lines.extend(("", "## Incomplete runs", ""))
        for run in failures:
            detail = run.get("parse_error") or run.get("status")
            lines.append(f"- `{run['run_id']}`: {detail}")
    return "\n".join(lines) + "\n"


def write_reports(manifest, out_dir: Path):
    runs = [normalize_run(run, out_dir) for run in manifest["runs"]]
    aggregates = aggregate_runs(runs)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "manifest": "manifest.json",
        "aggregates": aggregates,
        "runs": runs,
    }
    _write_json(out_dir / "report.json", report)
    with (out_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            writer.writerow(run)
    (out_dir / "report.md").write_text(
        render_markdown(manifest, runs, aggregates), encoding="utf-8"
    )
    print(f"[matrix] wrote {out_dir / 'report.json'}")
    print(f"[matrix] wrote {out_dir / 'runs.csv'}")
    print(f"[matrix] wrote {out_dir / 'report.md'}")
    return 0 if any(run["status"] == "complete" for run in runs) else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True, help="artifact directory")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--resolutions", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-mode", choices=("direct-arena", "normal"), default="normal")
    parser.add_argument("--cond-cache", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--smoke-arg",
        action="append",
        default=[],
        help="extra child-smoke argument; repeat and use --smoke-arg=--flag",
    )
    parser.add_argument("--execute", action="store_true", help="run the GPU matrix")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip raw JSON files with a done event")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.steps <= 0 or args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        parser.error("need 0 <= --warmup-steps < --steps")
    if args.execute and args.report_only:
        parser.error("--execute and --report-only are mutually exclusive")
    _reject_reserved_smoke_args(args.smoke_arg)
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    manifest_path = out_dir / "manifest.json"
    if args.report_only:
        if not manifest_path.exists():
            raise SystemExit(f"missing manifest: {manifest_path}")
        return write_reports(_load_json(manifest_path), out_dir)

    if args.execute and manifest_path.exists():
        if not args.resume:
            raise SystemExit(
                f"{manifest_path} already exists; choose a new directory or pass --resume"
            )
        manifest = _load_json(manifest_path)
    else:
        manifest = make_manifest(args, out_dir)
    print_schedule(manifest)
    if not args.execute:
        print("\nDry run only. Add --execute to start the GPU benchmark matrix.")
        return 0
    returncode = execute(
        manifest,
        out_dir,
        resume=args.resume,
        keep_going=args.keep_going,
        wait_for_gpu=args.wait_for_gpu,
    )
    report_code = write_reports(manifest, out_dir)
    return returncode or report_code


if __name__ == "__main__":
    raise SystemExit(main())
