# Quantization Training Benchmark Matrix

Mutable implementation and run status belongs in git-bug ticket `393f8bf`.
This document fixes the comparison protocol and artifact contract.

## Outcome

Produce repeatable, directly comparable Krea2 LoRA training measurements for:

1. `fp8`: FP8 weights with dequantized BF16 forward and grad-input.
2. `fp8_forward`: native FP8 forward with BF16 grad-input.
3. `fp8_forward_backward`: native FP8 forward and FP8 grad-input.
4. `convrot8`.
5. `convrot4`.

The result must be readable as one summary table without manually inspecting
the stdout of each full-model run. Raw evidence must remain available.

## Protocol

`scripts/bench_quantization_matrix.py` drives the existing
`scripts/smoke_krea2_train_cuda.py` full-model harness. All arms use the same
seed, resolution, batch size, adapter setup, compile mode, and number of warmup
and measured steps. The controller rotates arm order and alternates its
direction across repeats to reduce fixed startup, thermal, and run-order bias.

The controller is dry-run-only unless `--execute` is passed. It acquires the
shared smoke GPU lock once per child run and records the matrix run ID in the
lock detail. The child is launched with `--no-gpu-lock` to avoid nested lock
acquisition.

The default loading mode is `normal`, because the benchmark should include the
ordinary load-then-attach lifecycle. `direct-arena` remains available for
iteration when startup time is not part of the question. Startup and steady
training timings are reported separately.

Use at least three repeats for a decision. Compare medians across independent
runs. Treat post-warmup Dynamo frames as invalid steady-state evidence rather
than folding recompilation time into the backend comparison.

## Artifacts

Each benchmark directory contains:

- `manifest.json`: repository identity, dirty state, fixed settings, arm order,
  exact commands, and incremental run status.
- `runs/*.json`: the native full smoke event stream for every arm/repeat.
- `logs/*.log`: complete merged stdout/stderr for diagnosis.
- `runs.csv`: one normalized row per run.
- `report.json`: normalized per-run records and per-arm aggregates.
- `report.md`: compact comparison table and incomplete-run list.

The aggregate table reports steady step time and speedup versus `fp8`, forward
and backward CUDA phase times, allocator peaks, minimum CUDA free memory, and
DXGI usage. Raw JSON remains authoritative for deeper diagnosis.

## Acceptance

- Dry-run output shows exactly the five intended arms and their independent FP8
  forward/backward gates.
- GPU execution cannot start without `--execute` and is serialized through the
  shared smoke lock helper.
- A completed or partially completed matrix can be resumed without rerunning
  native JSON files that already contain a `done` event.
- Reports can be rebuilt from stored artifacts with `--report-only`.
- No real training job or CUDA benchmark is started as part of implementing the
  controller.
