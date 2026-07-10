---
name: cuda-testing
description: How to validate CUDA/FP8/memory-manager behavior on the real GPU in this repo - synthetic-model methodology, what needs user permission, and the test-order-leak policy. Load when writing or debugging anything under tests/, benchmarking with scripts/bench_*, or verifying GPU behavior after a memory-management change.
---

# CUDA testing methodology

## The ground rules

- **Small real-GPU scripts are the normal way to check this code; no need to
  ask first.** Run them through the project venv:
  `venv/Scripts/python.exe -m pytest tests/ -q` (finishes in seconds) or an
  ad-hoc script like `venv/Scripts/python.exe scripts/bench_bounce_fill_group.py`.
- **Full training runs (`python run.py ...`) need the user's explicit
  go-ahead**: minutes to hours, real datasets/checkpoints, and a unit test is
  not a substitute. A pre-bash hook asks for confirmation on these.
- Env vars are legitimate ONLY as test harness controls and one-off debug
  scripts. Anything that must affect a real training job goes through the job
  config (and UI schema when applicable) -- env vars do not exist in runs
  launched from the web UI.

## The synthetic-model pattern

Validate real CUDA behavior without a real model:

- Build tiny synthetic modules with the same dtypes/quant wrappers as the
  real path (torchao FP8; note "qfloat8" configs actually run torchao via the
  shim + cache).
- Count kernel-level calls to prove the fast path is taken -- e.g. a
  `_scaled_mm` call counter monkeypatch to confirm FP8 matmuls happen.
- `torch.cuda.synchronize()` before every timing measurement; CUDA is async
  and unsynchronized timings are fiction.
- Assert on mechanism (calls made, bytes moved, residency states), not on
  wall-clock, except in explicitly-named bench scripts.
- Existing tests under `tests/` are the style guide: bounce pool, block
  stream, working-reserve sim, shape keys.

## Test-order leaks: the standing policy

Some `tests/` files pass alone but fail in the full suite, because
process-global state (pin ledger, CUDA allocator, qfloat8->torchao shim)
outlives a test. **Do not hunt these.** The complete procedure:

1. Run the file alone (passes) and the suite with the suspect file ignored
   (passes). That is all the evidence needed to call it a leak, not a
   regression.
2. Note it on git-bug ticket `f2aceba` and move on. No bisecting, no
   sys.modules spelunking, no restructuring other tests.
3. If a test *you are adding* triggers one, test the seam directly instead
   of driving the whole machine (assert a helper collects the right entries
   rather than building real pinned packs).

## Training-shaped smoke without a trainer

`scripts/smoke_krea2_train_cuda.py`: 5 fake LoRA training steps, no dataset,
no trainer process. Baseline ~3.4-4.0 s/step @512px eager.
`--train-compile-blocks` mirrors the trainer wiring. **Freeze the base model
before applying the LoRA** -- the harness depends on it.

## Fail-fast conventions for new code

Prefer explicit config flags and informative errors over silent fallbacks in
training/inference paths. Every new memory-manager behavior gets a focused
test under `tests/` -- the controllers have CPU/sim coverage precisely
because GPU CI does not exist.
