> **SUPERSEDED 2026-07-12** - archived, not shipped. The A/P/C/D upstream stack
> and the model-agnostic subplans were retired when the immutable arena runtime
> became the sole transformer backend (commit `3dd7f38`): PRs C and D were built
> on `_BouncingLinearFn`, which no longer exists on the active path. Kept as the
> record of the reasoning. **Current plan:**
> `tasks/open/UPSTREAM_ARENA_EXTRACTION_PLAN.md` (PR P's crash-guard framing and
> upstream audit survive there as Stage 1; PR A as the deferred Stage 4).

# PR A - Resident sampling + native FP8 sampling (step-by-step)

> **git-bug:** `db47d2d`. Strategy/rationale: `docs/decisions/UPSTREAM_PR_PLAN.md`
> (PR A section owns the function list, acceptance criteria, and porting notes -
> not repeated here). Status lives in the ticket; this file is the execution plan.

Independent of everything else. Can start now.

## Steps

1. **Snapshot upstream.** Fetch `ostris/ai-toolkit` `main`; note whether PR #930
   (Quanto qfloat8) has merged - it decides step 5.
2. **Clean worktree.** Branch `pr/fp8-resident-sampling` off upstream `main`
   (worktree, not this checkout). No cherry-picks from `faster-dop` - hand-port only.
3. **Hand-port the sampling function set** listed in the strategy doc (PR A
   section): `inference_resident`, `_smart_sampling_plan`, `_sampling_candidates`,
   FP8 enable/disable, move helpers, byte helpers, `fp8_linear_inference` +
   `_FP8_STATS`. Trim all training-streaming references (`_smart_training_plan`,
   rings, bounce pool, `_fp8_training_*`) as you port - the trimmed
   `_clear_cuda_pipeline_state` may degrade to `empty_cache()`.
4. **Config + wiring.** Add `layer_offloading_smart_sampling` and
   `layer_offloading_fp8_sampling` (both default false) to `config_modules.py`;
   wrap sampling in `BaseSDTrainProcess.py` behind the first flag only.
5. **Quanto stance (post-#930).** Decide: (a) state torchao-layout scope in the
   PR description (Quanto falls back to normal forward - our gates already do
   this), or (b) add dual-layout extraction (`_data`/`_scale` alongside
   `qdata`/`scale`) in `fp8_linear_inference`. Prefer (b) if #930 has merged -
   it is mechanical and pre-empts the obvious review question.
6. **Verify** per the strategy doc's checklist: py_compile the four touched
   files; FP8 GEMM smoke (native path output vs `nn.Linear` reference, CPU
   tensors return `None`); flags-off unchanged behavior; non-quantized model
   samples normally; OOM/exception restores layout. Use the venv CUDA
   methodology (synthetic modules, call counter, synchronize before timing).
7. **Cross-model check.** One short sampling run on a non-Krea quantized model
   (Z-Image or Flux float8) with both flags on.
8. **Open the PR.** Description per strategy doc: problem / change / fallback /
   explicit not-included list. Note the FP8 support matrix question for Ostris.

## Done when

PR is open upstream, all acceptance criteria in the strategy doc demonstrated
in the PR description or linked logs.
