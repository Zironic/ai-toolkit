# PR D - Native FP8 frozen-base training (step-by-step)

> **git-bug:** `db47d2d`. Strategy/gates/fallback: `docs/decisions/UPSTREAM_PR_PLAN.md`
> PR D section. Status lives in the ticket; this file is the execution plan.

Hard dependency on PR C: the FP8-native training path is a branch inside
`_BouncingLinearFn` (the streaming autograd function C installs) - without C's
hooks there is nowhere for this code to run. The *diff* is small and
self-contained; the dependency is structural. Do not attempt a standalone
resident-FP8 variant - new code, weaker perf story.

Prep work (steps 1-3) can happen while C is in review; the draft PR itself can
exist early as evidence for C's value.

## Steps

1. **Prerequisite: `f1c45ae` resolved** (done in PR C phase 1) - D's headline
   claim depends on the gate reliably enabling on real runs.
2. **Inventory the port surface** (small): the FP8 branch of
   `_BouncingLinearFn.forward`/`backward` (grad-input via `_scaled_mm`,
   including the Ada rowwise-scale workaround), the attach-time gate loop in
   `attach_smart_training`, `_refresh_training_fp8_flags`, and the
   `layer_offloading_fp8_training` config flag (default false).
3. **Correctness harness.** Synthetic GPU test through the venv: FP8-native
   forward + grad-input vs the dequant reference path - output and grad
   tolerances recorded, `_scaled_mm` call counter proves the native path ran,
   fallback counter stays 0 on supported shapes and goes positive (without
   error) on an unsupported shape.
4. **Gate matrix test.** Each gate individually false (trainable weight,
   non-FP8 dtype, CPU tensors, missing `_scaled_mm` simulated) falls back to
   dequant with at most one warning.
5. **Quanto stance.** Follow PR A's decision: torchao-scoped with clean Quanto
   fallback, or dual-layout if A shipped it. Keep the two PRs' support
   matrices identical - one story for reviewers.
6. **Extract onto the C branch** (stacked) as `pr/fp8-native-training`. Diff
   should touch only `manager_modules.py`, `manager.py` (gate + refresh),
   `config_modules.py`, and tests.
7. **Perf evidence.** One before/after comparison on a real run (step time +
   `_scaled_mm` calls/step from the perf log) - this is the PR's selling
   point; include it in the description.
8. **Open as draft early** (optional but recommended) linked from C's PR
   description; convert to ready when C merges.

## Done when

Draft PR exists stacked on C with the correctness harness green and perf
evidence attached; converted to ready and merged after C.
