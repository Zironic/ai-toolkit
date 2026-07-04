# Upstream PR strategy for the memory-management work

> **git-bug:** `db47d2d` (open) — isolate the resident-sampling PR first.
> Status lives in the ticket; this file is the strategy.

> **Terminology note.** The reserve vocabulary was renamed; see the glossary at the top of `../../tasks/open/AUTOTUNE_PLAN.md`. In short: **working_reserve** = our transient working set (the old "headroom"), **system_reserve** = measured uncontrolled VRAM (context/cuDNN/cudagraphs/Windows/other apps), **wddm_margin** = the cushion above the ~500 MB WDDM churn cliff, **usable** = what's left to place resident weights + ring into. "headroom" in the prose below means **working_reserve**.

This document describes how to turn the local `faster-dop` memory-management work into upstreamable PRs for `ostris/ai-toolkit` without dragging in local-only workflow code.

The important correction from the earlier plan: do not lead with the full training streaming core. Lead with the independent, model-agnostic sampling improvement first. The training memory manager is valuable, but it has more coupling, more risk, and more Krea/4070 assumptions to generalize before it is a good first upstream PR.

## Constraints

- Every PR must be off by default or preserve current behavior when its flags are off.
- Unsupported GPUs, unsupported models, or missing PyTorch features must fall back cleanly.
- Do not include local-only workflow changes: DOP prompt cache experiments, TE worker changes, `captions_*` helpers, `search_replace.py`, Hugging Face drive setup, or dataset-specific defaults.
- Heavy diagnostics are development tools, not necessarily upstream product surface. Upstream should get only low-noise diagnostics needed to understand which path was selected and why.
- Any training-memory PR must become any-GPU / any-model enough before submission. Krea-specific assumptions should be gated or moved into model integration code.

## Recommended upstream stack

Fewer PRs are better here. The earlier 9-10 PR stack was useful for thinking, but it is too much review overhead and some of the ordering was backwards relative to the code's real dependency graph.

The committed stack is **A then C**. B stays local (its measurements serve as
evidence in C's PR description, but the tool is not upstream product surface).
D is a small, self-contained follow-up once C lands.

| PR | Theme | Depends on | Status |
| --- | --- | --- | --- |
| A | Fast resident sampling for quantized/turbo models + native FP8 sampling | nothing | Lead PR - build now |
| P | DXGI shared-budget probe + pinned-memory crash guard (bugfix framing: upstream `attach` pins unboundedly, `manager_modules.py:204`) | nothing | Standalone, parallel with A; minimal clamp only, intelligence stays in C |
| C | Bounded training streaming core + bounce pool + safety fixes + automatic memory budget planner + stream attach adapter | P (soft: ledger-proxy fallback without it) | After A; gated on validation (see ticket) |
| D | Native FP8 frozen-base training only | C (hard: lives inside C's streaming autograd fn) | Self-contained diff, stacked follow-up after C |
| B | Training performance log / benchmark tooling | - | **Local only** - evidence for C, not a PR |

Step-by-step execution plans: `tasks/open/UPSTREAM_PR_{A,P,C,D}_*_PLAN.md`.
Post-#930 note: upstream `qfloat8` = Quanto once ostris/ai-toolkit PR #930
merges; A and D must either scope native FP8 to torchao-layout tensors (with
clean Quanto fallback) or ship dual-layout extraction - decide in A, mirror in D.

Checkpoint-retention autotuning was dropped from D: in practice it did not
help (local finding, 2026-07-05). The retention knob itself stays local.

## PR A: Fast resident sampling and native FP8 sampling

### One-sentence summary

Make FP8-quantized models load into a sampling-friendly resident layout and optionally run native FP8 GEMM on supported GPUs, instead of re-streaming and dequantizing every layer on every denoise step.

### Why this should go first

- It is independent of the training streaming core.
- It is model-agnostic: the current local implementation iterates `module.modules()` and does not require Krea-specific `.blocks` structure.
- It is GPU-safe: native FP8 execution is already gated behind `_scaled_mm`, SM89+, correct dtype/shape constraints, and same-device qdata/scale tensors.
- It is easy to explain: faster sampling for quantized/turbo models, with graceful fallback everywhere else.
- It gives Ostris a useful feature without asking him to review the full training offload system.

### Config surface

Two flags, both default `false`:

- `layer_offloading_smart_sampling`: use a resident sampling layout while sampling, then restore the previous training/offload layout afterward.
- `layer_offloading_fp8_sampling`: inside that layout, use native FP8 GEMM when supported. This is a second independent gate, not silently implied by smart sampling.

### Functions / code to hand-port

Port by hand from the local branch. Do not cherry-pick bundled commits, because they pull in the training streaming core and unrelated diagnostics.

From `toolkit/memory_management/manager.py`:

- `inference_resident`
- `_smart_sampling_plan`
- `_sampling_candidates`
- `_enable_fp8_sampling`
- `_disable_fp8_sampling`
- `_move_module_parameters`
- `_move_unmanaged_parameters`
- `_clear_cuda_pipeline_state`, trimmed for PR A
- `_cuda_memory`, `_format_cuda_memory`, `_diagnostics_enabled`
- byte helpers such as `_module_bytes`, `_direct_module_bytes`, `_stream_bytes`
- constants such as `LINEAR_MODULES`, `CONV_MODULES`, and any small sampling-only helpers they need

From `toolkit/memory_management/manager_modules.py`:

- `fp8_linear_inference`
- `_FP8_STATS`
- `_fp8_stats_enabled`

Wiring:

- `jobs/process/BaseSDTrainProcess.py`: wrap sampling in `MemoryManager.inference_resident(..., fp8_sampling=...)` only when the smart sampling flag is enabled.
- `toolkit/config_modules.py`: add the two config flags.
- Optional UI/docs: expose both flags and explain that FP8 sampling only runs when FP8 tensors are already resident on a supported CUDA device.

### Important porting notes

- Trim `inference_resident` so it does not depend on training-streaming state such as `_smart_training_plan`, `_fp8_training_layers`, training rings, or bounce pools.
- In PR A, `_clear_cuda_pipeline_state` can degrade to clearing sampling-local state and calling `torch.cuda.empty_cache()` if needed. It should not require the training ring machinery.
- The partial fallback branch may reuse upstream's existing `attach(..., offload_percent=1.0)` path if the fully resident move cannot fit.
- Keep diagnostics behind the existing diagnostics/debug gate. Default behavior should be quiet.
- Do not include training streaming, bounce pool, offload profiler, checkpoint autotuner, DOP, TE worker, captions helpers, or local scripts.

### Acceptance criteria

- With both flags off, sampling behavior is unchanged.
- Non-quantized models patch nothing and sample normally.
- On SM < 8.9, missing `_scaled_mm`, CPU, unsupported shapes, wrong dtype, or off-card qdata/scale tensors, `fp8_linear_inference` returns `None` and the normal forward path runs.
- At most one clear warning is emitted for unsupported requested native FP8 sampling; no spam.
- If the resident move OOMs, the previous layout is restored and sampling falls back cleanly.
- On exception during sampling, the training/offload layout is restored.
- PR A builds and runs without the training streaming core from PR C.

### Verification

Use the project venv and the existing CUDA-test methodology: `venv\Scripts\python.exe`, synthetic modules where possible, synchronize before timing, and count native FP8 calls/fallbacks.

```powershell
venv\Scripts\python.exe -m py_compile toolkit\memory_management\manager.py `
  toolkit\memory_management\manager_modules.py toolkit\config_modules.py `
  jobs\process\BaseSDTrainProcess.py
```

Additional checks:

- FP8 GEMM smoke test:
  - CUDA qdata/scale on supported hardware returns an output through the native path;
  - CPU qdata/scale returns `None` / falls back;
  - output shape matches a reference `nn.Linear` forward.
- Model-agnostic sampling check:
  - run a short sample on a non-Krea quantized model such as Z-Image or Flux float8 with smart sampling + FP8 sampling enabled;
  - confirm sample is produced and diagnostics report resident/streamed/FP8 counts if diagnostics are enabled.
- Fallback check:
  - same flags on a non-quantized model still sample successfully;
  - if available, test on pre-Ada capability path or simulate unsupported `_scaled_mm`.
- Disabled-default check:
  - flags off should match current sampling behavior.
- If UI is included, run TypeScript and call out any pre-existing `.next/types` route-param noise separately.

### PR description should say

- Problem: quantized/turbo models can spend sampling time re-streaming or dequantizing weights instead of keeping an inference-friendly layout.
- Change: add explicit resident sampling and optional native FP8 sampling flags.
- Fallback: unsupported hardware/models automatically use existing forward paths.
- Not included: training streaming, Krea smart training integration, bounce pool, checkpoint autotuner, heavy profiler, DOP/TE/caption workflow changes.

## PR B: Training performance log / lightweight benchmark tooling

This is independent and useful, but optional. It may be better to keep it local unless Ostris wants it.

Scope if submitted:

- opt-in JSONL performance log;
- total step time;
- data loading and batch preparation;
- normal forward and backward;
- optional DOP section timing if present, but no DOP-cache feature work;
- optimizer step;
- resolution buckets based on total pixels;
- peak allocated/reserved memory by bucket.

Rules:

- off by default;
- no stdout spam;
- no heavy per-layer profiler in this PR;
- useful even without smart memory manager.

Reason to hold it locally:

- downstream users may not need it;
- it can be used as evidence in PR C/D without becoming product surface.

## PR C: Bounded training streaming core

This is the real training memory-manager payload. It should wait until PR A lands or Ostris signals appetite for the larger change.

### Scope

Include:

- bounded streaming core for frozen base weights;
- bounce pool / pinned staging infrastructure;
- safety fixes that only matter once streaming exists, such as `CPU_FILLING` buffer reuse prevention and OOM ring recovery;
- process/job runtime reset for memory-manager global state;
- automatic memory budget planner;
- stream attach adapter;
- minimal low-noise runtime summary when enabled.

Do not include:

- native FP8 training math;
- checkpoint-retention autotuner;
- Krea-only assumptions hardcoded into the core;
- heavy local profiler as product surface.

### Automatic memory budget planner

> **Status note (2026-07-04).** This planner has since been built on `faster-dop`
> and exceeds the spec below (per-bucket measured peaks, cross-bucket worst-case
> governance, grow-to-measured-peak, measured `system_reserve`, manual-mode cliff
> safety net), and the generalization items at the end of this PR C section are
> largely done (no Krea/`.blocks` coupling, device-derived margins, guarded DXGI
> fallback). What still gates PR C is *validation*, not generality — see the
> ticket (`db47d2d`) for the current blocker list. The spec below is kept as the
> design rationale.

The smart training path should not require users to guess a headroom number before it is safe. Manual override is useful for development, but `auto` should be the upstream-friendly default.

The planner should be resolution-aware. A single global headroom number is too crude because activation/workspace pressure scales with pixel count, while model weight size is mostly fixed. The safe plan is a two-stage policy: start conservative from static estimates, then refine per resolution bucket from measured peaks.

Cold-start heuristic:

- determine usable VRAM from device total/free and current torch allocated/reserved state;
- define a resident-model floor: keep at least a small part of the base model resident, initially around `500 MiB` minimum and preferably `1 GiB`;
- define a hard allocation roof: leave at least about `1.5 GiB` for Windows/WDDM/display/driver pressure and never plan up to reported total VRAM;
- estimate activation, PyTorch allocator cache, recompute, and temporary workspace headroom from both model size and the current resolution bucket;
- use a conservative model-size fallback before measurements exist, starting around `50%` of base model size for mid/high resolution buckets;
- combine the hard roof and working-headroom estimate before choosing resident/offloaded split;
- treat the resident floor as a minimum, not a target;
- for a 12 GiB FP8 base model, 50% working headroom is about 6 GiB, so with OS/display safety this lands near the locally successful 7 GiB headroom region;
- prefer slightly slower but safe over OOM-prone.

Resolution bucket policy:

- bucket each training batch by total pixel count, using the same nearest-square buckets as the perf log: `256`, `512`, `768`, `1024`, and optionally an overflow bucket for larger jobs;
- maintain a rolling peak record per bucket for allocated/reserved memory during forward/backward, plus an incremental peak above the pre-step baseline;
- plan headroom for the current bucket from the larger of:
  - cold-start estimate for that bucket;
  - measured rolling peak for that bucket plus a safety margin;
  - neighboring-bucket interpolation if the current bucket has too little history;
- let lower-resolution buckets keep more model resident when measured activation pressure is low;
- force higher-resolution buckets to offload more aggressively when measured activation pressure grows;
- decay or invalidate bucket measurements when key settings change, such as batch size, gradient checkpointing, DOP, checkpoint-retention level, dtype/quantization, optimizer, or native FP8 training mode;
- after an OOM, increase the safety margin for that bucket and retry with a smaller resident set.

This gives the planner a safe first step and then lets it learn that, for example, 256/512 jobs can afford more resident model memory than 768/1024 jobs on the same card.

Acceptance criteria:

- streaming memory is bounded by the planner or explicit override;
- automatic headroom is selected per resolution bucket, not as one fixed global number;
- CPU pinned memory is bounded and does not scale accidentally with model size;
- in-flight buffers cannot be reused while still being filled or consumed;
- OOM recovery clears ring state cleanly;
- sequential jobs do not inherit stale memory-manager global state;
- disabled path preserves existing behavior.

Generalization work before upstream:

- remove or gate Krea/4070-specific assumptions;
- make candidate layer selection model-provided or model-agnostic;
- decide whether the first integration should be Krea or a small synthetic/model-agnostic example;
- keep heavy profiler local unless requested.

## PR D: Native FP8 frozen-base training

This PR depends on PR C, but is otherwise very self-contained: the FP8-native
forward/backward path is a drop-in alternative to the dequant path behind
existing gates, touching only the streamed-linear forward and its attach-time
gating. That makes it a small, reviewable follow-up rather than a second big PR.

### Native FP8 frozen-base training

Goal: for frozen FP8 base weights in LoRA-style training, use native FP8 scaled matmul for forward and backward grad-input where safe, instead of materializing BF16 weights for every streamed layer.

Gates:

- quantized FP8 base;
- frozen base weight / no grad-weight requirement;
- qdata/scale/activation on same CUDA device;
- `_scaled_mm` available;
- supported capability and shape/dtype constraints.

Fallback:

- unsupported layers use existing BF16/dequant path;
- trainable weights are not made FP8-native;
- no optimizer-state FP8 changes.

### Checkpoint-retention tuning: dropped from D (local only)

Formerly part of this PR. Dropped 2026-07-05: in practice the autotuned
checkpoint retention did not deliver a useful speedup, so it is not worth
upstream review cost or maintenance surface. The `checkpoint_autotuner.py`
machinery and its config knob stay local-only.

## Local-only work to keep out of upstream PRs

- DOP prior disk cache experiments;
- DOP prompt/cache behavior tied to local dataset workflow;
- TE worker changes unless submitted as a separate text-embedding-cache PR;
- `captions.json` workflow helpers;
- local search/replace scripts;
- Hugging Face cache-drive configuration;
- project-specific default paths or LA Jinx job assumptions;
- broad UI restructuring unrelated to the submitted feature;
- heavy per-layer offload profiler unless Ostris asks for it.

## Extraction mechanics for PR A

1. Create a clean worktree from upstream `main`, e.g. branch `pr/fp8-resident-sampling`.
2. Hand-port the bounded function set for PR A. Do not cherry-pick bundled local commits.
3. Keep the diff narrow:
   - `toolkit/memory_management/manager.py`
   - `toolkit/memory_management/manager_modules.py`
   - `toolkit/config_modules.py`
   - `jobs/process/BaseSDTrainProcess.py`
   - optional UI/docs files
4. Verify with py_compile, CUDA smoke tests, sampling fallback tests, and TypeScript if UI changed.
5. PR description explicitly lists what is not included.

## Review hygiene checklist

- Defaults preserve current behavior.
- Feature flags are explicit and independent.
- Unsupported hardware/config falls back cleanly.
- Native FP8 only runs when FP8 tensors are already resident on the correct CUDA device.
- No hidden coupling to Krea unless the PR is explicitly a Krea integration.
- No local dataset/workflow code.
- Diagnostics are opt-in or low-noise.
- Python changed files compile.
- Relevant CUDA smoke tests pass.
- If UI changed, TypeScript results are reported with pre-existing noise separated.

## Open questions for Ostris

- Does he want the heavy performance/offload diagnostics upstream, or should they stay local/dev-only?
- Should smart memory options live under existing layer offload config or under a new memory-management section?
- For PR C, should the streaming core land with Krea as first integration, or with a smaller model-agnostic example?
- Should the automatic memory budget planner expose only `auto` plus advanced override, or show derived resident/headroom/ring numbers in UI?
- Should per-resolution bucket headroom be surfaced in logs/UI, or kept as internal planner state unless diagnostics are enabled?
- Should the OS/display reserve be Windows/WDDM-specific or platform-dependent generally?
- What support matrix does he want for native FP8 paths?

## One-sentence upstream framing

Start with the independent sampling win: explicit resident sampling plus optional native FP8 GEMM for quantized models, then follow with the larger training streaming system only after the first PR proves the approach and maintainer appetite.
