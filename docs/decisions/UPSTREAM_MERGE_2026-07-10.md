# Upstream Merge Decisions - 2026-07-10

## Purpose

This records the manual decisions made while merging `upstream/main` at
`fed9357` into `faster-dop`.

- Pre-merge local commit: `1eb8824`
- Merge commit: `ed8c458`
- Pushed branch: `origin/faster-dop`

The merge changed 47 tracked files. Git merged 40 without content conflicts.
The seven manually resolved files are documented below.

## Resolution Principles

1. Preserve the fork's smart memory management, streamed execution, compiled
   paths, WDDM safety behavior, and TE-cache worker modes.
2. Accept upstream model features and APIs unless they invalidate those paths.
3. Prefer correct eager execution for reference-image calls over silently using
   compiled graphs whose signatures do not contain reference K/V state.
4. Traverse a model only once for TorchAO quantization.

## Manual Decisions

### `extensions_built_in/diffusion_models/krea2/krea2.py`

This was resolved as a functional union.

Kept from the fork:

- Streaming safetensors reads, avoiding a full checkpoint plus cast copy in RAM.
- Unit-at-a-time quantization and the quantized-transformer cache.
- Smart attach, pinned arena, checkpoint keep-last, in-graph and regional
  compile, sampling guards, OOM demotion, and compile-cache behavior.
- `te_only` and `skip_te` cache-worker modes.
- Device-only `.to(device)` for a quantized transformer. Passing a dtype can
  dequantize the whole model into a very large FP32 allocation.

Accepted from upstream:

- Edit/reference-image support and VAE reference latents.
- Qwen3-VL visual processing and the Conv3d patch-embed-to-GEMM replacement.
- Isolated reference attention and the `kv_cache` option.
- Zero-normalized Krea2 CFG: `max(0, guidance_scale - 1)`.
- Assistant-LoRA handling before quantization.

Integration details:

- Preview code builds reference latents before entering the existing sampling
  compile/offload lifecycle, then passes them into `Krea2Pipeline`.
- The local sampling guard, compile setup/fallback, teardown, and OOM recovery
  remain authoritative around the reference-aware pipeline call.
- `vl_processor` is explicitly `None` with `skip_te`; the normal path unpacks
  all four values returned by `_load_text_encoder`.

Review points: verify control-image ordering and whether zero-normalized CFG is
correct for every raw/turbo checkpoint used by the fork.

### `extensions_built_in/diffusion_models/krea2/src/pipeline.py`

- The signature retains local `batch_cfg` and upstream `ref_latents`.
- Batched CFG remains enabled for ordinary text-to-image calls.
- Reference calls deliberately use sequential CFG. Batched CFG creates a
  batch-two K/V cache, while local OOM recovery can fall back to sequential
  batch one. Sequential reference CFG keeps the cache batch shape invariant.
- Upstream reference K/V capture/reuse is retained.
- Local step trimming, incremental demotion, compiled-state invalidation,
  batched-CFG OOM fallback, and tiled VAE-decode recovery are retained.

Tradeoff: reference previews do not get the batched-CFG speedup. Re-enabling it
requires an explicit cache rebuild when CFG mode changes.

### `extensions_built_in/diffusion_models/krea2/src/mmdit.py`

- `Attention.forward` combines local streamed-weight `leaves` with upstream
  `ref_span`, `kv_capture`, and `kv_cache`.
- Reference K/V is captured after QK normalization and RoPE, matching upstream.
- Cached K/V is appended before attention.
- `SingleStreamBlock.forward` forwards reference arguments while retaining the
  local streamed attention/MLP implementation.
- Reference tokens use upstream t=0 modulation and optional isolation masks.
- Reference tokens are excluded from the returned noisy-image prediction.
- Cached-reference masks are appended to the live attention mask.

Compiled-path boundary:

- Regional compile, in-graph sampling, and in-graph training are disabled for a
  call containing reference tokens, K/V capture, or cached K/V reuse.
- Those calls use eager blocks, or normal non-reentrant checkpointing in
  training; reference arguments are forwarded through checkpointed calls.
- Calls without reference state retain the existing compiled/streamed paths.

Reason: the local compiled signatures accept ordinary `(hidden, timestep,
freqs, mask)` inputs and do not represent reference cache state. Eager fallback
is slower but explicit and correct.

Review points: add numerical capture-versus-reuse equivalence coverage and a
reference-conditioned backward test with checkpointing.

### `extensions_built_in/diffusion_models/z_image/z_image.py`

- Preserve `te_only` without constructing the transformer.
- Resolve extras without loading weights: a single-file checkpoint uses the
  standard extras repository, while a full local checkpoint can provide its
  tokenizer/text encoder/VAE.
- Normal jobs use upstream `load_transformer`, including single-file conversion.
- Apply assistant LoRA before quantization.
- Quantize, attach layer offloading once, then perform low-VRAM movement.
- Preserve `skip_te` behavior.

### `jobs/process/BaseSDTrainProcess.py`

Accepted upstream's `include_pretrained_lora=True` parameter on
`get_latest_save_path`, while retaining `name is None`. If enabled and no job
checkpoint exists, the configured pretrained LoRA may be used as the load path.

### `toolkit/dataloader_mixins.py`

Kept the fork's embedding-cache flow:

- Pre-scan cached versus missing files and return early when complete.
- Error clearly when entries are missing but `skip_te`/`FakeTextEncoder` means
  the text encoder is unavailable.
- Encode only `files_needing_encode`.
- Require a control path when control images are part of text embeddings.
- Preserve single/multiple-control selection and atomic cache writes.

The upstream loop over every file would weaken the cache-worker invariant and
caption-change diagnostics.

### `toolkit/util/quantize.py`

Combined the local TorchAO traversal fix with upstream Ostris quantization:

- TorchAO runs once at the root with a filter, then returns. Re-running it for
  each named module can quantize descendants through a parent and then attempt
  to quantize them again directly.
- `ostristype` converts eligible linear modules with
  `convert_linear_to_ostris`.
- Other types continue through Quanto `_quantize_submodule`.
- Retained the installed-version-compatible `IntxWeightOnlyConfig` import.

During resolution, temporarily selecting the whole upstream file pulled in
`UIntXWeightOnlyConfig`, unavailable in the repository's TorchAO version, and
caused test collection to fail. Restoring Git's auto-merged file and resolving
only the conflict hunk fixed it. This intermediate mistake was not committed.

## Automatically Accepted Upstream Changes

All non-conflicting changes at `fed9357` were accepted, including Krea2 text
encoder changes, LTX2/Qwen Image/WAN updates, trainer and LoRA changes, new
Orbit/Ostris quantizers, config/UI exposure, API route hardening, and version
updates.

This is not a claim that every auto-merged line received manual semantic review;
it records that Git applied them without textual conflict and that focused
validation passed with them present.

## Validation

- All seven resolved Python files passed `py_compile`.
- `git diff --cached --check` passed before committing.
- The focused Krea2/offload suite passed: 85 tests, 17 warnings, and 2 subtests.

Coverage included compiled streamed slices, in-graph LoRA and sampling,
model-agnostic in-graph seams, residency invariance, borrowed-pack counts, Krea2
scheduler/SKC behavior, quantized pinning, pinned-arena bypass, residency
ownership, and synthetic training compile behavior.

## Addressed High Risks

### High - Reference mask when no padding exists - Fixed

The defect was confirmed: `_mask(mask)` returns `None` when every token is
valid, and otherwise returns a key-only `(B, 1, 1, L)` mask. Reference isolation
and cached-key concatenation both require explicit live query rows.

The fix keeps the ordinary fast path unchanged and materializes
`(B, 1, L, L)` only when needed:

- Isolated reference attention expands the key mask (or the original all-valid
  `padmask` when `_mask` returned `None`) before applying its query-dependent
  isolation matrix.
- Cached K/V reuse performs the same expansion before concatenating the
  `(B, 1, L, R)` cached-reference key mask.

Tests cover all-true unpadded capture/reuse and a two-sample case with unequal
reference counts and a padded reference token.

### High - Reference K/V capture-versus-reuse equivalence - Covered

Direct MMDiT tests compare the target-token output from a reference capture pass
with a later cached-reuse pass. Coverage includes all-valid masks and unequal
per-sample reference masks. A separate test exercises the public
`predict_velocity` packing and cache-dictionary path. All comparisons pass at
`rtol=1e-5`, `atol=1e-6`, and the tests verify one cache entry per transformer
block.

### High - Reference-conditioned backward with checkpointing - Covered

A focused CPU test compares non-reentrant checkpointed and eager reference
forwards/backwards on identical small models. It compares outputs, target input
gradients, reference input gradients, text-context gradients, gradient presence
for every model parameter, and every populated parameter gradient at
`rtol=1e-5`, `atol=1e-6`.

The comparison passes for both ordinary reference modulation and
`isolate_refs=True`.
## Other Validation Gaps

- No direct full-pipeline test for Krea2 reference-latent packing.
- No full GPU training/preview job was run.
- The broad repository test suite was not run.

## Repository State Note

The plan formerly at `tasks/open/OSTRIS_NAIVE_PREFETCH_BENCHMARK_PLAN.md`
appeared untracked after the pre-merge checkpoint. It was not from upstream,
was not included in either commit, and remained local when the branch was
pushed. It is now archived at
`tasks/done/OSTRIS_NAIVE_PREFETCH_BENCHMARK_PLAN.md`.
