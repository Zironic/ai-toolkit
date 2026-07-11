# Ostris Naive 2 GiB Prefetch Benchmark Plan

> **git-bug:** `267470b` (open) - implementation and benchmark status.
> This document is the durable design and acceptance criteria; run status and
> results belong on the ticket.

## Goal

Build the smallest credible optimized baseline from the current Ostris memory
manager, then compare it with this fork's current memory manager on the same
hardware and workload.

The experimental optimization is intentionally simple:

```text
current managed layer enters its forward wrapper
-> discard layers that are now behind the cursor
-> enqueue the current layer if it is not ready
-> enqueue later managed layers in attach order
-> stop when live current+future materialized weights reach about 2 GiB
-> wait only for the current layer's ready event
-> compute the current layer and release it
```

This is a benchmark comparator, not a proposed production memory manager. It
must stay easy to audit and must not acquire trace learning, adaptive VRAM
planning, WDDM control, bounce pools, promotion/demotion, or model-specific
scheduling.

## Source Baseline

Implement the comparator in a clean worktree based on a pinned
`ostris/ai-toolkit` commit. At plan-writing time, local `upstream/main` is:

```text
fed9357234f12da9e6e00cc6b3e20b34a1369843
```

Before implementation, fetch upstream once, record the selected full SHA on
ticket `267470b`, and do not move the baseline during the benchmark. Record the
full SHA of this fork's comparison commit as well.

The upstream implementation at the pinned snapshot has a per-device ring with
four forward slots, but each layer still stages only its own weight and then
immediately waits for that slot. A deeper ring does not by itself enqueue future
layers. The comparator should preserve the upstream code except for the narrow
forward-prefetch cache and benchmark instrumentation described here.

Do this work in clean sibling worktrees so unrelated local state cannot
contaminate the comparator or its recorded SHAs.

## Benchmark Arms

Use three named arms:

| Arm | Code | Purpose |
| --- | --- | --- |
| `upstream-stock` | Pinned Ostris snapshot, no prefetch patch | Shows the gain from the naive patch itself. |
| `upstream-naive-2g` | Same snapshot plus this fixed 2 GiB forward window | Primary optimized baseline for comparison with this fork. |
| `local-current` | Pinned current fork commit | Measures the full current memory-manager result. |

The primary comparison is `upstream-naive-2g` versus `local-current`.
`upstream-stock` is a calibration arm: without it, a close result would not say
whether the naive optimization was effective or merely harmless.

## Deliberate Constraints

- The target is exactly `2 * 1024**3` bytes. Keep it as one benchmark constant,
  not an environment variable, config option, or UI setting.
- Only forward weights are prefetched. Leave upstream backward re-fetch and
  gradient staging unchanged.
- Use upstream's existing replacement `_mm_forward` wrapper as the trigger. A
  second PyTorch pre-hook per leaf is unnecessary.
- Use attach order as predicted execution order. Do not record or learn an
  execution trace.
- Use one CUDA transfer stream and one ready event per cached entry.
- Do not inspect live free VRAM or reduce the window after an OOM.
- Do not add a pinned-memory budget. The comparator preserves upstream's host
  pinning behavior.
- Do not add native FP8 math, compile support, block streaming, sampling layout
  changes, checkpoint tuning, or any local smart-memory feature.
- CPU and unsupported/custom forward signatures keep the existing upstream
  fallback.

### Preserve naive weaknesses

The comparator preserves upstream limitations and naive-prefetch failure modes
unless the patch would otherwise produce incorrect math or an unsafe CUDA
tensor lifetime. In particular, do not add:

- speculative-transfer cancellation;
- coordination or a shared budget across separate managers;
- checkpoint-recomputation awareness or a backward-phase bypass;
- branch prediction or dynamic execution-order learning;
- backward-aware transfer scheduling;
- allocator-pressure handling or adaptive window reduction.

An OOM, backward interference, wasted branch transfer, or recomputation-induced
contention is a result. Correct autograd semantics and safe CUDA buffer lifetime
are implementation requirements, not optimizations. A separately labeled
diagnostic run may use another window constant only after the fixed 2 GiB
primary result is captured.

## Minimal Design

### 1. Build one deterministic managed-layer list

Extend upstream `MemoryManager` with:

```python
self.forward_layers = []
self.forward_prefetch = None
```

When `LinearLayerMemoryManager` or `ConvLayerMemoryManager` attaches, register
that layer manager once and assign its zero-based `forward_index`. Register only
offloaded/managed layers; resident layers selected by `offload_percent` are not
part of the window.

The existing attach traversal and `modules_processed` set remain authoritative.
Do not introduce another recursive traversal, sort module names, or special-case
Krea blocks. Nested ARA managers keep their own list and cache.

### 2. Add a small per-manager forward cache

Implement a private helper in `manager_modules.py`, conceptually:

```python
class _NaiveForwardPrefetch:
    target_bytes = 2 * 1024**3

    def __init__(self, manager):
        self.manager = manager
        self.entries = {}       # forward_index -> entry
        self.live_bytes = 0
        self.last_index = None
```

All mutable state is instance-owned. Each entry owns:

```text
layer manager identity / forward index
materialized CUDA weight
CUDA bias, if present
target compute dtype
ready event
estimated live materialized bytes
```

The cache owns tensor lifetimes. Do not store prefetched tensors on the module or
replace its CPU `Parameter`; autograd must continue to see and save the CPU
master and use upstream's existing backward path.

No lock is required for the benchmark's single-threaded model execution. State
is per `MemoryManager`, not module-global. The 2 GiB target therefore applies
independently to every attached manager. Parent, nested ARA, transformer, and
text-encoder managers may exceed 2 GiB in aggregate. This is an intentional
naive limitation, not a global memory guarantee.

### 3. Count retained GPU materialized bytes, not source bytes

The 2 GiB target is the expected retained CUDA representation:

- ordinary floating weight: `numel * element_size`;
- quantized weight: `numel * element_size(target_dtype)` because upstream moves
  the wrapper and then materializes a floating CUDA weight;
- bias: its actual CUDA byte size, or its expected target representation if a
  cast is added;
- metadata/event overhead is ignored.

Always admit the current layer even if it alone exceeds the target. Stop adding
future layers before the next admission would exceed 2 GiB, except that one
layer is admitted when the cache is empty so progress cannot stall.

This makes the per-manager invariant approximately:

```text
live retained cache tensor references <= 2 GiB, or one oversize layer
```

Byte-window accounting covers final retained CUDA weights and biases only.
Temporary quantized wrappers, dequantization outputs, cast outputs, and allocator
workspace are not included. Their effects appear in actual allocated/reserved
peaks and may cause OOM. Do not add workspace prediction or adaptive admission.

### 4. Fill the window from the forward wrapper

Immediately before `_BouncingLinearFn.apply` or `_BouncingConv2dFn.apply`, call:

```python
entry = manager.forward_prefetch.acquire(layer_manager, x.dtype)
```

`acquire` performs four crude operations:

1. Find the layer's attach-time index.
2. If the index moved backward or repeated, clear the remaining cache and start
   again at this layer. This is the only recompute/repetition behavior.
3. Drop unconsumed cached entries with smaller indices, then enqueue the current
   and later layers in attach order until the live-byte target is reached.
4. Make the compute stream wait on the current entry's ready event and return
   the non-Tensor entry.

Enqueue each entry on upstream's existing forward transfer stream with
`non_blocking=True`, explicitly outside ordinary autograd recording:

```python
with torch.no_grad(), torch.cuda.stream(transfer_stream):
    ...
```

Factor materialization logic for float, quantized, linear, and conv weights out
of the existing custom autograd functions and reuse it. There must not be two
different dequant/cast implementations. The first layer pays cold-start transfer
latency; later layers may overlap compute with already queued transfers.

### 5. Consume safely without changing backward

Choose the non-Tensor cache-entry signature rather than leaving the ownership
model open-ended. The linear call is conceptually:

```python
_BouncingLinearFn.apply(
    x,
    weight_cpu,
    bias_cpu,
    prefetched_entry,
    device,
)
```

Do the equivalent for conv. The original CPU weight and bias remain Tensor
inputs and are saved for backward. The custom `forward` reads CUDA weight/bias
from the non-Tensor entry. Update each `backward` return tuple for the added
non-Tensor argument while leaving `_stage_backward_weight`, grad-input math, and
D2H gradient staging unchanged.

Before dropping the current entry, protect every CUDA tensor it owns on the
consuming compute stream:

```python
compute_stream = torch.cuda.current_stream(device)
w_gpu.record_stream(compute_stream)
if b_gpu is not None:
    b_gpu.record_stream(compute_stream)
```

Then remove the current entry and its logical live-byte count. This
`record_stream` rule is mandatory: the cache has changed tensor ownership and
must not let allocator reuse race the enqueued linear/conv operation. Do not
retain GPU materializations in `ctx`; backward continues to re-fetch the CPU
master exactly as upstream does.

### 6. Clear lifecycle state without pretending to cancel work

Drop cached entry ownership and logical byte accounting:

- after the root managed module's forward returns, using one root forward hook;
- on attach failure;
- in `MemoryManager.detach` before `_DEVICE_STATE` is cleared;
- when execution rewinds to an earlier or repeated attach index;
- after an exception, using `always_call=True` for the root hook when supported.

Store and remove the root hook handle during detach. A branch that exits before
the final predicted leaf must not retain Python ownership of its unused future
window through ordinary backward.

Cleanup does not cancel CUDA operations already queued on the transfer stream,
drain that stream, immediately reduce allocator-reserved memory, or immediately
lower driver-local usage. Do not add synchronization or transfer cancellation to
make cleanup look instantaneous.

Checkpoint recomputation receives no special handling. A repeated/lower index
uses the same crude reset and may leave newly speculative entries after backward
until the next rewind or detach. Report that residual state; do not add backward
phase detection or checkpoint hooks.

## Expected File Changes

Keep the comparator patch small:

- `toolkit/memory_management/manager.py`
  - own the ordered layer list and cache;
  - register attached layer managers;
  - install/remove the root cleanup hook;
  - clear cache during detach.
- `toolkit/memory_management/manager_modules.py`
  - factor shared forward materialization helpers;
  - add `_NaiveForwardPrefetch` and its entry type;
  - acquire a prefetched entry from linear/conv `_mm_forward`;
  - consume it in the forward autograd functions;
  - leave backward logic unchanged.
- `tests/test_naive_forward_prefetch.py`
  - focused CPU/fake-CUDA state tests plus CUDA tests guarded with `skipUnless`.
- `scripts/bench_memory_manager_prefetch.py`
  - common synthetic benchmark and machine-readable result output.

Do not change config classes, the UI, model integrations, the training loop, or
production performance logging for the comparator.

## Diagnostics Needed for the Benchmark

Add counters to the experimental cache, returned by `stats()` rather than
printed on every layer:

```text
prefetch_enqueues
future_prefetch_enqueues
prefetch_bytes_enqueued
prefetch_hits
current_layer_inline_enqueues
unused_entries_discarded
unused_bytes_discarded
rewind_resets
dtype_resets
peak_live_prefetch_bytes
entries_discarded_at_cleanup
bytes_discarded_at_cleanup
cache_entries_after_forward
cache_entries_after_backward
```

Definitions:

- `prefetch_hit`: the current layer already has a dtype-compatible entry,
  whether or not its ready event has completed;
- `current_layer_inline_enqueue`: the current layer was missing and its own
  wrapper had to enqueue it;
- `future_prefetch_enqueue`: an entry created for a later attach index;
- `dtype_reset`: a cache generation was discarded because the requested
  materialization dtype changed;
- cleanup discard counts describe state before clearing; the resulting cache
  count must be asserted as zero separately.

The benchmark should emit machine-readable JSON containing:

```text
arm name and git SHA
GPU name, total VRAM, driver, PyTorch, and CUDA versions
GPU power limit, clock/persistence settings where available, and start temperature
Windows/WDDM mode, host RAM, and page-file configuration
all Python, PyTorch, CUDA, and NumPy seeds
workload parameters and config checksum
configured offload percentage
managed/resident layer counts and source/materialized byte totals
hash of the managed-layer identity list
manager attach/setup time
first cold forward/backward or sample time
warmup/measured iteration counts
steady-state full-window time and throughput
diagnostic forward and backward compute-stream event times
median, p10, p90, and standard deviation
warmup peak and measured peak allocated/reserved bytes
prefetch counters when present
output/loss correctness telemetry
```

Do not assume a particular power limit or clock policy. Record the actual values
and keep them constant across arms.

## Focused Verification

### Unit tests

1. **Attach order** - managed linear and conv layers register once in upstream
   attach order; ignored/resident layers do not register.
2. **Two-GiB fill** - varied fake sizes fill through the last layer that fits,
   admit one oversize current layer, and otherwise respect the per-manager
   live-reference invariant.
3. **Cache hit** - a future layer enqueued by layer `i` is consumed by layer
   `i+1` without another materialization.
4. **Forward skip** - jumping over a branch discards skipped entries and fills
   from the observed layer.
5. **Rewind/recompute** - a lower or repeated index clears stale entries and
   restarts without checkpoint-specific behavior.
6. **Dtype mismatch** - a BF16 entry is not reused for FP16/FP32 and increments
   `dtype_resets`.
7. **Cleanup** - normal return, exception, and detach record discarded entries,
   leave zero owned entries, and remove the root hook.
8. **Partial offload** - only actually managed layers consume the byte window.
9. **Backward preservation** - gradients/output match stock and the existing
   backward ring is still used.
10. **Quantized accounting** - the window counts retained dequantized target
    bytes, not compressed CPU bytes or temporary workspace.
11. **Per-manager semantics** - two managers may each fill an independent
    window; no global 2 GiB claim is made.
12. **Trainable floating parameter** - stock and naive output, weight gradient,
    bias gradient, and input gradient match; speculative materializations have no
    unintended `grad_fn`.
13. **Residual recompute state** - exercise repeated forward during backward and
    report `cache_entries_after_backward` without adding a special repair.

### CUDA correctness smoke

On a small deterministic model:

- compare stock and naive forward output and input gradients;
- cover frozen and trainable ordinary floating weights;
- cover `Linear`, `Conv2d`, and one supported quantized layout if available;
- run enough iterations for refill, branch skip, and recompute;
- assert logical cleanup reaches zero where cleanup is specified;
- stress lifetime by dropping a consumed entry, allocating substantial
  same-sized CUDA tensors, synchronizing, and verifying output correctness.

The lifetime stress is intended to catch a missing `record_stream`. Use
dtype-appropriate tolerances. The naive arm introduces no new math path, so
differences beyond ordinary backend nondeterminism are a bug.

## Benchmark Protocol

### Tier 1: synthetic streaming benchmark

Run the same `scripts/bench_memory_manager_prefetch.py` in all three worktrees.
Construct a deterministic sequence of frozen managed layers with mixed sizes and
more than 4 GiB of total source weights so the 2 GiB window repeatedly slides.
Keep activation shapes representative enough for useful compute/transfer
overlap, and emit the exact shape list.

Run two profiles:

- inference: forward only, starting upstream offload at `0.50`;
- training: forward and backward, starting upstream offload at `0.70`.

The local arm uses its autosizer. Record actual managed/resident bytes for every
arm; configured percentage alone is not a sufficient comparison.

At process start, seed every relevant generator before model construction and
upstream's random partial-offload selection:

```python
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
numpy.random.seed(seed)  # when NumPy is used
```

For each arm and repetition:

1. Start a fresh Python process and record machine controls.
2. Time manager attach/setup separately.
3. Time the first cold complete step/sample separately.
4. Run 5 untimed warmup iterations and synchronize.
5. Record warmup peaks, then reset peak allocated/reserved statistics.
6. Run at least 30 complete measured iterations without synchronizing between
   forward and backward.
7. Synchronize after the full measured window, then read time and memory peaks.

The authoritative throughput metric is CPU wall time around the fully
synchronized multi-step window:

```text
CUDA synchronize
start timer
run N complete steps
CUDA synchronize
stop timer
```

Forward/backward splits are diagnostic CUDA events on the compute stream. Record
`forward_start`, `forward_end`, and `backward_end`, but synchronize only after
the complete step/window. These event splits may not include speculative work
still queued on the transfer stream and must not replace full-step throughput.

Repeat each arm in this balanced order:

```text
repetition 1: stock -> naive -> local
repetition 2: naive -> local -> stock
repetition 3: local -> stock -> naive
```

Report setup, cold start, steady-state full-window throughput, diagnostic
forward/backward split, measured peaks, and residual cache state separately.

### Tier 2: real Krea2 training and inference

Prepare matched training and inference job configurations. Use the same dataset
under `C:\GenAI\ai-toolkit\datasets` and distinct output directories under
`C:\GenAI\ai-toolkit\output`. The user launches real jobs through the web
interface unless explicitly asking the implementation agent to start them.

Hold constant:

- model/checkpoint and quantization backend;
- LoRA type/rank and optimizer;
- dataset, captions, all RNG seeds, resolution, batch size, accumulation, and
  checkpointing mode;
- precision, attention backend, PyTorch/CUDA environment, and machine controls;
- sampling disabled during the measured training interval;
- saving disabled or scheduled after the measured interval;
- all non-memory-manager experimental math paths.

Use these upstream starting assumptions:

```text
training:  layer_offloading_transformer_percent = 0.70
inference: layer_offloading_transformer_percent = 0.50
```

Hold `layer_offloading_text_encoder_percent` constant across arms and record it.
Seed Python immediately before attach and record the managed-layer-list hash.

### Upstream best-case offload sweep

The starting percentages are not forced final settings. Tune `upstream-stock`
and `upstream-naive-2g` independently because the naive arm's 2 GiB window
changes its VRAM requirement.

Declare this safety predicate before looking at performance:

```text
no OOM or allocation retry
minimum driver-local free VRAM >= 0.5 GiB
no sustained growth into WDDM shared memory
no failed sample or training step
```

For each phase and upstream arm:

1. Run a short preflight at the 50% inference or 70% training starting point.
2. If unsafe, increase offload in 5 percentage-point steps until safe.
3. If safe, decrease offload in 5 percentage-point steps until the first unsafe
   candidate establishes the residency boundary.
4. Time the boundary-safe and next-safer candidates with the same preflight;
   choose the faster stable candidate.
5. Freeze that percentage before measured repetitions.

Preserve all preflight results, including OOMs. At a shared percentage, stock
and naive use the same RNG seed and managed-layer-list hash. Keep one paired
stock-versus-naive calibration at the naive arm's selected percentage so patch
effect can be separated from independent best-case tuning.

`local-current` uses its ordinary smart manager, prefetcher, and autosizing and
must satisfy the same safety predicate. Disable native FP8 forward/backward and
sampling, in-graph streaming, compile-streamed offload, block-stream-only mode,
and checkpoint autotuning so arithmetic/compile do not confound the comparison.

### Measured training runs

Use one fixed warmup of 20 complete training steps for every arm, then at least
30 measured steps; increase the measured count equally if variance is high. Do
not dynamically end warmup per arm. Repeat three fresh processes per arm in the
same balanced order as the synthetic benchmark. Repeat at a 768-class bucket if
all arms fit after the primary 512-class run.

Preserve correctness telemetry:

```text
initial and final measured loss
sampled loss sequence or checksum
NaN/Inf count
optimizer-step count
```

### Measured inference runs

Use one fixed prompt, seed, scheduler, resolution, and denoise-step count. Run
two untimed warmup samples followed by at least five measured samples per fresh
process, with three processes per arm in balanced order. Measure total sample
latency and synchronized denoiser-step latency. The local arm may use smart
sampling/autosizing, but native FP8 sampling remains disabled.

### Common result row

```text
phase | arm | run | resolution | configured offload | actual offloaded bytes |
safety pass/fail | setup/cold time | median step s | p90 step s | images/s |
peak allocated/reserved | minimum driver-local free | peak shared-memory use |
OOM/retry count | prefetch hit/miss | output/loss correctness
```

Use `scripts/digest_perf_log.py` for the local arm. Upstream must capture
equivalent synchronized step boundaries without porting the local manager or
profiler. If ordinary output is insufficient, add one byte-for-byte identical
benchmark-only timing shim to all worktrees as a commit separate from the naive
patch.

## Fairness Rules

- Every result names exact code SHA, config checksum, seeds, and machine state.
- Stock and naive upstream differ only by the naive-prefetch commit.
- At a shared percentage, stock and naive use the same seeded managed-layer
  identity list.
- The 50% inference and 70% training values are sweep starting points. Each
  upstream arm freezes its fastest stable percentage before measured runs.
- Preserve the full tuning sweep, including unsafe candidates.
- The local arm reports autosized resident/offloaded bytes and must satisfy the
  same predeclared VRAM/WDDM safety predicate.
- Any timing shim is byte-for-byte identical across arms.
- Use fresh processes and the declared balanced arm order.
- Keep GPU power limit, clock policy, other GPU applications, host-memory state,
  and page-file policy constant; report deviations.
- Compare medians with medians and distributions with distributions; do not
  compare one arm's best run with another arm's median.
- Report failed/OOM runs rather than dropping them.
- Report allocated and reserved VRAM plus driver-local/shared memory. The cache
  controls retained references, not allocator or temporary peaks.
- Primary results keep the fixed 2 GiB per-manager target even if another window
  is faster.
- Preserve raw JSON, configs, loss/output telemetry, and job logs.

## Interpretation Rules

The benchmark supports conclusions about:

- naive versus stock throughput;
- complete local-system throughput versus the best stable naive arm;
- fit/OOM/WDDM behavior under the declared safety predicate;
- whether local benefits appear only at larger resolutions or tighter memory.

It does not isolate the value of trace learning, the bounce pool, WDDM control,
adaptive residency, pin budgeting, or any other individual local mechanism.
Final reporting must describe this as a whole-system comparison against stock
Ostris plus a fixed attach-order 2 GiB forward lookahead.

## Acceptance Criteria

Implementation is complete when:

- the naive patch records one pinned upstream SHA;
- mutable cache state is instance-owned and the 2 GiB target is documented as
  per manager and retained-final-tensor accounting only;
- speculative materialization runs under `torch.no_grad()`;
- custom autograd receives a non-Tensor entry while CPU masters remain Tensor
  inputs and backward remains stock;
- every consumed CUDA tensor is recorded on the consuming stream before cache
  ownership is dropped;
- normal return, exception, rewind, and detach clean logical cache ownership;
- cleanup does not add transfer cancellation or synchronization;
- focused unit and CUDA correctness/lifetime tests pass;
- no product config/UI or adaptive comparator policy is added.

The benchmark is complete when:

- all three arms have at least three synthetic runs;
- all feasible arms have three real training and three real inference runs;
- peaks are reset after warmup and full synchronized step-window throughput is
  the authoritative performance metric;
- setup, cold start, steady state, diagnostic phase splits, actual memory,
  WDDM/shared memory, failures, and residual cache state are preserved;
- output/gradient tests and real-job loss/NaN telemetry show comparable math;
- upstream sweeps and frozen best-stable percentages are preserved;
- ticket `267470b` contains the final table and raw-artifact links.

There is no required performance winner. A tie, regression, or OOM is valid.
The decision is whether the complete local memory-management system provides
useful throughput or memory-safety gains beyond stock Ostris plus a fixed
attach-order 2 GiB forward lookahead, without attributing the difference to one
local mechanism.

## Implementation Order

1. Create clean pinned upstream and local benchmark worktrees.
2. Add the common synthetic benchmark and validate stock-upstream measurements.
3. Add attach-order registration and pure byte-window tests.
4. Factor upstream forward materialization without changing behavior.
5. Add the naive cache, entry events, acquire/consume, and lifecycle cleanup.
6. Run unit, CPU fallback, and CUDA correctness checks.
7. Run the three-arm synthetic benchmark and inspect the expected
   forward-only speedup signature.
8. Prepare matched real-job configs and benchmark-only timing shim if needed.
9. Hand the real jobs to the user for web-UI launch.
10. Digest results, update ticket `267470b`, and decide whether the local
    manager's added mechanisms are justified by the measured gap.
