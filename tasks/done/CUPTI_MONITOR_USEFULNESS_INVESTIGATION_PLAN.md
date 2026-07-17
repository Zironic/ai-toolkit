# CUPTI monitor usefulness investigation

> **git-bug:** `2d6b39c` (closed) - the evidence and final decision are recorded
> in the ticket.

> **Type:** bounded investigation plan. The first probe is a direct opt-in
> backend in the existing Krea2 training smoke; this is not an implementation
> plan for the larger tracing roadmap.

## Conclusion

**Reject for Toolkit's current native-Windows environment.** PyTorch 2.13 has
Windows wheels, but its experimental monitor implementation explicitly requires
NVIDIA's `cupti-python` package. As of 2026-07-16, NVIDIA publishes that package
only as manylinux x86_64/aarch64 wheels, with no Windows wheel or source
distribution. The current project environment is also Torch 2.12 and contains
neither `cupti` nor `torch.profiler._cupti_monitor`.

An opt-in training-smoke backend was prototyped far enough to prove that it
could fail before contention checks or Krea2 loading. It was not retained:
without an official native-Windows dependency path, it would be unexecutable
code in the environment this fork is intended to diagnose. WSL2/Linux would not
measure native WDDM behavior.

Reopen the investigation if NVIDIA publishes supported Windows
`cupti-python` packaging or PyTorch removes that runtime dependency.

## Decision outcome

Decide whether PyTorch's experimental CUPTI monitor should become a local,
opt-in timing evidence backend for Toolkit CUDA smokes.

The investigation is complete when it records one of these decisions:

1. **Adopt:** the monitor is stable on this Windows machine and explains a real
   transfer/compute scheduling difference that existing counters cannot.
2. **Keep stock:** a low-detail stock profiler provides equivalent evidence at
   equivalent cost, so a new backend has no demonstrated value.
3. **Defer:** the monitor is promising but is unavailable in Toolkit's supported
   Torch environment or is too experimental to carry now.
4. **Reject:** traces are unreliable, too perturbing, or cannot be correlated
   with Toolkit's semantic diagnostics.

An Adopt decision creates a separate implementation plan and ticket. It does
not authorize the multi-phase runtime instrumentation, digest tooling, smoke
rollout, or controller changes proposed in the source roadmap.

## Verified starting point

- The project venv currently reports `torch 2.12.0+cu132`. The CUPTI monitor was
  announced as an experimental, API-unstable PyTorch 2.13 feature. The installed
  2.12 build exposes `_ExperimentalConfig.custom_profiler_config`, but that
  alone does not prove that the monitor backend or its required bindings are
  present.
- `scripts/smoke_krea2_train_cuda.py` already traces selected tail steps with
  CPU and CUDA activities, exports a Chrome trace, annotates forward/loss/
  backward/gradient-statistics/optimizer phases, reports traced-versus-untraced
  step time, and destroys Kineto before runtime teardown.
- The current smoke enables `record_shapes=True` and `with_flops=True`. A fair
  backend comparison must disable both for the stock and monitor timeline arms;
  otherwise operator-detail overhead is confounded with backend overhead.
- `toolkit/memory_management/arena_offload/transfer.py` already records fetches,
  bytes, copies, CUDA-event H2D duration, host-side wait-call time, and depth
  pressure. Its own contract states that `wait_ms` does not establish whether
  the GPU compute stream was idle.
- Heavy profiler tooling is local-only under
  `docs/decisions/UPSTREAM_PR_PLAN.md`. Existing structured JSON remains the
  authority for policy, residency, memory safety, WDDM/NVML state, and
  correctness.

Official references:

- [PyTorch 2.13 CUPTI monitor announcement](https://pytorch.org/blog/pytorch-2-13-release-blog/)
- [PyTorch profiler API and overhead notes](https://docs.pytorch.org/docs/stable/profiler.html)

## Requirements and non-goals

The spike must answer four questions:

1. Does the monitor work repeatedly on this Windows/CUDA stack and exit cleanly?
2. Is its observer effect lower than, or otherwise preferable to, an
   apples-to-apples low-detail stock trace?
3. Can the exported trace be reconciled with existing Toolkit phase and
   transfer counters?
4. Does it reveal a decision-relevant fact that the counters do not, especially
   whether H2D work is hidden under compute or exposed on the critical path?

Non-goals:

- no runtime or memory-FSM dependency on profiler availability;
- no profiler scopes inside compiled block math;
- no per-block production instrumentation, general trace digest, shared trace
  controller, UI/config surface, or broader smoke rollout;
- no project-venv upgrade solely for this experiment;
- no full test suite or real dataset training job.

## Investigation sequence

### Gate 0: establish a supported test environment

1. Record the project Torch, CUDA runtime, driver, GPU, and Kineto/CUPTI
   capabilities on the ticket.
2. Verify the exact PyTorch 2.13 monitor configuration from the tagged PyTorch
   source or tests; do not infer the hidden configuration string from the
   presence of `custom_profiler_config`.
3. Determine whether the probe can run in an already available 2.13 environment
   or an isolated disposable environment. Do not replace or mutate the project
   venv. Installing another Torch build requires a separate explicit action.

**Stop condition:** if no safe 2.13 environment is available, decide Defer as
"not usable in the current Toolkit runtime" and record what future version
change would reopen the investigation.

### Gate 1: hook the monitor directly to the training smoke

Use `scripts/smoke_krea2_train_cuda.py` as the first probe because it already
owns the trace window, phase annotations, observer-overhead calculation, Chrome
export, and explicit Windows profiler destruction.

Add:

- `--trace-backend {stock,cupti-monitor}` with `stock` as the compatibility
  default;
- `--trace-detail {auto,timeline,operators}`, where `auto` preserves the stock
  operator report and selects the low-detail timeline for the monitor;
- the upstream PyTorch monitor configuration
  `{"backend":"cupti_monitor"}` through `_ExperimentalConfig`;
- an early capability check for both PyTorch's monitor integration and the
  optional `cupti` Python package, before contention checks or model loading;
- backend and resolved detail in the final trace summary.

Do not use the stock `key_averages()` path to rank monitor GPU events. The
monitor merges its GPU activity during Chrome export, so its kernel evidence
must be read from that trace. Keep the existing profiler stop, export, destroy,
and garbage-collection ordering.

**Stop condition:** if the backend cannot load on the current Windows/Torch
environment, record the exact capability failure. Do not load Krea2 and do not
upgrade the project venv merely to get past the gate.

### Gate 2: read one matched training trace

When a compatible environment exists, use `smoke-direct-to-arena`, a fixed
512px bucket, identical commit/cache/input settings, warmup excluded, and a
two-step tail trace. A full-model smoke needs the repository's normal user grant
(`.agent/allow-gpu-smokes.md`) or explicit approval.

Run the shortest comparison first:

1. low-detail stock trace;
2. low-detail CUPTI-monitor trace.

Each run supplies untraced steady steps for an in-process observer-cost
comparison. Inspect the exported monitor trace for the event categories PyTorch
itself requires: CPU operation, user annotation, CUDA runtime, kernel, and GPU
memcpy. Compare phase durations with the smoke's CUDA events and compare H2D
copy/duration structure with Toolkit's existing transfer report.

Only if the monitor trace is valid and appears to add useful overlap/idle
evidence, run one `--prefetch-depth 1` monitor contrast against the default
depth 2. That contrast is the value test: can the trace explain more or less
overlap, exposed transfer delay, ready slack, GPU idle time, or no critical-path
change because transfers were already hidden?

If the direct trace cannot answer that question beyond `fetches`, `copies`,
`h2d_ms`, and host `wait_ms`, the larger tracing roadmap is not justified. A
standalone synthetic probe is a fallback only when needed to isolate a smoke
failure; it is no longer a prerequisite.

## Decision rubric

Record backend viability, timing trust, and diagnostic value separately:

- **Viability:** three clean probe traces, required event categories present,
  bounded trace files, and reliable Windows teardown.
- **Timing trust:** compare profiler overhead with the distribution of untraced
  steady steps. If the observer effect exceeds that variation, trace timings
  are structure-only evidence even if the backend remains useful.
- **Counter reconciliation:** copy counts should reconcile after applying the
  same trace window and event filters; duration differences must be explained
  by their event boundaries.
- **Added value:** the trace must answer the controlled prefetch question with a
  mechanism that `fetches`, `copies`, `h2d_ms`, and host `wait_ms` alone do not
  establish.
- **Carrying cost:** record Torch-version coupling, hidden/API-unstable setup,
  parser complexity, trace/export cost, and failure behavior.

Prefer Keep stock when it gives the same answer with less version coupling.
Prefer Defer over changing Toolkit's supported Torch version solely to acquire
an experimental diagnostic. Adopt only as opt-in, local diagnostic tooling;
controller inputs remain unchanged.

## Durable evidence and closeout

Keep commands, environment identity, arm results, trace summaries, failures,
and the final decision in the git-bug ticket. Temporary traces and probes stay
under `.agent/tmp/` and are removed after their summarized evidence is recorded.

At closeout:

- **Adopt:** create a smaller implementation plan/ticket for a reusable probe,
  low-detail backend selection, manifest, and the minimum proven digest.
- **Keep stock:** create only the focused stock-profiler cleanup if the
  low-detail setting itself proved beneficial.
- **Defer/Reject:** record the reopening condition or failure evidence and close
  the investigation without adding production trace machinery.
