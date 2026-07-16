# Disk-backed block homes for the arena runtime

> **git-bug:** `570332b` (open). Status lives in the ticket; this file is the
> plan. This is a full redesign (2026-07-16) of the earlier legacy-MemoryManager
> disk-offload sketch, rebuilt on the generic arena runtime
> (`toolkit/memory_management/arena_offload/`, `canonical_arena.py`,
> `residency.py`, `immutable_runtime.py`) and on residency-aware pinning
> (commit `7b5f585e`).

## Motivation

The canonical arena keeps a populated host flat for **every** block for the
whole process lifetime, so host RAM cost = the entire (quantized) model.
Residency-aware pinning already splits those flats into two populations:

- **Hot** (pinned): the streamed set — read H2D every step — plus a reserve of
  the next expected demotions (`DEFAULT_DEMOTION_PIN_RESERVE_BLOCKS = 2`, in
  the deterministic order from `ordered_demotion_block_keys`).
- **Cold** (unpinned): resident blocks' flats. Because the arena is immutable,
  these bytes are pure redundancy — they are only read again when the block is
  demoted back into the streamed set, and demotion involves no D2H copy.

The cold population's bytes already live on disk (checkpoint or quantized
cache). Keeping a RAM copy of them is a policy choice, not a requirement.

**This plan: blocks outside the pinned window drop their host flats entirely
and become disk-only; the flat is rebuilt from disk when the block re-enters
the window.** Per-step behavior is untouched — all transitions happen at
residency-plan publishes (phase boundaries), where the pin reconcile already
runs.

### Explicitly rejected: streaming the streamed set from disk

Reading the *hot* set from disk per step stays bandwidth-doomed: a Krea2-scale
run streams ~65 GB/step against NVMe's 3-7 GB/s sequential read. Nothing in
the new pinning changes that math. Disk-backed homes are for the cold set
only. (If the hot set must exceed RAM, that is a different, transfer-bound
feature; do not grow this one into it.)

### Honest payoff scoping

Savings = resident-set bytes. On the 12 GB dev box most training blocks are
streamed, so training-phase savings are modest; the big local win is
**high-residency phases** (sampling under `inference_resident`, small models,
larger cards), where most flats idle. The train->sample->train boundary then
pays one bulk disk reload (seconds at NVMe rates for a multi-GiB set) — a
phase-boundary cost, consistent with where the runtime already pays
reconciliation costs.

## Design

### Block home states

Each canonical block is in exactly one state:

1. `materialized+pinned` — flat allocated, populated, registered (today's hot
   set: streamed ∪ demotion reserve).
2. `materialized` — flat allocated + populated, unregistered (transitional
   only under disk-backed mode; the steady state for cold blocks today).
3. `disk-only` — no host flat; the block's bytes exist only in its disk home.

Target state after every plan publish under disk-backed mode:
**materialized+pinned for (post-plan streamed set) ∪ (demotion reserve) ∪
(blocks whose sidecars this publish must build — promotion sources);
disk-only for everything else.** Promotion sources may be dematerialized
again once their copies settle (`ResidencyState.synchronize_copies`), exactly
mirroring how `_trim_plan_pins` already handles their pins.

### Disk home = an existing file, verified at attach; no new store format

Resolution order, decided once at attach:

1. **Quantized runs: the quantized-weights cache** (e.g. Krea2's
   `_save_quantized_transformer_cache` safetensors). Its tensor keys are the
   same managed source keys the arena build consumes, so provenance is exact.
   Disk-backed + quantize **requires** the cache to be enabled and written —
   requantize-on-reload is forbidden (slow, and a silent numeric variable).
2. **Unquantized runs: the source checkpoint itself**, iff verification
   passes: open the safetensors header (`safe_open` / a header parse like
   krea2's `_read_safetensors_header`) and confirm every managed source key in
   `build.state_schema` is present with matching nbytes/dtype. Loaders that
   remap key names between file and model fail this check by construction.
3. **Otherwise: fail fast** with an error naming the missing keys and the
   remedy (enable the quantized cache). No silent fallback to keeping RAM
   copies. A model-agnostic "arena image" dump (write the flats themselves to
   disk at attach) is a possible later extension if a real model needs it —
   it is not part of v1.

Reload of block K = ranged reads of K's source keys from the home file into
the freshly allocated flat at the offsets `state_schema` + the layout already
define — the same shape of work `populate_block_from_model` /
`populate_from_state_dict_consuming` do at load, re-runnable per block. The OS
page cache provides a free RAM hot tier for recently used homes.

### What must be built

1. **Retained provenance.** Today the load session consumes sources
   destructively and drops file identity. Record, per managed block: home file
   path, per-leaf (source key -> flat offset, nbytes, dtype), plus a file
   fingerprint (size + mtime or header hash) checked on every reload so a
   swapped file fails loud.
2. **Home lifecycle ops on `CanonicalArena`** alongside `pin_block` /
   `unpin_block`: `dematerialize_block` (assert unpinned + copies settled,
   free the flat) and `rematerialize_block` (allocate, ranged-read populate,
   verify, then pin — populate-before-register is 20x cheaper). Per-block
   lazy allocation already exists in the build path ("each block is allocated
   only when its turn begins"); reuse it.
3. **Rebinding.** Transfer-plan ranges are offset-relative and survive a new
   flat, but block ABIs capture `typed_view(pack.host_flat, ...)` host views
   and `immutable_signature()` captures `data_ptr`. Rematerialization must go
   through one explicit runtime API that rebuilds that block's host-view
   bindings and refreshes the stored signature — at plan-publish boundaries
   only. `_assert_arena_stable` keeps policing every *unexplained* mutation;
   a home transition outside the API stays a loud failure.
4. **Policy integration.** Extend the `_prepare_plan_pins` / `_trim_plan_pins`
   reconcile in `immutable_runtime.py`: prepare = rematerialize+pin everything
   entering the window (before any sidecar is removed); trim = after copies
   settle, unpin **and dematerialize** everything that left it. The window
   computation is `pin_requirements_for_plan` plus promotion sources — no new
   ordering logic.
5. **Config gate:** `layer_offloading_disk_backed` (bool, default `False`) on
   `ModelConfig`, plus the UI schema entry. Off = today's behavior, bit for
   bit.

### Option B (recorded, not chosen): fixed slot pool

Transformer trunks are usually layout-homogeneous, so a pool of
`window_size` identical pre-pinned flats could host whichever blocks are hot,
avoiding alloc/free/pin churn and keeping DXGI commit constant. Rejected for
v1: it still requires per-swap view rebinding, breaks on heterogeneous blocks,
and the churn it avoids is small at phase-boundary frequency (registration of
populated pages is ~150 GiB/s). Revisit only if reconcile time measurably
hurts.

## Gotchas

- **RAM is only saved by freeing the flat.** Unpinned-but-allocated memory
  still holds commit charge; letting the OS page it out is the pagefile
  behaving as an accidental, unaccounted disk tier. Free it or keep it.
- **Pin ledger accounting** must stay explicit-release through
  `pin_manager` for every transition (no finalizers, per its contract).
- **A residency retreat that demotes past the reserve** puts disk reads on
  the reconcile path. That is acceptable (it is a pressure event, not steady
  state) but must be visible in the perf log, not silent.
- **Quantized wrapper reconstruction stays the fragile part**: the reload
  writes raw bytes into the flat and the existing leaf views reinterpret
  them; the round-trip test below is the gate, same discipline as the
  block-stream gradient-parity test.

## Acceptance

- CPU unit tests (`tests/`): provenance verification (header/key/nbytes
  mismatch -> loud failure); dematerialize/rematerialize round-trip is
  bitwise-identical on a synthetic quantized arena; a home transition outside
  the explicit API trips `_assert_arena_stable`; pin ledger balances across a
  full window rotation.
- GPU smoke: `scripts/smoke_krea2_train_cuda.py` and
  `scripts/smoke_krea2_inference_cuda.py` with the flag on produce
  bitwise-identical step outputs vs flag off, with measured host-RAM
  footprint reduced by ~the resident-set bytes and zero per-step disk reads
  in steady state.
- Flag off: no behavior change (upstream-PR separability,
  `docs/decisions/UPSTREAM_PR_PLAN.md`).
