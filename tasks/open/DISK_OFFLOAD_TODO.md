# Future TODO: disk-backed offload (RAM-optimized streaming)

> **git-bug:** `570332b` (open) — design RAM-optimized streaming prototype.
> Status lives in the ticket; this file is the plan.

## Motivation

Today the offload split is:

- **Resident layers** live on GPU only — no CPU copy, no host RAM, not pinned.
- **Non-resident (streamed) layers** live in pinned CPU RAM; the bounce pool /
  ring stream them CPU→GPU per fetch.

So residents already cost no host RAM (the `pinned_weight_gib` auto-budget is a
*cap* sized to the whole model for demotion headroom, not an allocation). The
remaining RAM limit is the **non-resident set**: today it must fit in host RAM.
On a RAM-constrained machine you may want to offload more weight than RAM holds.

The RAM-optimized variant: hold the non-resident weights on **disk** (memory-
mapped / on-demand read) instead of RAM, and stream disk → pinned staging buffer
→ GPU. Host RAM then only holds the bounded pinned staging pool, not the whole
non-resident set.

## Is it architecturally possible? Yes.

The per-layer allocation lifecycle already exists and is exercised every step by
promote/demote:

- `demote_layer` → `_move_params_to_cpu_and_pin` **creates the entire CPU
  allocation** for a layer from scratch (materialize + pin), then installs the
  streaming forward. So "create the whole allocation on demotion" is already a
  supported, tested operation — not something new to invent.
- `promote_layer` **fully releases** that allocation (moves data to GPU, replaces
  the Parameter, and — as of the pin-budget fix — returns the pinned bytes).
- The bounce pool already sources from an arbitrary CPU tensor and stages it to a
  pinned buffer off-thread; the source does not have to be a normal RAM tensor.

So a disk-backed source slots into the existing seams: swap "the layer's CPU
Parameter" for "a disk-backed / regenerated allocation," reusing the same
demote(create)/promote(release) and bounce(stage)/ring(H2D) paths.

## What would need building

1. **Backing store.** A per-layer disk home for non-resident weights — e.g. an
   mmap of the quantized `qdata` + scales (safetensors or a raw blob), written
   once at attach. Quantized wrappers must round-trip: reuse the leaf
   flatten/rebuild helpers (`_flatten_leaves` / `_rebuild_from_leaves`, added for
   block staging) to serialize/reconstruct qdata + scale leaves.
2. **Fetch source = disk.** The bounce worker reads the layer's bytes from the
   mmap into its pinned staging buffer instead of from a resident CPU tensor.
   mmap + `copy_` into pinned already overlaps off-thread; the OS page cache
   gives a natural RAM tier. Prefetch lookahead must cover disk latency (deeper
   than the current RAM-source lookahead).
3. **Demotion creates the allocation from the store, not from GPU-only.** When a
   resident layer is demoted under this mode it has no disk home yet — write it
   out (or keep GPU→pinned as today and lazily page to disk). Promotion drops the
   RAM/pinned copy and keeps only the disk home.
4. **RAM budget = pinned staging pool only.** Decouple `pinned_weight_gib` (whole
   non-resident set) from a new "disk-backed" mode where the RAM cost is just the
   bounce pool budget + ring. Config gate, e.g. `layer_offloading_disk_backed`.

## Gotchas / open questions

- **Disk bandwidth vs step time.** NVMe ~3–7 GB/s read; the live run streams
  ~65 GB/step (10.8 GB × ~6 fetches). Disk-only would be transfer-bound unless
  the OS page cache keeps the hot set resident — i.e. this helps when the
  non-resident set only *slightly* exceeds RAM, not when it's 3× RAM.
- **Windows page cache + pinned interaction** — mmap read into pinned is two
  copies; a direct pinned read from disk would be better if feasible.
- **Quantized wrapper reconstruction from a byte blob** is the fragile part —
  lean on the block-staging leaf helpers, and gate behind a correctness test
  (round-trip a quantized weight through disk == original), same discipline as
  the block-stream gradient-parity test.

## Status

Not started. Recorded so the current pinned-weight work (auto-budget sized to the
whole model + promote/demote budget accounting) stays compatible: it already
assumes allocations are created on demote and released on promote, which is the
exact seam a disk-backed source would reuse. See also [[project_pinned_weight_autobudget]].
