"""Persistent pinned per-block flat-buffer arena for offloaded weights.

Ticket 534ea49: today, offloaded weights are pinned per-tensor
(cudaHostRegister) and every sampling boundary unpins/re-pins the whole
~9.9 GiB set -- seconds of page-lock kernel work per boundary. This module
pins ONCE, at attach time, into per-block flat host buffers (the same
``pack_block_host(repoint=True)`` mechanic the in-graph packs already use)
and repoints module Parameters as views into those flats. The arena hangs
off the model module itself (not the MemoryManager, which is destroyed and
rebuilt at every sampling boundary), so it survives detach/attach cycles.
Sampling boundaries then become pure ledger accounting: no unpin, no re-pin.

Only frozen weights are eligible (``requires_grad=True`` leaves fail
closed) -- the arena covers offloaded base weights, never trainable
adapters. Every arena flat is pinned under the ``"weights"`` ledger kind
(the existing weight-tier priority rules -- never evicts the bounce pool --
apply unchanged).

The in-graph arena protocol (Phase 3 Slice C)
---------------------------------------------
Nothing in this module, ``ingraph_stream``, or ``MemoryManager`` knows about
any particular model. A model opts into the arena by satisfying four points;
only (3) and (4) live in the model file, and neither contains arena logic.

1. **Freeze the base before attach.** Every offloaded base Linear must have
   ``requires_grad=False`` before ``attach_smart_training`` /
   ``inference_resident`` runs -- the arena is built inside attach and fails
   closed (``arena_trainable_leaf``) on a trainable leaf. Note the shared
   trainer's freeze (``BaseSDTrainProcess`` ``unet.requires_grad_(False)``)
   happens AFTER ``load_model()``, and ``load_model()`` is where the attach
   lives, so a model that offloads must freeze its own base first.

2. **Plumb the flag.** Pass ``use_pinned_arena`` (and, when the model's whole
   block set is streamed by a compiled trunk, ``stream_all_blocks``) into
   ``attach_smart_training`` / the ``inference_resident`` attach sites. The
   manager builds and reuses ``module._mm_weight_arena`` generically; the
   model does nothing here.

3. **Enumerate blocks.** Yield, per streamed block, a stable ``block_key: str``
   and that block's ``(name, module)`` linear entries. The key must be stable
   across attach cycles and agree with ``MemoryManager._offload_group_key``
   grouping. The shape of the model's block container is irrelevant.

4. **Enable via the shared helper.** ``enable_ingraph_sampling`` /
   ``enable_ingraph_training`` call
   ``ingraph_stream.build_or_borrow_block_packs(arena, entries_by_block, ...)``,
   which owns the borrow-or-own policy, the fail-closed reasons
   (``non_pinned_pack``, ``unsupported_quant_wrapper``, ``wrapper_pack_missing``,
   ``arena_borrow_required``) and the release-only-what-we-own cleanup. It
   returns a ``PackBuildResult`` keyed by ``block_key``; the model maps those
   back to its own indices and composes any per-block extras (LoRA,
   checkpointing) in its own block fn.

The arena itself is duck-typed by the helper (anything exposing
``try_borrow_pack(block_key, entries)``), so ``ingraph_stream`` never imports
this module -- which imports it.

The trainer seam is already generic: ``BaseSDTrainProcess`` looks up
``enable_ingraph_training`` with ``getattr`` and fails loud if absent, so a
conforming model needs no trainer changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch

from toolkit.memory_management import pin_manager
from toolkit.memory_management.ingraph_stream import (
    ArenaBorrowError,
    BlockPack,
    LeafSpec,
    _aligned_offsets,
    _flatten_leaves,
    _rebuild_from_leaves,
    leaf_view,
    pack_block_host,
    pack_block_host_from_flat,
    release_pack,
)

ARENA_KIND = "weights"

# Every live arena, held with a STRONG ref on purpose. An arena's flats are
# pinned in place with cudaHostRegister; if the arena (and its base tensors)
# are garbage-collected before ``release()`` runs, torch frees the storage
# while the pages are still registered with CUDA -- a dangling registration
# that makes the next allocation on those recycled pages raise CUDA "resource
# already mapped" (the 763bb75 collision, reproduced across tests). A weak ref
# loses that race; a strong ref guarantees the registration outlives nothing
# but an explicit release. ``release()`` discards from here, so production's
# one-per-model arena is reclaimed when it is torn down and the set does not
# grow unbounded. tests/conftest.py sweeps this after each test.
_LIVE_ARENAS: "set[PinnedWeightArena]" = set()


class ArenaLayoutError(ValueError):
    """Raised when a block's entries can't be admitted to the arena."""


def _entry_module_and_leaves(entry):
    if len(entry) != 2 or entry[1] is None:
        raise ArenaLayoutError(
            "arena_requires_module_entries: pinned-arena blocks must be built "
            "from (name, module) entries so params can be repointed"
        )
    name, module = entry
    weight = module.weight
    bias = getattr(module, "bias", None)
    return name, module, weight, bias


def _validate_entries_frozen(entries) -> None:
    for entry in entries:
        _, _, weight, bias = _entry_module_and_leaves(entry)
        if getattr(weight, "requires_grad", False):
            raise ArenaLayoutError(f"arena_trainable_leaf:{entry[0]}:weight")
        if bias is not None and getattr(bias, "requires_grad", False):
            raise ArenaLayoutError(f"arena_trainable_leaf:{entry[0]}:bias")


def _entries_total_bytes(entries) -> int:
    """Estimate a block's flat size (same alignment math pack_block_host
    uses) WITHOUT allocating -- lets build() decide pin-vs-pageable per block
    against a byte budget before committing to a pin_alloc attempt."""
    leaves = []
    for entry in entries:
        _, _, weight, bias = _entry_module_and_leaves(entry)
        w_data = weight.data if isinstance(weight, torch.nn.Parameter) else weight
        b_data = bias.data if isinstance(bias, torch.nn.Parameter) else bias
        leaves.extend(_flatten_leaves(w_data))
        if b_data is not None:
            leaves.extend(_flatten_leaves(b_data))
    _, total = _aligned_offsets(leaves)
    return total


@dataclass
class _ArenaBlock:
    pack: BlockPack
    generation: int
    entry_names: tuple[str, ...]


@dataclass
class ArenaBuildStats:
    blocks: int = 0
    pageable_blocks: int = 0
    pinned_bytes: int = 0
    pageable_bytes: int = 0


class PinnedWeightArena:
    """Owns one process's persistent pinned per-block flat buffers.

    Instances are meant to be hung off the model module
    (``module._mm_weight_arena``), not the MemoryManager -- see module
    docstring. Not thread-safe; build/invalidate/restore are expected to run
    on the same thread that drives attach/detach.
    """

    def __init__(self) -> None:
        self._blocks: dict[str, _ArenaBlock] = {}
        self._generation: dict[str, int] = {}
        # id(module) -> (block_key, entry_name)
        self._module_index: dict[int, tuple[str, str]] = {}
        _LIVE_ARENAS.add(self)

    # -- build --------------------------------------------------------

    def build(
        self, entries_by_block: dict, *, kind: str = ARENA_KIND,
        budget_bytes: Optional[int] = None,
    ) -> ArenaBuildStats:
        """Build (or rebuild) arena blocks.

        ``entries_by_block`` maps ``block_key -> iterable of (name, module)``
        entries (the same shape ``pack_block_host`` accepts, but module is
        required here since repointing is the whole point). Rebuilding a
        block key releases the previous pack's own grant (if any) and bumps
        that block's generation counter -- any module still tagged with the
        old generation is stale (see ``is_current``).

        ``budget_bytes`` (Phase 2 Slice B) caps how much of THIS call's own
        commitment may be pinned: blocks are built in ``entries_by_block``
        iteration order, and once the running total would exceed the budget,
        remaining blocks get pageable flats instead (still repointed --
        uniform layout, so bounce-pool/pack-borrow keying is unaffected).
        ``None`` means unlimited (only the OS/headroom backstop in
        ``pin_alloc(required=False)`` applies). Bytes already committed by an
        earlier ``build()`` call are NOT part of this budget -- callers
        (``_build_pinned_arena``) pass only the delta still available --
        EXCEPT that rebuilding an existing pinned block releases its old
        flat, so those bytes are credited back to the running budget here
        (a whole-block rebuild that merely grows a block by a few linears
        must not be charged the full new flat against a near-zero delta).
        """
        stats = ArenaBuildStats()
        budget_left = None if budget_bytes is None else int(budget_bytes)
        for block_key, raw_entries in entries_by_block.items():
            entries = list(raw_entries)
            _validate_entries_frozen(entries)
            previous = self._blocks.get(block_key)
            if budget_left is not None and previous is not None and previous.pack.pinned:
                budget_left += previous.pack.required_pin_bytes
            want_pin = (
                budget_left is None
                or _entries_total_bytes(entries) <= budget_left
            )
            # Release the old flat's pin BEFORE building the new pack: the
            # old buffer stays alive and valid (merely pageable) as the copy
            # source, but its budget is physically back -- cudaHostUnregister
            # returns DXGI budget immediately, so a rebuild never needs 2x
            # the block's bytes in transient headroom.
            if previous is not None:
                if previous.pack.pinned:
                    pin_manager.unregister_arena_storage(previous.pack.host_flat)
                release_pack(previous.pack)
                # Keep committed_pinned_bytes truthful even if the rebuild
                # below raises mid-way and the old record briefly survives.
                previous.pack.pinned = False
            # pin_mechanism="register": exact-size cudaHostRegister, no
            # caching-allocator power-of-two rounding. pin_alloc flats cost
            # up to 2x their nbytes in DXGI budget (observed live: 8.86 GiB
            # ledger -> 12.70 GiB usage), which starved the arena of the
            # last blocks on a full model.
            pack = pack_block_host(
                block_key, entries, repoint=True, pin=want_pin, kind=kind,
                pin_mechanism="register",
            )
            generation = self._generation.get(block_key, 0) + 1
            self._generation[block_key] = generation
            entry_names = []
            for entry in entries:
                name, module, _weight, _bias = _entry_module_and_leaves(entry)
                entry_names.append(name)
                module._mm_arena_block = block_key
                module._mm_arena_generation = generation
                self._module_index[id(module)] = (block_key, name)
            self._blocks[block_key] = _ArenaBlock(
                pack=pack, generation=generation, entry_names=tuple(entry_names)
            )
            if pack.pinned:
                # Views into this flat report is_pinned()==False (cudaHostRegister);
                # record the storage so the streaming bypass treats them as pinned.
                pin_manager.register_arena_storage(pack.host_flat)
            stats.blocks += 1
            if pack.pinned:
                stats.pinned_bytes += pack.required_pin_bytes
                if budget_left is not None:
                    budget_left -= pack.required_pin_bytes
            else:
                stats.pageable_blocks += 1
                stats.pageable_bytes += pack.required_pin_bytes
        return stats

    # -- membership / staleness ----------------------------------------

    def arena_block_of(self, module) -> Optional[str]:
        """Block key this module was built into, or None if not arena-backed.

        Membership persists across promote/demote cycles regardless of
        where the module's weight currently lives -- this answers "does
        this module belong to the arena", not "is its storage currently an
        arena view"."""
        entry = self._module_index.get(id(module))
        return None if entry is None else entry[0]

    def is_current(self, module) -> bool:
        """False once the module's block has been rebuilt or invalidated
        since this module was tagged (stale -- e.g. after an FP8 demote
        replaced a sibling Linear's Parameter in the same block)."""
        block_key = self.arena_block_of(module)
        if block_key is None:
            return False
        tagged = getattr(module, "_mm_arena_generation", None)
        return tagged == self._generation.get(block_key)

    def invalidate_block(self, block_key: str) -> None:
        """Mark a block stale (e.g. an FP8 demote replaced a Parameter with
        a fresh, non-arena tensor). Conservative at block granularity: ANY
        leaf replacement invalidates the whole block for zero-copy pack
        borrowing, even though only one Linear's storage actually moved.
        The block's flat bytes stay pinned/committed (orphaned, not freed)
        until the block is rebuilt."""
        self._generation[block_key] = self._generation.get(block_key, 0) + 1

    # -- restore ---------------------------------------------------------

    def restore_view(self, module, param_name: str) -> torch.nn.Parameter:
        """Copy a promoted/demoted param's current data back into its arena
        region and repoint the Parameter to that view again.

        Used when a module was promoted to GPU residency (its weight/bias
        Parameter no longer a view into the arena flat) and is now being
        re-offloaded: instead of pinning a fresh standalone buffer, write
        the data back into the arena and hand back a view, so the arena
        stays the single owner of this weight's host pin for the rest of
        the run."""
        entry = self._module_index.get(id(module))
        if entry is None:
            raise KeyError("module is not arena-backed")
        block_key, entry_name = entry
        block = self._blocks[block_key]
        spec = next((s for s in block.pack.linears if s.name == entry_name), None)
        if spec is None:
            raise KeyError(f"no arena entry named {entry_name!r} in block {block_key!r}")
        param = getattr(module, param_name, None)
        if not isinstance(param, torch.nn.Parameter):
            raise ValueError(f"{param_name} is not a Parameter on {entry_name!r}")
        if param_name == "weight":
            leaf_spec, scale_spec = spec.weight, spec.weight_scale
        elif param_name == "bias":
            leaf_spec, scale_spec = spec.bias, None
        else:
            raise ValueError(f"unsupported param_name {param_name!r}")
        if leaf_spec is None:
            raise KeyError(f"{entry_name!r} has no {param_name} leaf in block {block_key!r}")

        flat = block.pack.host_flat
        src = param.data
        src_cpu = src if src.device.type == "cpu" else src.to("cpu")
        current_leaves = _flatten_leaves(src_cpu)
        target_leaves = [leaf_view(flat, leaf_spec)]
        if scale_spec is not None:
            target_leaves.append(leaf_view(flat, scale_spec))
        if len(current_leaves) != len(target_leaves):
            raise ArenaLayoutError(f"arena_layout_mismatch:{entry_name}:{param_name}")
        for current, target in zip(current_leaves, target_leaves):
            target.copy_(current)
        view = target_leaves[0] if scale_spec is None else _rebuild_from_leaves(
            src_cpu, iter(target_leaves)
        )
        new_param = torch.nn.Parameter(view, requires_grad=param.requires_grad)
        setattr(module, param_name, new_param)
        module._mm_arena_block = block_key
        module._mm_arena_generation = self._generation.get(block_key)
        self._module_index[id(module)] = (block_key, entry_name)
        return new_param

    # -- ingraph pack borrowing (Slice 4) ---------------------------------

    def try_borrow_pack(self, block_key: str, linears) -> Optional[BlockPack]:
        """Return a zero-copy, zero-alloc BlockPack over this block's arena
        flat, or None if it can't be safely borrowed (caller must fall back
        to its own owned pack -- this never raises).

        ``linears`` may use a different naming convention than the arena's
        own per-module tags (e.g. ingraph's short per-block names vs the
        arena's full module paths) -- only actual module identity and live
        storage are checked, never names. Every module in ``linears`` must
        currently be arena-current (``is_current``); pack_block_host_from_flat
        then verifies each leaf's real storage against the flat, so a stale
        or partially-rebuilt block fails closed to None rather than ever
        returning a pack over mixed storage.
        """
        record = self._blocks.get(block_key)
        if record is None:
            print(f"[PinnedArena] borrow refused ({block_key}): block_not_in_arena")
            return None
        modules = []
        for entry in linears:
            if len(entry) != 2 or entry[1] is None:
                print(
                    f"[PinnedArena] borrow refused ({block_key}): "
                    "entries_without_module"
                )
                return None
            modules.append(entry[1])
        if not modules:
            print(f"[PinnedArena] borrow refused ({block_key}): no_entries")
            return None
        stale = [m for m in modules if not self.is_current(m)]
        if stale:
            # Distinguish the two very different causes: a module the arena
            # never built (it was RESIDENT when build() ran, so the streamed
            # set and the arena set disagree -- a coverage bug) versus one
            # whose block was rebuilt/invalidated under it (a staleness bug).
            names = dict(zip((entry[0] for entry in linears), modules))
            missing = [n for n, m in names.items() if self.arena_block_of(m) is None]
            outdated = [
                n for n, m in names.items()
                if self.arena_block_of(m) is not None and not self.is_current(m)
            ]
            print(
                f"[PinnedArena] borrow refused ({block_key}): "
                f"stale_modules={len(stale)}/{len(modules)} "
                f"not_in_arena={missing} generation_mismatch={outdated}"
            )
            return None
        try:
            pack = pack_block_host_from_flat(block_key, linears, record.pack.host_flat)
        except ArenaBorrowError as error:
            print(f"[PinnedArena] borrow refused ({block_key}): {error}")
            return None
        if not pack.pinned:
            print(
                f"[PinnedArena] borrow returns a PAGEABLE pack ({block_key}): "
                "block was built past the pin budget; strict ingraph will "
                "reject this as non_pinned_pack"
            )
        return pack

    # -- introspection ----------------------------------------------------

    def block_pack(self, block_key: str) -> Optional[BlockPack]:
        record = self._blocks.get(block_key)
        return None if record is None else record.pack

    def committed_pinned_bytes(self) -> int:
        return sum(
            record.pack.required_pin_bytes
            for record in self._blocks.values()
            if record.pack.pinned
        )

    def pageable_bytes(self) -> int:
        return sum(
            record.pack.required_pin_bytes
            for record in self._blocks.values()
            if not record.pack.pinned
        )

    def has_all_blocks_pinned(self) -> bool:
        return all(record.pack.pinned for record in self._blocks.values())

    def stats(self) -> ArenaBuildStats:
        """Totals across every block currently in the arena (not just ones
        touched by the most recent build() call) -- for attach diagnostics."""
        stats = ArenaBuildStats()
        for record in self._blocks.values():
            stats.blocks += 1
            if record.pack.pinned:
                stats.pinned_bytes += record.pack.required_pin_bytes
            else:
                stats.pageable_blocks += 1
                stats.pageable_bytes += record.pack.required_pin_bytes
        return stats

    def release(self) -> None:
        """Release every owned pack (process/test teardown only -- the
        arena is meant to live for the whole run otherwise)."""
        for record in self._blocks.values():
            if record.pack.pinned:
                pin_manager.unregister_arena_storage(record.pack.host_flat)
            release_pack(record.pack)
        self._blocks.clear()
        self._generation.clear()
        self._module_index.clear()
        _LIVE_ARENAS.discard(self)
