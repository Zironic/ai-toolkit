"""Canonical host arena (Slice 1, tasks/open/IMMUTABLE_TRANSFER_ARENA_PLAN.md).

One page-exclusive pinned host flat per block, immutable leaf metadata, and
a ONE-TIME repoint of frozen base Parameters into views over those flats
(Invariant 4). This is deliberately NOT the legacy ``pinned_arena.py``:
no generation counter, no ``is_current``/staleness oracle over live module
storage, no invalidate/restore/rebuild path, no borrowed-vs-owned pack
taxonomy. Per the plan's Decision section, promotion/demotion/sampling
transitions must never repoint a Parameter again once ``canonicalize()``
has run -- that is the job of the residency sidecars (Slice 3), not this
module.

Reuses ``ingraph_stream.pack_block_host`` for the actual build (it already
implements populate-then-register (Invariant 3, I1) and Parameter
repointing); this module only adds the "exactly once, sequencing-asserted,
fail-closed" discipline Invariant 4 requires on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from toolkit.memory_management import pin_manager
from toolkit.memory_management.ingraph_stream import (
    BlockPack,
    LinearSpec,
    pack_block_host,
    release_pack,
)

ARENA_KIND = "weights"


class CanonicalArenaError(ValueError):
    """A build/canonicalize/guard operation violated a canonical-arena
    invariant (tasks/open/IMMUTABLE_TRANSFER_ARENA_PLAN.md)."""


def _entry_module_and_leaves(entry):
    if len(entry) != 2 or entry[1] is None:
        raise CanonicalArenaError(
            "canonical_arena_requires_module_entries: blocks must be built "
            "from (name, module) entries so Parameters can be repointed"
        )
    name, module = entry
    weight = module.weight
    bias = getattr(module, "bias", None)
    return name, module, weight, bias


def _assert_entries_frozen(entries) -> None:
    """Invariant 4/Decision: the canonical arena is for FROZEN base weights
    only; LoRA/adapters stay ordinary trainable Parameters outside it. Run
    this over every block BEFORE any block is built, so a trainable leaf
    anywhere in the batch fails closed without repointing a single
    Parameter (no partial canonicalization from this particular cause)."""
    for entry in entries:
        name, _module, weight, bias = _entry_module_and_leaves(entry)
        if getattr(weight, "requires_grad", False):
            raise CanonicalArenaError(f"canonical_arena_trainable_leaf:{name}:weight")
        if bias is not None and getattr(bias, "requires_grad", False):
            raise CanonicalArenaError(f"canonical_arena_trainable_leaf:{name}:bias")


@dataclass(frozen=True)
class BlockRecord:
    """One block's immutable canonical host representation.

    No residency state, no generation counter, no live-module currentness
    test -- packs ARE views over arena records (plan Target Components #1).
    """

    block_key: str
    pack: BlockPack
    leaf_names: tuple[str, ...]
    modules: tuple[torch.nn.Module, ...]

    def module_for_leaf(self, leaf_name: str) -> torch.nn.Module:
        try:
            return self.modules[self.leaf_names.index(leaf_name)]
        except ValueError as error:
            raise KeyError(
                f"no leaf named {leaf_name!r} in block {self.block_key!r}"
            ) from error

    @property
    def host_flat(self) -> torch.Tensor:
        return self.pack.host_flat

    @property
    def committed_bytes(self) -> int:
        return self.pack.required_pin_bytes

    def leaf_spec(self, leaf_name: str) -> LinearSpec:
        for spec in self.pack.linears:
            if spec.name == leaf_name:
                return spec
        raise KeyError(f"no leaf named {leaf_name!r} in block {self.block_key!r}")


@dataclass
class CanonicalArenaStats:
    blocks: int = 0
    pinned_bytes: int = 0


class CanonicalArena:
    """Owns one process's canonical, page-exclusive pinned host blocks.

    Not thread-safe; ``canonicalize()``/``release()`` are expected to run on
    the same thread that drives model attach/detach, exactly once each per
    instance -- construct a new ``CanonicalArena`` rather than rebuilding.
    """

    def __init__(self) -> None:
        self._blocks: dict[str, BlockRecord] = {}
        self._canonicalized = False

    @property
    def canonicalized(self) -> bool:
        return self._canonicalized

    # -- canonicalize (Invariant 3 + 4) ------------------------------------

    def canonicalize(
        self, entries_by_block: dict, *, kind: str = ARENA_KIND
    ) -> CanonicalArenaStats:
        """Build every block's canonical flat and repoint its Parameters,
        exactly once. ``entries_by_block`` maps ``block_key -> iterable of
        (name, module)`` entries, all frozen (``requires_grad=False``).

        Caller sequencing responsibility (this method cannot see it): run
        AFTER load/quantize/freeze and BEFORE LoRA attach, optimizer
        construction, or compile (Invariant 4) -- once Parameters are
        repointed here, nothing may replace them again for the life of the
        arena.

        Fails loud and releases any blocks already built in this call if
        any block cannot be admitted (trainable leaf, unsupported quant
        wrapper, or pin budget exceeded) -- there is no silent pageable
        fallback in this arena (that is an admission-policy decision for
        the caller, Invariant 10, not a mechanism this class provides).
        """
        if self._canonicalized:
            raise CanonicalArenaError(
                "canonical_arena_double_canonicalize: canonicalize() may "
                "only run once per arena instance (Invariant 4) -- runtime "
                "promotion/demotion/sampling transitions must never "
                "repoint a Parameter again"
            )
        # Validate every block's leaves are frozen BEFORE building any of
        # them, so a trainable leaf anywhere fails closed with zero
        # Parameters repointed (see _assert_entries_frozen).
        normalized: dict[str, list] = {}
        for block_key, raw_entries in entries_by_block.items():
            entries = list(raw_entries)
            _assert_entries_frozen(entries)
            normalized[block_key] = entries

        stats = CanonicalArenaStats()
        try:
            for block_key, entries in normalized.items():
                pack = pack_block_host(
                    block_key, entries, repoint=True, pin=True, kind=kind,
                    pin_mechanism="register",
                )
                if not pack.pinned:
                    release_pack(pack)
                    raise CanonicalArenaError(
                        f"canonical_arena_pin_budget_exceeded:{block_key}"
                    )
                normalized_entries = tuple(
                    _entry_module_and_leaves(entry) for entry in entries
                )
                leaf_names = tuple(entry[0] for entry in normalized_entries)
                modules = tuple(entry[1] for entry in normalized_entries)
                self._blocks[block_key] = BlockRecord(
                    block_key=block_key,
                    pack=pack,
                    leaf_names=leaf_names,
                    modules=modules,
                )
                pin_manager.register_arena_storage(pack.host_flat)
                stats.blocks += 1
                stats.pinned_bytes += pack.required_pin_bytes
        except Exception:
            # Blocks built earlier in THIS call may already have repointed
            # Parameters onto their (now-being-released) flats -- release()
            # unregisters/unpins them but cannot un-repoint a Parameter
            # (original tensor identity isn't retained). The arena is left
            # un-canonicalized (self._canonicalized stays False) so a
            # caller checking that flag treats the whole attach as failed;
            # it must not proceed to LoRA/optimizer/compile on a
            # partially-repointed model.
            self.release()
            raise
        self._canonicalized = True
        return stats

    # -- whole-model .to() interception (Invariant 5) ----------------------

    @staticmethod
    def guard_whole_model_to(model: torch.nn.Module) -> None:
        """Forbid whole-model ``.to()``/``.cuda()``/``.cpu()`` on a model
        with canonicalized leaves: a model-wide move silently detaches
        every Parameter from its arena flat (copy semantics on the full
        move), the exact drift the legacy arena's ``restore_view`` existed
        to repair after the fact. Idempotent. Callers that need to move
        SOME parameters (LoRA, non-canonicalized submodules) must route
        through a canonical-arena-aware helper instead of raw ``.to()``."""
        if getattr(model, "_mm_canonical_to_guarded", False):
            return
        original_to = model.to

        def _guarded_to(*_args, **_kwargs):
            raise CanonicalArenaError(
                "canonical_arena_whole_model_to: whole-model .to()/.cuda()/"
                ".cpu() is forbidden once canonicalized leaves exist -- it "
                "would silently detach every Parameter from its arena flat. "
                "Route non-canonicalized regions through their own .to() "
                "calls, or move canonicalized weights via the residency "
                "sidecar path instead."
            )

        model.to = _guarded_to
        model._mm_canonical_to_guarded = True
        model._mm_canonical_to_original = original_to

    @staticmethod
    def unguard_whole_model_to(model: torch.nn.Module) -> None:
        original = getattr(model, "_mm_canonical_to_original", None)
        if original is not None:
            model.to = original
            del model._mm_canonical_to_original
        if hasattr(model, "_mm_canonical_to_guarded"):
            del model._mm_canonical_to_guarded

    # -- introspection ------------------------------------------------------

    def block_record(self, block_key: str) -> BlockRecord | None:
        return self._blocks.get(block_key)

    def block_pack(self, block_key: str) -> BlockPack | None:
        record = self._blocks.get(block_key)
        return None if record is None else record.pack

    def block_keys(self) -> tuple[str, ...]:
        return tuple(self._blocks.keys())

    def committed_pinned_bytes(self) -> int:
        return sum(record.committed_bytes for record in self._blocks.values())

    def stats(self) -> CanonicalArenaStats:
        return CanonicalArenaStats(
            blocks=len(self._blocks), pinned_bytes=self.committed_pinned_bytes()
        )

    # -- explicit unload ------------------------------------------------

    def release(self) -> None:
        """Release every block's pin registration and every committed
        byte (Invariant 1: pin_manager is the sole authority, every
        registered byte is released explicitly). Safe to call on a
        partially-built or already-released arena."""
        for record in self._blocks.values():
            pin_manager.unregister_arena_storage(record.pack.host_flat)
            release_pack(record.pack)
        self._blocks.clear()
        self._canonicalized = False
