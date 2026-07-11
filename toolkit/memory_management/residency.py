"""Manager-owned device residency sidecars for the immutable host arena.

Residency changes never mutate a Parameter or the canonical host allocation.
They publish optional device tensors keyed by stable ``(block, leaf)`` keys;
execution adapters choose a sidecar or a compact fetched view from a static
transfer plan.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from toolkit.memory_management import pin_manager
from toolkit.memory_management.canonical_arena import CanonicalArena
from toolkit.memory_management.ingraph_stream import (
    _flatten_leaves,
    _rebuild_from_leaves,
    leaf_view,
)

LeafKey = tuple[str, str]


class ResidencyError(RuntimeError):
    """A residency plan or transition violated an immutable-arena invariant."""


@dataclass(frozen=True)
class ResidencyPlan:
    phase: str
    resident_leaf_keys: frozenset[LeafKey]
    fingerprint: str

    @classmethod
    def build(cls, phase: str, resident_leaf_keys) -> ResidencyPlan:
        keys = frozenset((str(block), str(leaf)) for block, leaf in resident_leaf_keys)
        source = repr((str(phase), tuple(sorted(keys))))
        fingerprint = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
        return cls(str(phase), keys, fingerprint)

    @classmethod
    def from_smart_plan(
        cls, arena: CanonicalArena, smart_plan: dict, *, phase: str
    ) -> ResidencyPlan:
        """Adapt the existing planner's ``offload_ids`` decision to sidecars.

        This is the Slice 3 planner seam: priority and capacity remain owned by
        ``MemoryManager.smart_training_plan``; only the mutation target changes.
        """
        offload_ids = set(smart_plan.get("offload_ids", ()))
        resident = []
        for block_key in arena.block_keys():
            block = arena.block_record(block_key)
            for leaf_name, module in zip(
                block.leaf_names, block.modules, strict=True
            ):
                if id(module) not in offload_ids:
                    resident.append((block_key, leaf_name))
        return cls.build(phase, resident)

    def resident_in_block(self, block_key: str) -> frozenset[str]:
        return frozenset(leaf for block, leaf in self.resident_leaf_keys if block == block_key)


@dataclass(frozen=True)
class ResidentLeaf:
    key: LeafKey
    weight: torch.Tensor
    bias: torch.Tensor | None
    ready_event: torch.cuda.Event | None
    nbytes: int


@dataclass(frozen=True)
class ResidencyDelta:
    promoted: tuple[LeafKey, ...]
    demoted: tuple[LeafKey, ...]
    resident_bytes: int


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return sum(leaf.numel() * leaf.element_size() for leaf in _flatten_leaves(tensor))


def _record_stream(tensor: torch.Tensor | None, stream) -> None:
    if tensor is None:
        return
    for leaf in _flatten_leaves(tensor):
        leaf.record_stream(stream)


class ResidencyState:
    """Atomic per-Linear sidecar state over one immutable canonical arena."""

    def __init__(self, arena: CanonicalArena, device) -> None:
        if not arena.canonicalized:
            raise ResidencyError("residency_requires_canonicalized_arena")
        self.arena = arena
        self.device = torch.device(device)
        self._sidecars: dict[LeafKey, ResidentLeaf] = {}
        self._plan = ResidencyPlan.build("empty", ())
        self._copy_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )

    @property
    def plan(self) -> ResidencyPlan:
        return self._plan

    def _all_keys(self) -> frozenset[LeafKey]:
        return frozenset(
            (block_key, leaf_name)
            for block_key in self.arena.block_keys()
            for leaf_name in self.arena.block_record(block_key).leaf_names
        )

    @staticmethod
    def _pin_signature():
        return pin_manager.total_pinned_bytes(), pin_manager.pinned_bytes_by_kind()

    def _canonical_leaf(self, key: LeafKey):
        block_key, leaf_name = key
        block = self.arena.block_record(block_key)
        if block is None:
            raise ResidencyError(f"unknown_residency_block:{block_key}")
        try:
            spec = block.leaf_spec(leaf_name)
            module = block.module_for_leaf(leaf_name)
        except KeyError as error:
            raise ResidencyError(f"unknown_residency_leaf:{block_key}.{leaf_name}") from error
        return block, spec, module

    def _build_sidecar(self, key: LeafKey) -> ResidentLeaf:
        block, spec, module = self._canonical_leaf(key)
        stream_context = (
            torch.cuda.stream(self._copy_stream)
            if self._copy_stream is not None
            else torch.no_grad()
        )
        with torch.no_grad(), stream_context:
            weight_leaf = leaf_view(block.host_flat, spec.weight).to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            if spec.kind == "fp8_rowwise":
                if spec.weight_scale is None:
                    raise ResidencyError(f"missing_quant_scale:{key[0]}.{key[1]}")
                scale_leaf = leaf_view(block.host_flat, spec.weight_scale).to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                weight = _rebuild_from_leaves(
                    module.weight.data, iter((weight_leaf, scale_leaf))
                )
            elif spec.kind == "float":
                weight = weight_leaf
            else:
                raise ResidencyError(f"unsupported_sidecar_kind:{spec.kind}")
            bias = (
                None
                if spec.bias is None
                else leaf_view(block.host_flat, spec.bias).to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
            )
            event = None
            if self._copy_stream is not None:
                event = torch.cuda.Event()
                event.record(self._copy_stream)
        return ResidentLeaf(
            key=key,
            weight=weight,
            bias=bias,
            ready_event=event,
            nbytes=_tensor_bytes(weight) + _tensor_bytes(bias),
        )

    def reconcile(self, plan: ResidencyPlan) -> ResidencyDelta:
        desired = plan.resident_leaf_keys
        unknown = desired - self._all_keys()
        if unknown:
            block, leaf = sorted(unknown)[0]
            raise ResidencyError(f"unknown_residency_leaf:{block}.{leaf}")

        before_pins = self._pin_signature()
        current = set(self._sidecars)
        additions = tuple(sorted(desired - current))
        removals = tuple(sorted(current - desired))
        pending: dict[LeafKey, ResidentLeaf] = {}
        try:
            for key in additions:
                pending[key] = self._build_sidecar(key)
        except Exception as error:
            # Copies may already be queued on the private stream. Drain them
            # before pending tensors are released, while leaving published
            # sidecars and the active plan exactly unchanged.
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
            pending.clear()
            if self._pin_signature() != before_pins:
                raise ResidencyError(
                    "pin_ledger_changed_during_failed_promotion"
                ) from error
            raise

        next_sidecars = {
            key: value for key, value in self._sidecars.items() if key not in removals
        }
        next_sidecars.update(pending)
        self._sidecars = next_sidecars
        self._plan = plan
        if self._pin_signature() != before_pins:
            raise ResidencyError("pin_ledger_changed_during_residency_transition")
        return ResidencyDelta(additions, removals, self.resident_bytes())

    def promote(self, key: LeafKey, *, phase: str | None = None) -> bool:
        normalized = (str(key[0]), str(key[1]))
        if normalized in self._sidecars:
            return False
        plan = ResidencyPlan.build(
            phase or self._plan.phase, self._sidecars.keys() | {normalized}
        )
        self.reconcile(plan)
        return True

    def demote(self, key: LeafKey, *, phase: str | None = None) -> bool:
        normalized = (str(key[0]), str(key[1]))
        if normalized not in self._sidecars:
            return False
        plan = ResidencyPlan.build(
            phase or self._plan.phase, set(self._sidecars) - {normalized}
        )
        self.reconcile(plan)
        return True

    def resident_leaf(self, key: LeafKey) -> ResidentLeaf | None:
        sidecar = self._sidecars.get((str(key[0]), str(key[1])))
        if sidecar is None:
            return None
        if sidecar.ready_event is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_event(sidecar.ready_event)
            _record_stream(sidecar.weight, current)
            _record_stream(sidecar.bias, current)
        return sidecar

    def resident_tensor(self, key: LeafKey) -> torch.Tensor | None:
        sidecar = self.resident_leaf(key)
        return None if sidecar is None else sidecar.weight

    def streamed_leaf_names(self, block_key: str) -> tuple[str, ...]:
        block = self.arena.block_record(block_key)
        if block is None:
            raise ResidencyError(f"unknown_residency_block:{block_key}")
        resident = self._plan.resident_in_block(block_key)
        return tuple(name for name in block.leaf_names if name not in resident)

    def resident_bytes(self) -> int:
        return sum(sidecar.nbytes for sidecar in self._sidecars.values())

    def clear(self, *, phase: str = "clear") -> ResidencyDelta:
        return self.reconcile(ResidencyPlan.build(phase, ()))
