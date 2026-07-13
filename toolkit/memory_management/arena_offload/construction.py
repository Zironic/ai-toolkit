"""Destination-first transactional construction of canonical arena storage."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from toolkit.memory_management import pin_manager
from .layout import (
    BlockPack,
    LeafSpec,
    LinearSpec,
    flatten_leaves,
    inspect_block,
    linear_views,
    make_block_view_maker,
    typed_view,
)


class CanonicalBuildError(RuntimeError):
    pass


@dataclass
class _PreparedBlock:
    key: str
    entries: tuple
    layout: object
    flat: torch.Tensor
    pending: object
    handle: object | None = None
    pack: BlockPack | None = None


class PreparedCanonicalBuild:
    """A prepared arena build whose model publication is atomic."""

    def __init__(self, arena, entries_by_block, *, model=None, kind="weights"):
        self.arena = arena
        self.model = model
        self.kind = kind
        self.blocks = []
        self.destinations = {}
        self.entries_by_block = {}
        self._originals = []
        self._populated = False
        self._committed = False
        if arena.canonicalized:
            raise CanonicalBuildError("canonical_arena_double_canonicalize")
        try:
            for key, raw_entries in entries_by_block.items():
                self.add_block(key, raw_entries)
        except Exception:
            self.rollback()
            raise

    def add_block(self, key, raw_entries) -> None:
        """Prepare one final block layout for a bounded direct loader."""
        if self._populated or self._committed:
            raise CanonicalBuildError("canonical_build_already_populated")
        if key in self.entries_by_block:
            raise CanonicalBuildError(f"canonical_build_duplicate_block:{key}")
        entries = tuple(raw_entries)
        layout = inspect_block(key, entries)
        for name, module in entries:
            if module.weight.requires_grad or (
                getattr(module, "bias", None) is not None and module.bias.requires_grad
            ):
                raise CanonicalBuildError(f"canonical_arena_trainable_leaf:{key}:{name}")
            self._originals.append((module, module.weight, getattr(module, "bias", None)))
        flat, padded = pin_manager.pin_register_prepare(layout.nbytes)
        block = _PreparedBlock(key, entries, layout, flat, padded)
        self.blocks.append(block)
        self.entries_by_block[key] = entries
        for linear in layout.linears:
            for leaf in linear.leaf_descriptors:
                self.destinations[(key, linear.name, leaf.role)] = typed_view(flat, leaf)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None or not self._committed:
            self.rollback()
        return False

    def populate(self, source) -> None:
        try:
            source(self.destinations)
            self._finish_population()
        except Exception:
            self.rollback()
            raise

    def populate_from_model(self) -> None:
        try:
            for destination_key, source in self.model_source_leaves():
                self.destinations[destination_key].copy_(source)
            self._finish_population()
        except Exception:
            self.rollback()
            raise

    def populate_block_from_model(self, block_key: str) -> None:
        """Copy one bounded loaded block without finalizing the whole build."""
        for destination_key, source in self.model_source_leaves(block_key=block_key):
            self.destinations[destination_key].copy_(source)

    def release_block_sources_to_meta(self, block_key: str) -> None:
        """Drop a direct loader's bounded source after its final flat is populated."""
        block = next(item for item in self.blocks if item.key == block_key)
        meta_flat = torch.empty(block.layout.nbytes, dtype=torch.uint8, device="meta")
        replacements = {}
        modules = dict(block.entries)
        meta_linears = []
        for linear in block.layout.linears:
            weight, bias = linear_views(meta_flat, linear)
            module = modules[linear.name]
            meta_weight = torch.nn.Parameter(
                weight, requires_grad=linear.weight_requires_grad
            )
            meta_bias = (
                None
                if bias is None
                else torch.nn.Parameter(bias, requires_grad=linear.bias_requires_grad)
            )
            module._parameters["weight"] = meta_weight
            if bias is not None:
                module._parameters["bias"] = meta_bias
            replacements[module] = (meta_weight, meta_bias)
            meta_linears.append(replace(linear, weight_template=meta_weight.data))
        block.layout = replace(block.layout, linears=tuple(meta_linears))
        self._originals = [
            (module, *replacements[module])
            if module in replacements
            else (module, weight, bias)
            for module, weight, bias in self._originals
        ]

    def finish_population(self) -> None:
        try:
            self._finish_population()
        except Exception:
            self.rollback()
            raise

    def model_source_leaves(self, *, block_key: str | None = None):
        """Yield loaded model leaves keyed exactly like final destinations."""
        for block in self.blocks:
            if block_key is not None and block.key != block_key:
                continue
            by_name = dict(block.entries)
            for linear in block.layout.linears:
                module = by_name[linear.name]
                weight_leaves = flatten_leaves(module.weight.data)
                bias = getattr(module, "bias", None)
                bias_leaves = [] if bias is None else flatten_leaves(bias.data)
                for descriptor, source in zip(
                    linear.leaf_descriptors, weight_leaves + bias_leaves
                ):
                    yield (block.key, linear.name, descriptor.role), source

    def _finish_population(self) -> None:
        for block in self.blocks:
            handle = pin_manager.pin_register_commit(
                block.flat, block.layout.nbytes, self.kind, required=False
            )
            if not handle.pinned:
                pin_manager.release(handle)
                raise CanonicalBuildError(f"canonical_arena_pin_budget_exceeded:{block.key}")
            block.handle = handle
            block.flat = handle.tensor
            # Validate every supported wrapper reconstruction before publication.
            for linear in block.layout.linears:
                linear_views(block.flat, linear)
        self._populated = True

    def commit(self):
        if not self._populated:
            self.rollback()
            raise CanonicalBuildError("canonical_build_not_populated")
        from toolkit.memory_management.canonical_arena import BlockRecord, CanonicalArenaStats
        published = []
        try:
            for block in self.blocks:
                specs = []
                for linear in block.layout.linears:
                    weight, bias = linear_views(block.flat, linear)
                    module = dict(block.entries)[linear.name]
                    module.weight = torch.nn.Parameter(weight, requires_grad=linear.weight_requires_grad)
                    if bias is not None:
                        module.bias = torch.nn.Parameter(bias, requires_grad=linear.bias_requires_grad)
                    published.append(module)
                    leaves = linear.leaf_descriptors
                    weight_role = (
                        "qdata" if linear.leaf("scale") is not None else "float_weight"
                    )
                    weight_spec = LeafSpec(
                        leaves[0].offset, leaves[0].nbytes, leaves[0].dtype,
                        leaves[0].shape, weight_role,
                    )
                    bias_leaf = linear.leaf("bias")
                    scale_leaf = linear.leaf("scale")
                    specs.append(LinearSpec(
                        linear.name, weight_spec,
                        None if bias_leaf is None else LeafSpec(**bias_leaf.__dict__),
                        linear.weight_requires_grad, linear.bias_requires_grad,
                        "fp8_rowwise" if scale_leaf is not None else "float",
                        None if scale_leaf is None else LeafSpec(**scale_leaf.__dict__),
                        linear.native_fp8_eligible,
                    ))
                pack = BlockPack(block.key, block.flat, tuple(specs), block.layout.nbytes, True,
                                 tuple(x.native_fp8_eligible for x in block.layout.linears),
                                 pin_handle=block.handle)
                pack.view_maker = make_block_view_maker(pack)
                block.pack = pack
                names = tuple(name for name, _ in block.entries)
                modules = tuple(module for _, module in block.entries)
                self.arena._blocks[block.key] = BlockRecord(block.key, pack, names, modules)
                pin_manager.register_arena_storage(block.flat)
            self.arena._canonicalized = True
            if self.model is not None:
                self.arena.guard_whole_model_to(self.model)
            self._committed = True
            return CanonicalArenaStats(len(self.blocks), sum(b.layout.nbytes for b in self.blocks))
        except Exception:
            self.rollback()
            raise

    def rollback(self) -> None:
        for module, weight, bias in reversed(self._originals):
            # Bypass user/module publication hooks: rollback must remain valid
            # even when the injected/real failure was an attribute assignment.
            module._parameters["weight"] = weight
            if bias is not None:
                module._parameters["bias"] = bias
        if self.model is not None:
            self.arena.unguard_whole_model_to(self.model)
        for block in self.blocks:
            if block.pack is not None:
                pin_manager.unregister_arena_storage(block.flat)
            pin_manager.release(block.handle)
            block.handle = None
        self.arena._blocks.clear()
        self.arena._canonicalized = False
        self._populated = False
