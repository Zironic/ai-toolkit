"""Generic compile-neutral source state for immutable transformer runtimes.

The source table owns only publication and execution exclusion. Canonical
storage remains owned by the arena, while ``ResidencyState`` continues to own
resident device sidecars.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from toolkit.memory_management import vram_budget
from toolkit.memory_management.ingraph_stream import (
    _flatten_leaves,
    checkpoint_recompute_context,
    compiled_checkpoint_context,
    configure_fetch_runtime,
    free_on_backward,
    in_recompute,
)
from toolkit.memory_management.residency import (
    ResidencyDelta,
    ResidencyPlan,
    ResidencyState,
)
from toolkit.memory_management.transfer_plan import (
    BlockTransferPlan,
    build_transfer_plan,
)


class ImmutableRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImmutableBlockABI:
    """Structural block information that remains stable for runtime life."""

    block_key: str
    leaf_names: tuple[str, ...]
    fp8_flags: tuple[bool, ...]
    leaf_layout: tuple[tuple[str, tuple[int, ...], str, bool, bool], ...]


@dataclass(frozen=True)
class ImmutableBlockSourceSnapshot:
    """Current source-selection metadata for one canonical block."""

    block_key: str
    leaf_names: tuple[str, ...]
    resident_leaf_names: frozenset[str]
    transfer: BlockTransferPlan | None
    ranges: torch.Tensor | None

    def assemble_leaf_args(
        self,
        residency: ResidencyState,
        compact_flat: torch.Tensor | None,
    ) -> tuple:
        record = residency.arena.block_record(self.block_key)
        if record is None:
            raise ImmutableRuntimeError(f"missing_canonical_block:{self.block_key}")

        args = []
        for leaf_name in self.leaf_names:
            spec = record.leaf_spec(leaf_name)
            if leaf_name in self.resident_leaf_names:
                sidecar = residency.resident_leaf((self.block_key, leaf_name))
                if sidecar is None:
                    raise ImmutableRuntimeError(f"missing_resident_source:{self.block_key}.{leaf_name}")
                args.append(_resident_args(sidecar, spec.kind))
                continue

            if compact_flat is None or self.transfer is None:
                raise ImmutableRuntimeError(f"missing_streamed_source:{self.block_key}.{leaf_name}")

            weight = self.transfer.compact_leaf_view(
                compact_flat,
                leaf_name,
                "weight",
            )
            bias = (
                None
                if spec.bias is None
                else self.transfer.compact_leaf_view(
                    compact_flat,
                    leaf_name,
                    "bias",
                )
            )
            scale = (
                None
                if spec.weight_scale is None
                else self.transfer.compact_leaf_view(
                    compact_flat,
                    leaf_name,
                    "weight_scale",
                )
            )
            args.append((weight, bias, scale))

        return tuple(args)


def _resident_args(sidecar, kind: str):
    leaves = _flatten_leaves(sidecar.weight)
    if kind == "float" and len(leaves) == 1:
        return leaves[0], sidecar.bias, None
    if kind == "fp8_rowwise" and len(leaves) == 2:
        return leaves[0], sidecar.bias, leaves[1]
    raise ImmutableRuntimeError(f"resident_sidecar_layout_mismatch:{sidecar.key[0]}.{sidecar.key[1]}")


def build_source_snapshot(
    residency: ResidencyState,
    plan: ResidencyPlan,
    abi: ImmutableBlockABI,
) -> ImmutableBlockSourceSnapshot:
    record = residency.arena.block_record(abi.block_key)
    if record is None:
        raise ImmutableRuntimeError(f"missing_canonical_block:{abi.block_key}")

    resident = plan.resident_in_block(abi.block_key)
    unknown = resident - frozenset(abi.leaf_names)
    if unknown:
        leaf_name = sorted(unknown)[0]
        raise ImmutableRuntimeError(f"unknown_residency_leaf:{abi.block_key}.{leaf_name}")

    streamed = tuple(leaf_name for leaf_name in abi.leaf_names if leaf_name not in resident)
    transfer = build_transfer_plan(record, streamed) if streamed else None
    ranges = None if transfer is None else transfer.ranges_tensor()
    return ImmutableBlockSourceSnapshot(
        block_key=abi.block_key,
        leaf_names=abi.leaf_names,
        resident_leaf_names=resident,
        transfer=transfer,
        ranges=ranges,
    )


@dataclass(frozen=True)
class ImmutableProgram:
    """One permanent eager program for one execution mode."""

    mode: str
    fingerprint: str
    trunk: object


def build_program_fingerprint(
    mode: str,
    block_abis,
    *,
    architecture_key: str,
    depth: int,
    checkpoint_mode: str,
    lora_shape=(),
    has_multiplier: bool = False,
) -> str:
    per_block = tuple(
        (
            abi.block_key,
            tuple(abi.leaf_names),
        )
        for abi in block_abis
    )
    source = repr(
        (
            "immutable-runtime-v1",
            str(architecture_key),
            str(mode),
            per_block,
            int(depth),
            str(checkpoint_mode),
            tuple(lora_shape),
            bool(has_multiplier),
        )
    )
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


def build_train_trunk(block_fns):
    block_fns = tuple(block_fns)

    def immutable_train_trunk(combined, tvec, freqs, mask):
        context_fn = compiled_checkpoint_context if torch.compiler.is_compiling() else checkpoint_recompute_context
        for block_fn in block_fns:
            combined = checkpoint(
                block_fn,
                combined,
                tvec,
                freqs,
                mask,
                use_reentrant=False,
                context_fn=context_fn,
            )
        return combined

    return immutable_train_trunk


def build_sample_trunk(block_fns):
    block_fns = tuple(block_fns)

    def immutable_sample_trunk(combined, tvec, freqs, mask):
        for block_fn in block_fns:
            combined = block_fn(combined, tvec, freqs, mask)
        return combined

    return immutable_sample_trunk


class ImmutableRuntimeSourceTable:
    """Atomically published per-block source snapshots."""

    def __init__(self, residency: ResidencyState, block_abis) -> None:
        self.residency = residency
        self.block_abis = tuple(block_abis)
        self._generation = 0
        self._active_executions = 0
        self._snapshots: tuple[ImmutableBlockSourceSnapshot, ...] | None = None
        self._plan: ResidencyPlan | None = None

        if residency.plan.phase != "empty":
            self._snapshots = self._build_snapshots(residency.plan)
            self._plan = residency.plan
            self._generation = 1

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def plan(self) -> ResidencyPlan | None:
        return self._plan

    @property
    def active_executions(self) -> int:
        return self._active_executions

    def _build_snapshots(
        self,
        plan: ResidencyPlan,
    ) -> tuple[ImmutableBlockSourceSnapshot, ...]:
        return tuple(build_source_snapshot(self.residency, plan, abi) for abi in self.block_abis)

    def begin_execution(self) -> int:
        if self._snapshots is None:
            raise ImmutableRuntimeError("no_residency_source_table")
        self._active_executions += 1
        return self._generation

    def end_execution(self, generation: int) -> None:
        if self._active_executions <= 0:
            raise ImmutableRuntimeError("immutable_execution_not_active")
        if int(generation) != self._generation:
            raise ImmutableRuntimeError("immutable_execution_generation_mismatch")
        self._active_executions -= 1

    def source(self, block_index: int) -> ImmutableBlockSourceSnapshot:
        snapshots = self._snapshots
        if snapshots is None:
            raise ImmutableRuntimeError("no_residency_source_table")
        try:
            return snapshots[int(block_index)]
        except IndexError as error:
            raise ImmutableRuntimeError(f"unknown_execution_block:{block_index}") from error

    def publish(self, plan: ResidencyPlan) -> ResidencyDelta:
        if self._active_executions:
            raise ImmutableRuntimeError("residency_transition_during_execution")

        snapshots = self._build_snapshots(plan)
        if self.residency.plan.fingerprint == plan.fingerprint:
            delta = ResidencyDelta((), (), self.residency.resident_bytes())
        else:
            delta = self.residency.reconcile(plan)

        self._snapshots = snapshots
        self._plan = plan
        self._generation += 1
        return delta

    def clear(self) -> None:
        if self._active_executions:
            raise ImmutableRuntimeError("source_table_clear_during_execution")
        self._snapshots = None
        self._plan = None


def _mark_dynamic_dim(tensor, dim: int) -> None:
    """Request a dynamic dim without enforcing an invalid dense interval."""
    torch._dynamo.maybe_mark_dynamic(tensor, int(dim))


def _leaf_layout(record) -> tuple:
    layout = []
    for leaf_name in record.leaf_names:
        spec = record.leaf_spec(leaf_name)
        layout.append(
            (
                leaf_name,
                tuple(spec.weight.shape),
                str(spec.weight.dtype),
                spec.bias is not None,
                spec.weight_scale is not None,
            )
        )
    return tuple(layout)


def build_block_abi(
    model,
    residency: ResidencyState,
    adapter,
    index: int,
) -> ImmutableBlockABI:
    block_key = adapter.block_key(model, index)
    record = residency.arena.block_record(block_key)
    if record is None:
        raise ImmutableRuntimeError(f"missing_canonical_block:{block_key}")

    block = adapter.execution_blocks(model)[index]
    expected = tuple(name for name, _module in adapter.leaf_entries(block))
    if record.leaf_names != expected:
        raise ImmutableRuntimeError(
            f"canonical_leaf_order_mismatch:{block_key}:expected={expected}:actual={record.leaf_names}"
        )

    return ImmutableBlockABI(
        block_key=block_key,
        leaf_names=record.leaf_names,
        fp8_flags=tuple(spec.fp8_qualifies for spec in record.pack.linears),
        leaf_layout=_leaf_layout(record),
    )


class ImmutableTransformerRuntime:
    """Permanent eager programs over mutable canonical-block source state."""

    TRAIN = "train"
    SAMPLE = "sample"

    def __init__(
        self,
        model,
        residency: ResidencyState,
        *,
        architecture_adapter,
        depth: int = 2,
        compile_blocks: bool = True,
        compile_dynamic: bool | None = True,
        compile_dynamic_hints: tuple[tuple[int, int | None, int | None], ...] = (),
    ) -> None:
        self._sampling_working_bytes: dict[tuple, int] = {}
        self._sampling_baseline = None
        # External-VRAM check is reported once per phase, not per image/step.
        self._foreign_vram_checked = False
        self.model = model
        self.residency = residency
        self.architecture_adapter = architecture_adapter
        self._blocks = self.architecture_adapter.execution_blocks(model)
        self.loras_by_block = {}
        self.lora_multiplier = None
        self.depth = max(1, int(depth))
        self.compile_blocks = bool(compile_blocks)
        self.compile_dynamic = (
            None if compile_dynamic is None else bool(compile_dynamic)
        )
        self.compile_dynamic_hints = tuple(compile_dynamic_hints or ())
        self._hint_range_warned: set[tuple] = set()
        self._arena_signature = self.residency.arena.immutable_signature()

        self._block_abis = tuple(
            build_block_abi(
                model,
                residency,
                self.architecture_adapter,
                index,
            )
            for index in range(len(self._blocks))
        )
        self._sources = ImmutableRuntimeSourceTable(residency, self._block_abis)
        self._block_kernels: dict[tuple[str, int, tuple[bool, ...]], object] = {}
        self._block_fns: dict[str, tuple] = {}
        self._programs: dict[str, ImmutableProgram] = {}
        self._finalized = False
        self._finalization_signature = None
        self._active_token = None
        self._active_mode = None
        self.stats = {
            "residency_transitions": 0,
            "source_generation": self._sources.generation,
        }
        self.sampling_fallback_plan = ResidencyPlan.build(
            "sample_fallback",
            (),
        )

    @property
    def source_generation(self) -> int:
        return self._sources.generation

    @property
    def active_executions(self) -> int:
        return self._sources.active_executions

    @property
    def finalized(self) -> bool:
        return self._finalized

    def source(self, block_index: int) -> ImmutableBlockSourceSnapshot:
        """Current published source snapshot for one block."""
        return self._sources.source(block_index)

    @staticmethod
    def _execution_signature(loras_by_block, lora_multiplier):
        entries = []
        for index, loras in sorted((loras_by_block or {}).items()):
            for name, entry in sorted(loras.items()):
                entries.append((int(index), str(name), id(entry)))
        return tuple(entries), id(lora_multiplier)

    def finalize_execution(
        self,
        *,
        loras_by_block=None,
        lora_multiplier=None,
    ):
        signature = self._execution_signature(
            loras_by_block,
            lora_multiplier,
        )
        if self._finalized:
            if signature != self._finalization_signature:
                raise ImmutableRuntimeError(
                    "incompatible_immutable_runtime_finalization"
                )
            return self

        self.loras_by_block = dict(loras_by_block or {})
        self.lora_multiplier = lora_multiplier
        configure_fetch_runtime(depth=self.depth)
        self._block_fns = {
            self.TRAIN: tuple(
                self._make_stable_block_fn(index, self.TRAIN)
                for index in range(len(self._blocks))
            ),
            self.SAMPLE: tuple(
                self._make_stable_block_fn(index, self.SAMPLE)
                for index in range(len(self._blocks))
            ),
        }
        self._programs = {
            self.TRAIN: self._build_program(self.TRAIN),
            self.SAMPLE: self._build_program(self.SAMPLE),
        }
        self._finalization_signature = signature
        self._finalized = True
        return self

    def _require_finalized(self) -> None:
        if not self._finalized:
            raise ImmutableRuntimeError("immutable_runtime_not_finalized")


    @contextmanager
    def execution(self, mode: str):
        self._require_finalized()
        if mode not in (self.TRAIN, self.SAMPLE):
            raise ImmutableRuntimeError(f"unknown_execution_mode:{mode}")
        if self._active_token is not None:
            raise ImmutableRuntimeError(
                f"immutable_execution_already_active:{self._active_mode}"
            )

        token = object()
        generation = self._sources.begin_execution()
        self._active_token = token
        self._active_mode = mode
        try:
            yield self
        finally:
            if self._active_token is not token:
                raise ImmutableRuntimeError("immutable_execution_token_mismatch")
            self._active_token = None
            self._active_mode = None
            self._sources.end_execution(generation)

    @contextmanager
    def sampling(self, *, shape_key: tuple, **activate_kwargs):
        succeeded = False
        try:
            self.activate_sampling_image(
                shape_key=shape_key,
                **activate_kwargs,
            )
            with self.execution(self.SAMPLE):
                yield self
            succeeded = True
        finally:
            if succeeded:
                self.finish_sampling_image(shape_key=shape_key)
            else:
                self._sampling_baseline = None
    def can_run_current_call(
        self,
        tvec,
        freqs,
        mask,
        *,
        ref_kv_capture=None,
        blockcaches=None,
    ) -> bool:
        return self._finalized and self.architecture_adapter.can_run_current_call(
            (tvec, freqs, mask),
            ref_kv_capture=ref_kv_capture,
            blockcaches=blockcaches,
        )

    def _assert_arena_stable(self, where: str) -> None:
        current = self.residency.arena.immutable_signature()
        if current != self._arena_signature:
            raise ImmutableRuntimeError(
                f"arena_mutated_at_boundary:{where}: canonical host flats "
                "or registrations changed across a phase boundary"
            )

    def _get_block_kernel(self, index: int, mode: str, fp8_flags):
        fp8_flags = tuple(fp8_flags)
        key = (str(mode), int(index), fp8_flags)
        existing = self._block_kernels.get(key)
        if existing is not None:
            return existing

        block = self._blocks[index]
        training = mode == self.TRAIN

        def block_kernel(
            x,
            tvec,
            freqs,
            mask,
            leaf_args,
            lora_args,
        ):
            return self.architecture_adapter.forward_block(
                block,
                x,
                (tvec, freqs, mask),
                leaf_args,
                fp8_flags,
                lora_args,
                training=training,
            )

        kernel = block_kernel
        if self.compile_blocks:
            kernel = torch.compile(
                kernel,
                mode="default",
                fullgraph=False,
                dynamic=self.compile_dynamic,
            )
        self._block_kernels[key] = kernel
        return kernel

    def set_compile_dynamic_hints(self, hints) -> None:
        """Install mark_dynamic hints derived after the runtime was prepared.

        The trainer can only compute sequence bounds once the datasets exist,
        which is long after `prepare_arena_offload`. Hints are read per call, so
        installing them any time before the first compiled block call is enough.
        A block kernel that already traced would have to re-specialize, so
        refuse once anything is compiled rather than pay a silent recompile.
        """
        hints = tuple(tuple(hint) for hint in (hints or ()))
        if hints == self.compile_dynamic_hints:
            return
        if self._block_kernels:
            raise RuntimeError(
                "compile_dynamic_hints changed after block kernels were built; "
                "set them before the first forward pass."
            )
        self.compile_dynamic_hints = hints
        self._hint_range_warned.clear()

    def _warn_hint_out_of_range(self, dim, size, lo, hi) -> None:
        key = (dim, size)
        if key in self._hint_range_warned:
            return
        self._hint_range_warned.add(key)
        print(
            f"[immutable] dim {dim} size {size} is outside the declared dynamic "
            f"range [{lo}, {hi}]; compiling a dedicated shape for it. "
            "Widen compile_dynamic_hints to avoid the extra compile."
        )

    def _current_lora_args(self, index: int):
        return self.architecture_adapter.build_lora_args(
            index,
            self.loras_by_block,
            self.lora_multiplier,
        )

    def _make_stable_block_fn(self, index: int, mode: str):
        abi = self._block_abis[index]
        kernel = self._get_block_kernel(index, mode, abi.fp8_flags)
        training = mode == self.TRAIN

        def block_fn(x, tvec, freqs, mask):
            source = self._sources.source(index)
            transfer = source.transfer
            token = None
            compact_flat = None

            if transfer is not None:
                host = self.residency.arena.block_record(source.block_key).host_flat
                nbytes = int(transfer.compact_nbytes)
                guard = x.reshape(-1)[:1].clone() if training else x
                token = torch.ops.mm.fetch_start_multi_after(
                    host,
                    source.ranges,
                    nbytes,
                    guard,
                )
                compact_flat = torch.ops.mm.fetch_wait(token, nbytes)
                if training and torch.is_grad_enabled():
                    x = free_on_backward(x, token)

            leaf_args = source.assemble_leaf_args(
                self.residency,
                compact_flat,
            )
            if self.compile_blocks and self.compile_dynamic_hints:
                # Must be set on this exact tensor instance every call (a
                # fresh x each step) before it crosses the torch.compile
                # boundary in `kernel`, so a resolution-bucket run requests a
                # dynamic sequence dimension instead of specializing eagerly.
                for dim, lo, hi in self.compile_dynamic_hints:
                    size = int(x.shape[dim])
                    if (lo is not None and size < lo) or (
                        hi is not None and size > hi
                    ):
                        # A shape the bounds did not anticipate. Marking it
                        # anyway is a hard ConstraintViolation; skipping the
                        # hint just costs this shape its own specialization.
                        self._warn_hint_out_of_range(dim, size, lo, hi)
                        continue
                    # Krea sequence lengths are aligned, so the kernel may infer
                    # divisibility guards (for example size % 8 == 0). A hard
                    # min/max mark claims every integer in the interval is
                    # valid and raises ConstraintViolationError. Weak marking
                    # requests dynamism without forbidding specialization when
                    # an inferred alignment constraint requires it.
                    _mark_dynamic_dim(x, dim)
            out = kernel(
                x,
                tvec,
                freqs,
                mask,
                leaf_args,
                self._current_lora_args(index),
            )

            if token is not None:
                if training:
                    if not in_recompute():
                        torch.ops.mm.fetch_free_after(token, out)
                else:
                    torch.ops.mm.fetch_free_after(token, out)
            return out

        return block_fn

    def _build_program(self, mode: str) -> ImmutableProgram:
        lora_shape = tuple((index, tuple(sorted(entries))) for index, entries in sorted(self.loras_by_block.items()))
        fingerprint = build_program_fingerprint(
            mode,
            self._block_abis,
            architecture_key=self.architecture_adapter.architecture_key,
            depth=self.depth,
            checkpoint_mode="full" if mode == self.TRAIN else "none",
            lora_shape=lora_shape,
            has_multiplier=self.lora_multiplier is not None,
        )
        trunk = (
            build_train_trunk(self._block_fns[mode])
            if mode == self.TRAIN
            else build_sample_trunk(self._block_fns[mode])
        )
        return ImmutableProgram(
            mode=mode,
            fingerprint=fingerprint,
            trunk=trunk,
        )

    def set_residency_plan(self, plan: ResidencyPlan) -> ResidencyDelta:
        self._assert_arena_stable("pre_residency_publish")
        delta = self._sources.publish(plan)
        self._assert_arena_stable("post_residency_publish")
        self.stats["residency_transitions"] += 1
        self.stats["source_generation"] = self._sources.generation
        return delta

    def activate(self, mode: str, plan: ResidencyPlan) -> ImmutableProgram:
        self._require_finalized()
        if mode not in (self.TRAIN, self.SAMPLE):
            raise ImmutableRuntimeError(f"unknown_execution_mode:{mode}")
        self.set_residency_plan(plan)
        return self.program(mode)

    def program(self, mode: str) -> ImmutableProgram:
        self._require_finalized()
        try:
            return self._programs[mode]
        except KeyError as error:
            raise ImmutableRuntimeError(f"unknown_execution_mode:{mode}") from error

    def activate_sampling_fallback(self) -> ImmutableProgram:
        self.set_residency_plan(self.sampling_fallback_plan)
        return self.program(self.SAMPLE)

    def transition_training_blocks(self, block_keys, *, resident: bool) -> dict:
        """Atomically add or remove complete training blocks in one plan."""
        current = self._sources.plan or self.residency.plan
        if current.phase != self.TRAIN:
            raise ImmutableRuntimeError(
                f"training_block_transition_requires_train:{current.phase}"
            )
        requested = tuple(dict.fromkeys(str(key) for key in block_keys))
        protected = frozenset(
            (str(block), str(leaf))
            for block, leaf in getattr(
                self.model,
                "_mm_immutable_protected_training_leaf_keys",
                (),
            )
        )
        next_keys = set(current.resident_leaf_keys)
        changed = []
        for key in requested:
            abi = next(
                (item for item in self._block_abis if item.block_key == key),
                None,
            )
            if abi is None:
                raise ImmutableRuntimeError(f"unknown_training_block:{key}")
            leaf_keys = tuple((key, leaf) for leaf in abi.leaf_names)
            present = tuple(
                item for item in leaf_keys if item in current.resident_leaf_keys
            )
            if present and len(present) != len(leaf_keys):
                raise ImmutableRuntimeError(
                    f"partial_training_block_layout:{key}"
                )
            if not resident and any(item in protected for item in leaf_keys):
                raise ImmutableRuntimeError(f"protected_training_block:{key}")
            if bool(present) == bool(resident):
                continue
            changed.append(key)
            if resident:
                next_keys.update(leaf_keys)
            else:
                next_keys.difference_update(leaf_keys)
        if not changed:
            return {
                "changed": False,
                "block_keys": requested,
                "resident": bool(resident),
                "plan": current,
            }
        next_plan = ResidencyPlan.build(self.TRAIN, next_keys)
        delta = self.set_residency_plan(next_plan)
        self.model._mm_immutable_training_plan = next_plan
        return {
            "changed": True,
            "block_keys": tuple(changed),
            "resident": bool(resident),
            "resident_bytes": self.residency.resident_bytes(),
            "delta": delta,
            "plan": next_plan,
        }

    def transition_training_block(self, block_key: str, *, resident: bool) -> dict:
        """Atomically add or remove one complete training block by stable key."""
        current = self._sources.plan or self.residency.plan
        if current.phase != self.TRAIN:
            raise ImmutableRuntimeError(
                f"training_block_transition_requires_train:{current.phase}"
            )
        key = str(block_key)
        abi = next((item for item in self._block_abis if item.block_key == key), None)
        if abi is None:
            raise ImmutableRuntimeError(f"unknown_training_block:{key}")
        leaf_keys = tuple((key, leaf) for leaf in abi.leaf_names)
        present = tuple(item for item in leaf_keys if item in current.resident_leaf_keys)
        if present and len(present) != len(leaf_keys):
            raise ImmutableRuntimeError(f"partial_training_block_layout:{key}")
        want_resident = bool(resident)
        if bool(present) == want_resident:
            return {
                "changed": False,
                "block_key": key,
                "resident": want_resident,
                "plan": current,
            }
        protected = frozenset(
            (str(block), str(leaf))
            for block, leaf in getattr(
                self.model,
                "_mm_immutable_protected_training_leaf_keys",
                (),
            )
        )
        if not want_resident and any(item in protected for item in leaf_keys):
            raise ImmutableRuntimeError(f"protected_training_block:{key}")

        next_keys = set(current.resident_leaf_keys)
        if want_resident:
            next_keys.update(leaf_keys)
        else:
            next_keys.difference_update(leaf_keys)
        next_plan = ResidencyPlan.build(self.TRAIN, next_keys)
        delta = self.set_residency_plan(next_plan)
        self.model._mm_immutable_training_plan = next_plan
        return {
            "changed": True,
            "block_key": key,
            "resident": want_resident,
            "resident_bytes": self.residency.resident_bytes(),
            "delta": delta,
            "plan": next_plan,
        }

    def next_training_promotion_bytes(self) -> int:
        current = self._sources.plan or self.residency.plan
        if current.phase != self.TRAIN:
            return 0
        protected = frozenset(
            (str(block), str(leaf))
            for block, leaf in getattr(
                self.model,
                "_mm_immutable_protected_training_leaf_keys",
                (),
            )
        )
        candidates = []
        for abi in self._block_abis:
            keys = tuple((abi.block_key, leaf) for leaf in abi.leaf_names)
            if any(key in current.resident_leaf_keys for key in keys):
                continue
            if any(key in protected for key in keys):
                continue
            record = self.residency.arena.block_record(abi.block_key)
            candidates.append(int(record.committed_bytes))
        return min(candidates, default=0)
    def increase_training_residency(
        self,
        available_growth_bytes: int,
        *,
        max_blocks: int = 1,
    ) -> dict:
        available = max(0, int(available_growth_bytes))
        current = self._sources.plan or self.residency.plan
        if current.phase != self.TRAIN:
            raise ImmutableRuntimeError(
                f"training_residency_growth_requires_train:{current.phase}"
            )

        protected = frozenset(
            (str(block), str(leaf))
            for block, leaf in getattr(
                self.model,
                "_mm_immutable_protected_training_leaf_keys",
                (),
            )
        )
        candidates = []
        for order, abi in enumerate(self._block_abis):
            keys = tuple((abi.block_key, leaf) for leaf in abi.leaf_names)
            resident_count = sum(
                key in current.resident_leaf_keys
                for key in keys
            )
            if resident_count:
                # Initial and controller plans are whole-block. Preserve
                # source-table support for partial plans without growing them.
                continue
            if any(key in protected for key in keys):
                continue
            record = self.residency.arena.block_record(abi.block_key)
            candidates.append(
                (
                    int(record.committed_bytes),
                    order,
                    abi.block_key,
                    keys,
                )
            )

        candidates.sort(key=lambda item: (item[0], item[1]))
        added = []
        predicted = 0
        limit = max(0, int(max_blocks))
        for nbytes, _order, _block_key, keys in candidates:
            if limit and len(added) >= limit:
                break
            if predicted + nbytes > available:
                continue
            added.append((nbytes, keys))
            predicted += nbytes

        previous_plan = current
        if added:
            next_keys = set(current.resident_leaf_keys)
            for _nbytes, keys in added:
                next_keys.update(keys)
            next_plan = ResidencyPlan.build(self.TRAIN, next_keys)
            self.set_residency_plan(next_plan)
            self.model._mm_immutable_training_plan = next_plan
        else:
            next_plan = current

        added_keys = tuple(
            sorted(
                key
                for _nbytes, keys in added
                for key in keys
            )
        )
        actual_growth = sum(
            self.residency.resident_leaf_bytes(key)
            for key in added_keys
        )
        return {
            "available_growth_bytes": available,
            "predicted_growth_bytes": int(predicted),
            "actual_growth_bytes": int(actual_growth),
            "added_leaf_keys": added_keys,
            "added_blocks": tuple(sorted({key[0] for key in added_keys})),
            "previous_plan": previous_plan,
            "plan": next_plan,
        }
    def reduce_training_residency(self, required_relief_bytes: int) -> dict:
        requested = max(0, int(required_relief_bytes))
        current = self._sources.plan or self.residency.plan
        if current.phase != self.TRAIN:
            raise ImmutableRuntimeError(f"training_residency_reduction_requires_train:{current.phase}")

        protected = frozenset(
            (str(block), str(leaf))
            for block, leaf in getattr(
                self.model,
                "_mm_immutable_protected_training_leaf_keys",
                (),
            )
        )
        candidates = []
        for abi in self._block_abis:
            keys = tuple((abi.block_key, leaf) for leaf in abi.leaf_names)
            resident_keys = tuple(key for key in keys if key in current.resident_leaf_keys)
            if not resident_keys or any(key in protected for key in keys):
                continue
            nbytes = sum(self.residency.resident_leaf_bytes(key) for key in resident_keys)
            candidates.append((nbytes, abi.block_key, resident_keys))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        removed = []
        relieved = 0
        for nbytes, _block_key, keys in candidates:
            if relieved >= requested:
                break
            removed.extend(keys)
            relieved += nbytes

        if removed:
            next_keys = set(current.resident_leaf_keys) - set(removed)
            next_plan = ResidencyPlan.build(self.TRAIN, next_keys)
            self.set_residency_plan(next_plan)
            self.model._mm_immutable_training_plan = next_plan
        else:
            next_plan = current

        return {
            "requested_relief_bytes": requested,
            "relieved_bytes": int(relieved),
            "removed_leaf_keys": tuple(sorted(removed)),
            "removed_blocks": tuple(sorted({key[0] for key in removed})),
            "remaining_resident_bytes": self.residency.resident_bytes(),
            "plan": next_plan,
        }

    def full_model_resident_bytes(self) -> int:
        """Bytes to hold every canonical block resident (the 'want' figure)."""
        arena = self.residency.arena
        total = 0
        for block_key in arena.block_keys():
            record = arena.block_record(block_key)
            if record is not None:
                total += int(record.committed_bytes)
        return total

    def report_foreign_vram_once(self, device, *, phase: str, working_reserve_bytes: int) -> None:
        """Phase-start external-VRAM check for callers that know their reserve.

        ``have`` is the residency budget: what we already hold, plus whatever is
        left on the card once the activation working set is reserved.
        """
        if self._foreign_vram_checked:
            return
        self._foreign_vram_checked = True
        if torch.device(device).type != "cuda":
            return
        try:
            free_bytes = vram_budget.device_free_bytes(device)
            reserved_bytes = int(torch.cuda.memory_reserved(device))
            allocated_bytes = int(torch.cuda.memory_allocated(device))
        except Exception:
            return
        reclaimable = max(0, reserved_bytes - allocated_bytes)
        have = self.residency.resident_bytes() + max(
            0, free_bytes + reclaimable - max(0, int(working_reserve_bytes))
        )
        self._report_foreign_vram(
            device,
            phase=phase,
            reserved_bytes=reserved_bytes,
            have_bytes=have,
        )

    def _report_foreign_vram(self, device, *, phase: str, reserved_bytes, have_bytes) -> None:
        """Say so when ANOTHER tenant on the GPU is what forces us to stream.

        Without this the failure is invisible: residency silently shrinks, the
        step time multiplies, and nothing in the log points at the real cause
        (a leftover job, a ComfyUI server, a game). Fires once per phase.
        """
        try:
            free_bytes, total_bytes = vram_budget.device_mem_info(device)
        except Exception:
            return
        report = vram_budget.assess_foreign_vram(
            total_bytes=total_bytes,
            free_bytes=free_bytes,
            torch_reserved_bytes=reserved_bytes,
            want_bytes=self.full_model_resident_bytes(),
            have_bytes=have_bytes,
        )
        message = vram_budget.format_foreign_vram_warning(report, phase=phase)
        if message:
            print(message)

    def activate_sampling_image(
        self,
        *,
        shape_key: tuple,
        cold_working_bytes: int,
        fixed_working_bytes: int | None,
        cold_floor_bytes: int,
        hot_floor_bytes: int,
        measured_pad_bytes: int = 256 * 1024**2,
        measured_floor_bytes: int = 512 * 1024**2,
    ) -> ImmutableProgram:
        device = self.residency.device
        if device.type != "cuda":
            self.set_residency_plan(self.sampling_fallback_plan)
            return self.program(self.SAMPLE)

        learned = int(self._sampling_working_bytes.get(shape_key, 0))
        if fixed_working_bytes is not None:
            working_bytes = max(0, int(fixed_working_bytes))
            floor_bytes = max(0, int(cold_floor_bytes))
            reserve_source = "fixed"
        elif learned > 0:
            working_bytes = max(
                int(measured_floor_bytes),
                learned + int(measured_pad_bytes),
            )
            floor_bytes = max(0, int(hot_floor_bytes))
            reserve_source = "measured"
        else:
            working_bytes = max(0, int(cold_working_bytes))
            floor_bytes = max(0, int(cold_floor_bytes))
            reserve_source = "cold"

        # Physical free across ALL processes (NVML-backed). mem_get_info would
        # over-report here whenever anything else is on the card -- a game, a
        # ComfyUI server, an orphaned job -- and we would size residency against
        # VRAM that does not exist, then page silently at ~4x the step time.
        free_bytes = vram_budget.device_free_bytes(device)
        allocated_bytes = torch.cuda.memory_allocated(device)
        reserved_bytes = torch.cuda.memory_reserved(device)
        reclaimable_cache = max(0, reserved_bytes - allocated_bytes)
        current_sidecars = self.residency.resident_bytes()
        resident_budget = max(
            0,
            current_sidecars + int(free_bytes) + reclaimable_cache - working_bytes - floor_bytes,
        )
        if not self._foreign_vram_checked:
            self._foreign_vram_checked = True
            self._report_foreign_vram(
                device,
                phase="sampling",
                reserved_bytes=reserved_bytes,
                have_bytes=resident_budget,
            )

        current_plan = self._sources.plan or self.residency.plan
        plan = ResidencyPlan.fit_whole_blocks(
            self.residency.arena,
            resident_budget,
            phase=self.SAMPLE,
            prefer_resident_keys=current_plan.resident_leaf_keys,
        )
        self.set_residency_plan(plan)

        torch.cuda.synchronize(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        self._sampling_baseline = {
            "shape_key": shape_key,
            "allocated": baseline_allocated,
            "reserved": baseline_reserved,
            "working_bytes": working_bytes,
            "floor_bytes": floor_bytes,
            "source": reserve_source,
        }

        print(
            "[MemoryManager] immutable sampling layout: "
            f"source={reserve_source} "
            f"working={working_bytes / 1024**3:.2f} GiB "
            f"floor={floor_bytes / 1024**3:.2f} GiB "
            f"sidecars={self.residency.resident_bytes() / 1024**3:.2f} GiB "
            f"device_free={vram_budget.device_free_bytes(device) / 1024**3:.2f} GiB "
            f"plan={plan.fingerprint}"
        )
        return self.program(self.SAMPLE)

    def finish_sampling_image(self, *, shape_key: tuple) -> int:
        baseline = self._sampling_baseline
        if baseline is None or baseline["shape_key"] != shape_key:
            return 0

        device = self.residency.device
        torch.cuda.synchronize(device)
        allocated_peak = torch.cuda.max_memory_allocated(device)
        reserved_peak = torch.cuda.max_memory_reserved(device)
        allocated_growth = max(
            0,
            allocated_peak - int(baseline["allocated"]),
        )
        reserved_growth = max(
            0,
            reserved_peak - int(baseline["reserved"]),
        )
        observed = max(allocated_growth, reserved_growth)
        previous = int(self._sampling_working_bytes.get(shape_key, 0))
        self._sampling_working_bytes[shape_key] = max(previous, observed)
        self._sampling_baseline = None

        print(
            "[MemoryManager] immutable sampling measurement: "
            f"shape={shape_key} "
            f"observed_working={observed / 1024**3:.2f} GiB "
            f"learned={self._sampling_working_bytes[shape_key] / 1024**3:.2f} GiB"
        )
        return observed

    def run(self, combined, tvec, freqs, mask):
        self._require_finalized()
        if self._active_token is None:
            raise ImmutableRuntimeError("immutable_execution_not_active")
        if self._sources.plan is None:
            raise ImmutableRuntimeError(
                "no_residency_source_table: publish a plan before execution"
            )
        mode = self._active_mode
        # The active phase -- not grad mode -- selects the program. A training
        # step legitimately contains no-grad forwards (diff-output preservation
        # runs a prior prediction with the network detached), and those stay on
        # the TRAIN program: same residency plan, same fetch policy, and the
        # train block fn is already no-grad safe. The reverse is still a bug:
        # a grad-enabled call inside a sampling phase would build a graph the
        # forward-only program never planned working memory for.
        if mode == self.SAMPLE and torch.is_grad_enabled():
            raise ImmutableRuntimeError(
                f"immutable_execution_mode_mismatch:"
                f"active={self.SAMPLE}:call={self.TRAIN}"
            )
        return self._programs[mode].trunk(combined, tvec, freqs, mask)

    def close(self) -> None:
        if self._sources.active_executions:
            raise ImmutableRuntimeError("cannot_close_during_execution")
        self._sources.clear()
        self._programs.clear()
        self._block_fns.clear()
        self._block_kernels.clear()

def prepare_immutable_runtime(
    transformer,
    residency: ResidencyState,
    *,
    architecture_adapter,
    depth: int = 2,
    compile_blocks: bool = True,
    compile_dynamic: bool | None = True,
    compile_dynamic_hints: tuple[tuple[int, int | None, int | None], ...] = (),
) -> ImmutableTransformerRuntime:
    existing = getattr(transformer, "_immutable_runtime", None)
    if existing is not None:
        if (
            existing.residency is residency
            and existing.architecture_adapter is architecture_adapter
        ):
            return existing
        raise ImmutableRuntimeError("immutable_runtime_already_prepared")

    runtime = ImmutableTransformerRuntime(
        transformer,
        residency,
        architecture_adapter=architecture_adapter,
        depth=depth,
        compile_blocks=compile_blocks,
        compile_dynamic=compile_dynamic,
        compile_dynamic_hints=compile_dynamic_hints,
    )
    transformer._immutable_runtime = runtime
    return runtime
