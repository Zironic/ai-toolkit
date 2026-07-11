"""Compile-neutral execution over an immutable canonical weight arena.

Residency is runtime source state. The eager train and sample programs and the
compiled pure-math block kernels are created once per executor. Publishing a
new ResidencyPlan only reconciles device sidecars and swaps source snapshots.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

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


class KreaImmutableArenaError(RuntimeError):
    pass


@dataclass(frozen=True)
class KreaBlockABI:
    """Structural block information that cannot change during executor life."""

    block_key: str
    leaf_names: tuple[str, ...]
    fp8_flags: tuple[bool, ...]
    leaf_layout: tuple[tuple[str, tuple[int, ...], str, bool, bool], ...]


@dataclass(frozen=True)
class KreaBlockSourceSnapshot:
    """Plan-dependent sources for one canonical block."""

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
            raise KreaImmutableArenaError(
                f"missing_canonical_block:{self.block_key}"
            )

        args = []
        for leaf_name in self.leaf_names:
            spec = record.leaf_spec(leaf_name)
            if leaf_name in self.resident_leaf_names:
                sidecar = residency.resident_leaf((self.block_key, leaf_name))
                if sidecar is None:
                    raise KreaImmutableArenaError(
                        f"missing_resident_source:{self.block_key}.{leaf_name}"
                    )
                args.append(_resident_args(sidecar, spec.kind))
                continue

            if compact_flat is None or self.transfer is None:
                raise KreaImmutableArenaError(
                    f"missing_streamed_source:{self.block_key}.{leaf_name}"
                )

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
    raise KreaImmutableArenaError(
        f"resident_sidecar_layout_mismatch:{sidecar.key[0]}.{sidecar.key[1]}"
    )


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


def build_block_abi(model, residency: ResidencyState, index: int) -> KreaBlockABI:
    block_key = f"blocks.{index}"
    record = residency.arena.block_record(block_key)
    if record is None:
        raise KreaImmutableArenaError(f"missing_canonical_block:{block_key}")

    expected = tuple(
        name
        for name, _module in model._block_linear_entries(model.blocks[index])
    )
    if record.leaf_names != expected:
        raise KreaImmutableArenaError(
            f"canonical_leaf_order_mismatch:{block_key}:"
            f"expected={expected}:actual={record.leaf_names}"
        )

    return KreaBlockABI(
        block_key=block_key,
        leaf_names=record.leaf_names,
        fp8_flags=tuple(spec.fp8_qualifies for spec in record.pack.linears),
        leaf_layout=_leaf_layout(record),
    )


def build_block_source_snapshot(
    residency: ResidencyState,
    plan: ResidencyPlan,
    abi: KreaBlockABI,
) -> KreaBlockSourceSnapshot:
    record = residency.arena.block_record(abi.block_key)
    if record is None:
        raise KreaImmutableArenaError(
            f"missing_canonical_block:{abi.block_key}"
        )

    resident = plan.resident_in_block(abi.block_key)
    unknown = resident - frozenset(abi.leaf_names)
    if unknown:
        leaf_name = sorted(unknown)[0]
        raise KreaImmutableArenaError(
            f"unknown_residency_leaf:{abi.block_key}.{leaf_name}"
        )

    streamed = tuple(
        leaf_name
        for leaf_name in abi.leaf_names
        if leaf_name not in resident
    )
    transfer = build_transfer_plan(record, streamed) if streamed else None
    ranges = None if transfer is None else transfer.ranges_tensor()
    return KreaBlockSourceSnapshot(
        block_key=abi.block_key,
        leaf_names=abi.leaf_names,
        resident_leaf_names=resident,
        transfer=transfer,
        ranges=ranges,
    )


def build_block_source_plan(
    model,
    residency: ResidencyState,
    index: int,
) -> KreaBlockSourceSnapshot:
    abi = build_block_abi(model, residency, index)
    return build_block_source_snapshot(residency, residency.plan, abi)


class KreaRuntimeSourceTable:
    """Atomically published per-block source snapshots."""

    def __init__(self, residency: ResidencyState, block_abis) -> None:
        self.residency = residency
        self.block_abis = tuple(block_abis)
        self._generation = 0
        self._active_executions = 0
        self._snapshots: tuple[KreaBlockSourceSnapshot, ...] | None = None
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
    ) -> tuple[KreaBlockSourceSnapshot, ...]:
        return tuple(
            build_block_source_snapshot(self.residency, plan, abi)
            for abi in self.block_abis
        )

    def begin_execution(self) -> int:
        if self._snapshots is None:
            raise KreaImmutableArenaError("no_residency_source_table")
        self._active_executions += 1
        return self._generation

    def end_execution(self, generation: int) -> None:
        if self._active_executions <= 0:
            raise KreaImmutableArenaError("immutable_execution_not_active")
        if int(generation) != self._generation:
            raise KreaImmutableArenaError(
                "immutable_execution_generation_mismatch"
            )
        self._active_executions -= 1

    def source(self, block_index: int) -> KreaBlockSourceSnapshot:
        snapshots = self._snapshots
        if snapshots is None:
            raise KreaImmutableArenaError("no_residency_source_table")
        try:
            return snapshots[int(block_index)]
        except IndexError as error:
            raise KreaImmutableArenaError(
                f"unknown_execution_block:{block_index}"
            ) from error

    def publish(self, plan: ResidencyPlan) -> ResidencyDelta:
        if self._active_executions:
            raise KreaImmutableArenaError(
                "residency_transition_during_execution"
            )

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
            raise KreaImmutableArenaError(
                "source_table_clear_during_execution"
            )
        self._snapshots = None
        self._plan = None


class KreaImmutableArenaAdapter:
    """Eager compatibility adapter over one current residency plan."""

    def __init__(
        self,
        model,
        residency: ResidencyState,
        *,
        loras_by_block=None,
        lora_multiplier=None,
        depth: int = 2,
    ) -> None:
        self._executor = KreaImmutablePlanExecutor(
            model,
            residency,
            loras_by_block=loras_by_block,
            lora_multiplier=lora_multiplier,
            depth=depth,
            compile_blocks=False,
        )
        if residency.plan.phase == "empty":
            raise KreaImmutableArenaError(
                "eager_adapter_requires_active_residency_plan"
            )
        self._executor.set_residency_plan(residency.plan)

    def forward_blocks(self, combined, tvec, freqs, mask):
        return self._executor.run(combined, tvec, freqs, mask)


@dataclass(frozen=True)
class KreaPlanProgram:
    """One permanent eager program for one execution mode."""

    mode: str
    fingerprint: str
    trunk: object


def build_execution_fingerprint(
    mode: str,
    residency_plan: ResidencyPlan,
    block_plans,
    *,
    depth: int,
    checkpoint_mode: str,
    loras_by_block=None,
    has_multiplier: bool = False,
) -> str:
    del residency_plan
    per_block = tuple(
        (
            plan.block_key,
            tuple(plan.leaf_names),
        )
        for plan in block_plans
    )
    lora_shape = tuple(
        (index, tuple(sorted(entries)))
        for index, entries in sorted((loras_by_block or {}).items())
    )
    source = repr(
        (
            "krea-runtime-source-v1",
            str(mode),
            per_block,
            int(depth),
            str(checkpoint_mode),
            lora_shape,
            bool(has_multiplier),
        )
    )
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


class KreaImmutablePlanExecutor:
    """Permanent eager programs over mutable canonical-block source state."""

    TRAIN = "train"
    SAMPLE = "sample"

    def __init__(
        self,
        model,
        residency: ResidencyState,
        *,
        loras_by_block=None,
        lora_multiplier=None,
        depth: int = 2,
        compile_blocks: bool = True,
    ) -> None:
        self._sampling_working_bytes: dict[tuple, int] = {}
        self._sampling_baseline = None
        self.model = model
        self.residency = residency
        self.loras_by_block = dict(loras_by_block or {})
        self.lora_multiplier = lora_multiplier
        self.depth = max(1, int(depth))
        self.compile_blocks = bool(compile_blocks)
        self._arena_signature = self.residency.arena.immutable_signature()

        self._block_abis = tuple(
            build_block_abi(model, residency, index)
            for index in range(len(model.blocks))
        )
        self._sources = KreaRuntimeSourceTable(residency, self._block_abis)
        self._block_kernels: dict[tuple[str, int, tuple[bool, ...]], object] = {}

        configure_fetch_runtime(depth=self.depth)

        self._block_fns = {
            self.TRAIN: tuple(
                self._make_stable_block_fn(index, self.TRAIN)
                for index in range(len(model.blocks))
            ),
            self.SAMPLE: tuple(
                self._make_stable_block_fn(index, self.SAMPLE)
                for index in range(len(model.blocks))
            ),
        }
        self._programs = {
            self.TRAIN: self._build_program(self.TRAIN),
            self.SAMPLE: self._build_program(self.SAMPLE),
        }
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

    def begin_execution(self) -> int:
        return self._sources.begin_execution()

    def end_execution(self, generation: int) -> None:
        self._sources.end_execution(generation)

    def abort_execution(self, generation: int) -> None:
        self._sources.end_execution(generation)

    def _assert_arena_stable(self, where: str) -> None:
        current = self.residency.arena.immutable_signature()
        if current != self._arena_signature:
            raise KreaImmutableArenaError(
                f"arena_mutated_at_boundary:{where}: canonical host flats "
                "or registrations changed across a phase boundary"
            )

    def _get_block_kernel(self, index: int, mode: str, fp8_flags):
        fp8_flags = tuple(fp8_flags)
        key = (str(mode), int(index), fp8_flags)
        existing = self._block_kernels.get(key)
        if existing is not None:
            return existing

        block = self.model.blocks[index]
        training = mode == self.TRAIN

        def block_kernel(
            x,
            tvec,
            freqs,
            mask,
            leaf_args,
            lora_args,
        ):
            return block.forward_streamed(
                x,
                tvec,
                freqs,
                mask,
                leaf_args,
                fp8_flags,
                training=training,
                loras=lora_args,
            )

        kernel = block_kernel
        if self.compile_blocks:
            kernel = torch.compile(
                kernel,
                mode="default",
                fullgraph=False,
            )
        self._block_kernels[key] = kernel
        return kernel

    def _current_lora_args(self, index: int):
        loras = self.loras_by_block.get(index)
        if not loras:
            return None
        return type(self.model)._block_lora_tuple(
            loras,
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
                host = self.residency.arena.block_record(
                    source.block_key
                ).host_flat
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

    @staticmethod
    def _train_trunk(block_fns):
        def immutable_train_trunk(combined, tvec, freqs, mask):
            context_fn = (
                compiled_checkpoint_context
                if torch.compiler.is_compiling()
                else checkpoint_recompute_context
            )
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

    @staticmethod
    def _sample_trunk(block_fns):
        def immutable_sample_trunk(combined, tvec, freqs, mask):
            for block_fn in block_fns:
                combined = block_fn(combined, tvec, freqs, mask)
            return combined

        return immutable_sample_trunk

    def _build_program(self, mode: str) -> KreaPlanProgram:
        block_abis = tuple(
            KreaBlockSourceSnapshot(
                block_key=abi.block_key,
                leaf_names=abi.leaf_names,
                resident_leaf_names=frozenset(),
                transfer=None,
                ranges=None,
            )
            for abi in self._block_abis
        )
        fingerprint = build_execution_fingerprint(
            mode,
            ResidencyPlan.build("structural", ()),
            block_abis,
            depth=self.depth,
            checkpoint_mode="full" if mode == self.TRAIN else "none",
            loras_by_block=self.loras_by_block,
            has_multiplier=self.lora_multiplier is not None,
        )
        trunk = (
            self._train_trunk(self._block_fns[mode])
            if mode == self.TRAIN
            else self._sample_trunk(self._block_fns[mode])
        )
        return KreaPlanProgram(
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

    def activate(self, mode: str, plan: ResidencyPlan) -> KreaPlanProgram:
        if mode not in (self.TRAIN, self.SAMPLE):
            raise KreaImmutableArenaError(f"unknown_execution_mode:{mode}")
        self.set_residency_plan(plan)
        return self.program(mode)

    def program(self, mode: str) -> KreaPlanProgram:
        try:
            return self._programs[mode]
        except KeyError as error:
            raise KreaImmutableArenaError(
                f"unknown_execution_mode:{mode}"
            ) from error

    def activate_sampling_fallback(self) -> KreaPlanProgram:
        self.set_residency_plan(self.sampling_fallback_plan)
        return self.program(self.SAMPLE)

    def reduce_training_residency(self, required_relief_bytes: int) -> dict:
        requested = max(0, int(required_relief_bytes))
        current = self._sources.plan or self.residency.plan
        if current.phase != self.TRAIN:
            raise KreaImmutableArenaError(
                f"training_residency_reduction_requires_train:{current.phase}"
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
        for abi in self._block_abis:
            keys = tuple((abi.block_key, leaf) for leaf in abi.leaf_names)
            resident_keys = tuple(
                key for key in keys if key in current.resident_leaf_keys
            )
            if not resident_keys or any(key in protected for key in keys):
                continue
            nbytes = sum(
                self.residency.resident_leaf_bytes(key)
                for key in resident_keys
            )
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
    ) -> KreaPlanProgram:
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

        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        allocated_bytes = torch.cuda.memory_allocated(device)
        reserved_bytes = torch.cuda.memory_reserved(device)
        reclaimable_cache = max(0, reserved_bytes - allocated_bytes)
        current_sidecars = self.residency.resident_bytes()
        resident_budget = max(
            0,
            current_sidecars
            + int(free_bytes)
            + reclaimable_cache
            - working_bytes
            - floor_bytes,
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
            f"device_free={torch.cuda.mem_get_info(device)[0] / 1024**3:.2f} GiB "
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
        if self._sources.plan is None:
            raise KreaImmutableArenaError(
                "no_residency_source_table: publish a plan before execution"
            )
        mode = self.TRAIN if torch.is_grad_enabled() else self.SAMPLE
        return self._programs[mode].trunk(combined, tvec, freqs, mask)

    def close(self) -> None:
        if self._sources.active_executions:
            raise KreaImmutableArenaError(
                "cannot_close_during_execution"
            )
        self._sources.clear()
        self._programs.clear()
        self._block_fns.clear()
        self._block_kernels.clear()
