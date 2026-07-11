"""Krea2 functional adapters for immutable-arena residency plans.

``KreaImmutableArenaAdapter`` is the Slice 4 eager path (one adapter per
residency-plan fingerprint). ``KreaImmutablePlanExecutor`` is the Slice 5
compiled path: separate train/sample callables over ONE canonical arena,
keyed by an execution/residency fingerprint (plan Invariant 8), with
boundary phase switching that never rebuilds, repoints, or re-registers
host storage, plus a prebuilt all-streamed sampling fallback plan.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from toolkit.memory_management import pin_manager
from toolkit.memory_management.ingraph_stream import (
    _flatten_leaves,
    checkpoint_recompute_context,
    compiled_checkpoint_context,
    configure_fetch_runtime,
    free_on_backward,
    in_recompute,
)
from toolkit.memory_management.residency import ResidencyPlan, ResidencyState
from toolkit.memory_management.transfer_plan import (
    BlockTransferPlan,
    build_transfer_plan,
)


class KreaImmutableArenaError(RuntimeError):
    pass


@dataclass(frozen=True)
class KreaBlockSourcePlan:
    block_key: str
    leaf_names: tuple[str, ...]
    transfer: BlockTransferPlan | None
    ranges: torch.Tensor | None
    fp8_flags: tuple[bool, ...]


def build_block_source_plan(model, residency: ResidencyState, index: int) -> KreaBlockSourcePlan:
    """Turn one block's canonical record + the CURRENT residency plan into a
    static source plan: which leaves stream (and through which coalesced
    ranges) and which read from sidecars. Shared by the Slice 4 eager
    adapter and the Slice 5 compiled executor."""
    block_key = f"blocks.{index}"
    record = residency.arena.block_record(block_key)
    if record is None:
        raise KreaImmutableArenaError(f"missing_canonical_block:{block_key}")
    expected = tuple(name for name, _module in model._block_linear_entries(
        model.blocks[index]
    ))
    if record.leaf_names != expected:
        raise KreaImmutableArenaError(
            f"canonical_leaf_order_mismatch:{block_key}:"
            f"expected={expected}:actual={record.leaf_names}"
        )
    streamed = residency.streamed_leaf_names(block_key)
    transfer = build_transfer_plan(record, streamed) if streamed else None
    ranges = None if transfer is None else transfer.ranges_tensor()
    return KreaBlockSourcePlan(
        block_key=block_key,
        leaf_names=record.leaf_names,
        transfer=transfer,
        ranges=ranges,
        fp8_flags=tuple(spec.fp8_qualifies for spec in record.pack.linears),
    )


def _resident_args(sidecar, kind: str):
    leaves = _flatten_leaves(sidecar.weight)
    if kind == "float" and len(leaves) == 1:
        return leaves[0], sidecar.bias, None
    if kind == "fp8_rowwise" and len(leaves) == 2:
        return leaves[0], sidecar.bias, leaves[1]
    raise KreaImmutableArenaError(
        f"resident_sidecar_layout_mismatch:{sidecar.key[0]}.{sidecar.key[1]}"
    )


class KreaImmutableArenaAdapter:
    """Run Krea2 blocks from resident sidecars or compact fetched views.

    This adapter is eager by design (Slice 4). A single instance captures one
    immutable residency-plan fingerprint. Reconcile to another plan and build a
    new adapter; phase-callable caching belongs to Slice 5.
    """

    def __init__(
        self,
        model,
        residency: ResidencyState,
        *,
        loras_by_block=None,
        lora_multiplier=None,
        depth: int = 2,
    ) -> None:
        self.model = model
        self.residency = residency
        self.plan_fingerprint = residency.plan.fingerprint
        self.loras_by_block = dict(loras_by_block or {})
        self.lora_multiplier = lora_multiplier
        self.block_plans = tuple(
            build_block_source_plan(model, residency, index)
            for index in range(len(model.blocks))
        )
        configure_fetch_runtime(depth=depth)

    def _assert_current(self) -> None:
        if self.residency.plan.fingerprint != self.plan_fingerprint:
            raise KreaImmutableArenaError(
                "residency_plan_changed: rebuild the eager adapter for the new phase"
            )

    def _leaf_args(self, block_plan: KreaBlockSourcePlan, compact_flat=None):
        record = self.residency.arena.block_record(block_plan.block_key)
        args = []
        for leaf_name in block_plan.leaf_names:
            spec = record.leaf_spec(leaf_name)
            sidecar = self.residency.resident_leaf((block_plan.block_key, leaf_name))
            if sidecar is not None:
                args.append(_resident_args(sidecar, spec.kind))
                continue
            if compact_flat is None or block_plan.transfer is None:
                raise KreaImmutableArenaError(
                    f"missing_streamed_source:{block_plan.block_key}.{leaf_name}"
                )
            transfer = block_plan.transfer
            weight = transfer.compact_leaf_view(compact_flat, leaf_name, "weight")
            bias = (
                None
                if spec.bias is None
                else transfer.compact_leaf_view(compact_flat, leaf_name, "bias")
            )
            scale = (
                None
                if spec.weight_scale is None
                else transfer.compact_leaf_view(
                    compact_flat, leaf_name, "weight_scale"
                )
            )
            args.append((weight, bias, scale))
        return tuple(args)

    def _lora_args(self, index: int):
        loras = self.loras_by_block.get(index)
        if not loras:
            return None
        return self.model._block_lora_tuple(loras, self.lora_multiplier)

    def forward_block(self, index, x, tvec, freqs, mask, *, checkpointed=False):
        self._assert_current()
        block_plan = self.block_plans[index]
        block = self.model.blocks[index]
        token = None
        compact = None
        if block_plan.transfer is not None:
            token = torch.ops.mm.fetch_start_multi(
                self.residency.arena.block_record(block_plan.block_key).host_flat,
                block_plan.ranges,
                block_plan.transfer.compact_nbytes,
            )
            compact = torch.ops.mm.fetch_wait(
                token, block_plan.transfer.compact_nbytes
            )
            if torch.is_grad_enabled():
                x = free_on_backward(x, token)

        out = block.forward_streamed(
            x,
            tvec,
            freqs,
            mask,
            self._leaf_args(block_plan, compact),
            block_plan.fp8_flags,
            training=torch.is_grad_enabled(),
            loras=self._lora_args(index),
        )
        if token is not None:
            if not torch.is_grad_enabled():
                torch.ops.mm.fetch_free(token)
            elif checkpointed and not in_recompute():
                # First-pass checkpoint tensors are discarded. Recompute's
                # ticket is instead freed by free_on_backward at last read.
                torch.ops.mm.fetch_free(token)
        return out

    def forward_blocks(self, combined, tvec, freqs, mask):
        self._assert_current()
        cutoff = len(self.model.blocks) - self.model._checkpoint_keep_last
        for index in range(len(self.model.blocks)):
            if (
                self.model.gradient_checkpointing
                and torch.is_grad_enabled()
                and index < cutoff
            ):
                def block_fn(x, vec, block_freqs, block_mask, block_index=index):
                    return self.forward_block(
                        block_index,
                        x,
                        vec,
                        block_freqs,
                        block_mask,
                        checkpointed=True,
                    )

                combined = checkpoint(
                    block_fn,
                    combined,
                    tvec,
                    freqs,
                    mask,
                    use_reentrant=False,
                    context_fn=checkpoint_recompute_context,
                )
            else:
                combined = self.forward_block(index, combined, tvec, freqs, mask)
        return combined


@dataclass(frozen=True)
class KreaPlanProgram:
    """One phase's executable over the immutable arena.

    ``fingerprint`` is the execution/residency layout fingerprint of plan
    Invariant 8 -- independent of pinnedness, host-flat identity, and pack
    object identity. ``trunk`` is the (possibly compiled) callable; its
    closures capture the CURRENT sidecars and are rebuilt at every
    activation, so a demoted sidecar is never kept alive by a stale
    program (Dynamo reuses the underlying graphs across closure rebuilds
    -- measured, plan test 14)."""

    mode: str
    fingerprint: str
    residency_fingerprint: str
    trunk: object
    block_plans: tuple


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
    """Invariant 8 fingerprint: streamed/resident leaf sets per block,
    transfer spans + destination layout (via each BlockTransferPlan's own
    fingerprint), checkpoint mode, quant identity (fp8 flags), and ring
    depth. Deliberately EXCLUDES host-flat identity, pinnedness, and pack
    object identity."""
    per_block = tuple(
        (
            plan.block_key,
            "resident" if plan.transfer is None else plan.transfer.fingerprint,
            plan.fp8_flags,
        )
        for plan in block_plans
    )
    lora_shape = tuple(
        (index, tuple(sorted(entries)))
        for index, entries in sorted((loras_by_block or {}).items())
    )
    source = repr(
        (
            str(mode),
            residency_plan.fingerprint,
            per_block,
            int(depth),
            str(checkpoint_mode),
            lora_shape,
            bool(has_multiplier),
        )
    )
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


class KreaImmutablePlanExecutor:
    """Slice 5: compiled train/sample residency plans + phase switching.

    Owns one immutable canonical arena (through its ResidencyState) and, per
    phase, a compiled callable specialized on that phase's residency/layout
    plan. Boundary switching = ``activate(mode, plan)``: reconcile sidecars,
    rebuild closures, reuse compiled graphs. The arena's host flats,
    registrations, and pin ledger are asserted UNCHANGED at every activation
    (plan Endpoint Acceptance Criteria) -- a boundary never rebuilds or
    repoints host storage.

    Recompile budget: distinct plan fingerprints (and resolution buckets
    under ``dynamic=False``) are intentional, bounded Dynamo cache entries
    on the two trunk code objects. The old ``raise_dynamo_recompile_limit``
    workaround is NOT used here: I2 traced the boundary failure to the
    legacy layout guard this path deletes, not to genuine recompiles.
    """

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
        compile_trunks: bool = True,
    ) -> None:
        self.model = model
        self.residency = residency
        self.loras_by_block = dict(loras_by_block or {})
        self.lora_multiplier = lora_multiplier
        self.depth = max(1, int(depth))
        self.compile_trunks = bool(compile_trunks)
        self._programs: dict[str, KreaPlanProgram | None] = {
            self.TRAIN: None,
            self.SAMPLE: None,
        }
        self._seen_fingerprints: set[str] = set()
        self.stats = {"activations": 0, "plan_builds": 0, "plan_reuse": 0}
        # Emergency all-streamed sampling plan, prebuilt so a mid-denoise
        # demotion never has to build host-side state (Invariant 8): zero
        # resident leaves, every block a single-copy fully-streamed fetch.
        self.sampling_fallback_plan = ResidencyPlan.build("sample_fallback", ())
        self._arena_signature = self._capture_arena_signature()
        configure_fetch_runtime(depth=self.depth)

    # -- arena stability (Endpoint Acceptance Criteria) ---------------------

    def _capture_arena_signature(self):
        arena = self.residency.arena
        return (
            tuple(
                (block_key, arena.block_record(block_key).host_flat.data_ptr())
                for block_key in arena.block_keys()
            ),
            pin_manager.total_pinned_bytes(),
            tuple(sorted(pin_manager.pinned_bytes_by_kind().items())),
        )

    def _assert_arena_stable(self, where: str) -> None:
        current = self._capture_arena_signature()
        if current != self._arena_signature:
            raise KreaImmutableArenaError(
                f"arena_mutated_at_boundary:{where}: canonical host flats, "
                "registrations, or the pin ledger changed across a phase "
                "boundary -- residency transitions must never touch host "
                f"storage (expected {self._arena_signature!r}, got {current!r})"
            )

    # -- program construction ------------------------------------------------

    def _hoisted_leaf_entries(self, block_plan: KreaBlockSourcePlan):
        """Per-leaf source descriptors with resident sidecars resolved NOW
        (trace-time constants for the compiled trunk). Streamed leaves stay
        symbolic: their views are sliced from the fetched compact flat inside
        the traced fn."""
        record = self.residency.arena.block_record(block_plan.block_key)
        entries = []
        for leaf_name in block_plan.leaf_names:
            spec = record.leaf_spec(leaf_name)
            sidecar = self.residency.resident_leaf(
                (block_plan.block_key, leaf_name)
            )
            if sidecar is not None:
                entries.append((None, _resident_args(sidecar, spec.kind)))
                continue
            if block_plan.transfer is None:
                raise KreaImmutableArenaError(
                    f"missing_streamed_source:{block_plan.block_key}.{leaf_name}"
                )
            entries.append(
                (
                    (
                        leaf_name,
                        spec.bias is not None,
                        spec.weight_scale is not None,
                    ),
                    None,
                )
            )
        return tuple(entries)

    def _make_block_fn(self, index: int, mode: str, block_plan: KreaBlockSourcePlan):
        block = self.model.blocks[index]
        entries = self._hoisted_leaf_entries(block_plan)
        transfer = block_plan.transfer
        fp8_flags = block_plan.fp8_flags
        loras = self.loras_by_block.get(index, {})
        multiplier = self.lora_multiplier
        block_lora_tuple = type(self.model)._block_lora_tuple
        training = mode == self.TRAIN

        def assemble(compact_flat):
            args = []
            for streamed, resident in entries:
                if streamed is None:
                    args.append(resident)
                    continue
                leaf_name, has_bias, has_scale = streamed
                weight = transfer.compact_leaf_view(compact_flat, leaf_name, "weight")
                bias = (
                    transfer.compact_leaf_view(compact_flat, leaf_name, "bias")
                    if has_bias
                    else None
                )
                scale = (
                    transfer.compact_leaf_view(
                        compact_flat, leaf_name, "weight_scale"
                    )
                    if has_scale
                    else None
                )
                args.append((weight, bias, scale))
            return tuple(args)

        if transfer is None:
            leaf_args = assemble(None)

            def resident_fn(x, tvec, freqs, mask):
                lora_args = block_lora_tuple(loras, multiplier) if loras else None
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

            return resident_fn

        host = self.residency.arena.block_record(block_plan.block_key).host_flat
        ranges = block_plan.ranges
        nbytes = int(transfer.compact_nbytes)

        if training:

            def train_fn(x, tvec, freqs, mask):
                compiling = torch.compiler.is_compiling()
                if compiling:
                    token = torch.ops.mm.fetch_start_multi_after(
                        host, ranges, nbytes, x
                    )
                else:
                    token = torch.ops.mm.fetch_start_multi(host, ranges, nbytes)
                flat = torch.ops.mm.fetch_wait(token, nbytes)
                leaf_args = assemble(flat)
                lora_args = block_lora_tuple(loras, multiplier) if loras else None
                if torch.is_grad_enabled():
                    # Saved-token swap: backward frees the recompute generation.
                    x = free_on_backward(x, token)
                out = block.forward_streamed(
                    x,
                    tvec,
                    freqs,
                    mask,
                    leaf_args,
                    fp8_flags,
                    training=True,
                    loras=lora_args,
                )
                if not in_recompute():
                    if compiling:
                        torch.ops.mm.fetch_free_after(token, out)
                    else:
                        torch.ops.mm.fetch_free(token)
                return out

            return train_fn

        def sample_fn(x, tvec, freqs, mask):
            token = torch.ops.mm.fetch_start_multi_after(host, ranges, nbytes, x)
            flat = torch.ops.mm.fetch_wait(token, nbytes)
            leaf_args = assemble(flat)
            lora_args = block_lora_tuple(loras, multiplier) if loras else None
            out = block.forward_streamed(
                x,
                tvec,
                freqs,
                mask,
                leaf_args,
                fp8_flags,
                loras=lora_args,
            )
            torch.ops.mm.fetch_free_after(token, out)
            return out

        return sample_fn

    @staticmethod
    def _train_trunk(block_fns):
        def immutable_train_trunk(combined, tvec, freqs, mask):
            # Select the context fn HERE, not via a late-binding wrapper:
            # the checkpoint HOP calls context_fn() outside the compiling
            # frame, where a wrapper's own is_compiling() check would pick
            # the eager (non-TorchDispatchMode) contexts and fail the HOP's
            # assertion. Same shape as _ingraph_training_trunk.
            context_fn = (
                compiled_checkpoint_context
                if torch.compiler.is_compiling()
                else checkpoint_recompute_context
            )
            for fn in block_fns:
                combined = checkpoint(
                    fn,
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
            for fn in block_fns:
                combined = fn(combined, tvec, freqs, mask)
            return combined

        return immutable_sample_trunk

    # -- phase switching -----------------------------------------------------

    def activate(self, mode: str, plan: ResidencyPlan) -> KreaPlanProgram:
        """Reconcile residency to ``plan`` and (re)build ``mode``'s callable.

        Same-fingerprint reactivation rebuilds only the Python closures;
        Dynamo reuses the previously compiled graphs (identity of host flats
        and sidecar tensors is not guarded on -- measured). The opposite
        mode's program is dropped when it no longer matches the live
        residency plan, so its closures cannot keep demoted sidecars alive
        or run against sidecars that were rebuilt since.
        """
        if mode not in (self.TRAIN, self.SAMPLE):
            raise KreaImmutableArenaError(f"unknown_execution_mode:{mode}")
        self._assert_arena_stable(f"pre_activate:{mode}")
        if self.residency.plan.fingerprint != plan.fingerprint:
            self.residency.reconcile(plan)
        self._assert_arena_stable(f"post_reconcile:{mode}")

        block_plans = tuple(
            build_block_source_plan(self.model, self.residency, index)
            for index in range(len(self.model.blocks))
        )
        checkpoint_mode = "full" if mode == self.TRAIN else "none"
        fingerprint = build_execution_fingerprint(
            mode,
            plan,
            block_plans,
            depth=self.depth,
            checkpoint_mode=checkpoint_mode,
            loras_by_block=self.loras_by_block,
            has_multiplier=self.lora_multiplier is not None,
        )
        block_fns = tuple(
            self._make_block_fn(index, mode, block_plan)
            for index, block_plan in enumerate(block_plans)
        )
        trunk = (
            self._train_trunk(block_fns)
            if mode == self.TRAIN
            else self._sample_trunk(block_fns)
        )
        if self.compile_trunks:
            from toolkit.memory_management.ingraph_stream_scheduling import (
                install_ordering_pass,
            )

            install_ordering_pass()
            trunk = torch.compile(
                trunk, fullgraph=True, dynamic=False, mode="default"
            )
        program = KreaPlanProgram(
            mode=mode,
            fingerprint=fingerprint,
            residency_fingerprint=plan.fingerprint,
            trunk=trunk,
            block_plans=block_plans,
        )
        self._programs[mode] = program
        other = self.SAMPLE if mode == self.TRAIN else self.TRAIN
        other_program = self._programs[other]
        if (
            other_program is not None
            and other_program.residency_fingerprint != plan.fingerprint
        ):
            self._programs[other] = None
        self.stats["activations"] += 1
        if fingerprint in self._seen_fingerprints:
            self.stats["plan_reuse"] += 1
        else:
            self._seen_fingerprints.add(fingerprint)
            self.stats["plan_builds"] += 1
        return program

    def activate_sampling_fallback(self) -> KreaPlanProgram:
        """Switch to the prebuilt all-streamed sampling plan (emergency
        demotion path): every sidecar is released, every block fetches its
        full canonical span in one copy, and host storage is untouched."""
        return self.activate(self.SAMPLE, self.sampling_fallback_plan)

    def program(self, mode: str) -> KreaPlanProgram | None:
        return self._programs.get(mode)

    # -- execution -----------------------------------------------------------

    def run(self, combined, tvec, freqs, mask):
        """Dispatch to the grad-mode-appropriate callable, failing closed if
        that phase was not activated for the CURRENT residency plan."""
        mode = self.TRAIN if torch.is_grad_enabled() else self.SAMPLE
        program = self._programs.get(mode)
        if program is None:
            raise KreaImmutableArenaError(
                f"no_active_plan:{mode}: call activate({mode!r}, plan) at the "
                "phase boundary before running"
            )
        if program.residency_fingerprint != self.residency.plan.fingerprint:
            raise KreaImmutableArenaError(
                f"residency_plan_changed:{mode}: the live residency plan no "
                "longer matches this phase's program -- re-activate at the "
                "boundary"
            )
        return program.trunk(combined, tvec, freqs, mask)
