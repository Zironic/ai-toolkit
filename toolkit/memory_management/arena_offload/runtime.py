"""The arena offload runtime facade.

One object owns the arena, the residency state, the training plan, and the
immutable block executor. Model integrations and the shared trainer hold a
reference to it and nothing else -- no `_mm_*` field reads, no arena
construction, no residency manipulation.

Phase 1 is behavior-preserving: planning is still delegated to
`MemoryManager`'s smart planner (imported lazily, and only from this file, so
the seam is a single grep away). Phase 2 replaces those calls with
`arena_offload/policy.py`. Every such call below is marked PHASE-2.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from .. import allocator_cap
from ..vram_budget import apply_simulated_card
from .policy import (
    ArenaResidencyController,
    TrainingSignalWindow,
)

RUNTIME_ATTR = "_arena_offload_runtime"

GIB = 1024**3
BOOTSTRAP_MARGIN_BYTES = GIB
BOOTSTRAP_MIN_STEP = 3


class ArenaOffloadRuntime:
    """Lifecycle + execution contexts for one arena-offloaded transformer."""

    def __init__(
        self,
        model,
        *,
        device,
        adapter,
        config,
        arena,
        residency,
        executor,
        training_plan,
        smart_plan,
        canonical_modules,
    ) -> None:
        self._model = model
        self._device = device
        self._adapter = adapter
        self._config = config
        self._arena = arena
        self._residency = residency
        self._executor = executor
        self._training_plan = training_plan
        self._smart_plan = smart_plan
        self._canonical_modules = canonical_modules
        self._closed = False

        # Set by training_step(); the residency controller (git-bug 0c577ef)
        # reads these at the step boundary.
        self._last_shape_key: tuple | None = None
        self._last_step_num: int | None = None
        self._signals = TrainingSignalWindow()
        self._last_policy_error: str | None = None
        self._last_failure_event: dict | None = None
        self._policy = ArenaResidencyController()
        self._last_training_cap_target_bytes: int | None = None
        self._bootstrap_complete = False
        self._bootstrap_min_free_bytes: int | None = None
        self._bootstrap_budget_bytes = 0
        self._bootstrap_block_keys: tuple[str, ...] = ()
        self._training_fp8_restores = []
        self._training_fp8_singletons = 0
        self._sampling_fp8_singletons = 0

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def _prepare(
        cls,
        transformer,
        *,
        device,
        adapter,
        config,
        ignore_modules: Sequence[Any] | None = None,
    ) -> ArenaOffloadRuntime:
        # PHASE-2: MemoryManager is the legacy smart planner; policy.py replaces it.
        from ..canonical_arena import CanonicalArena
        from ..immutable_runtime import prepare_immutable_runtime
        from ..manager import MemoryManager
        from ..residency import ResidencyPlan, ResidencyState


        existing = getattr(transformer, RUNTIME_ATTR, None)
        if existing is not None:
            raise RuntimeError("arena_offload_already_prepared")

        # Before ANY planning reads a VRAM number: a simulated smaller card must
        # be in force for the whole run, not just the phases we remember to ask.
        apply_simulated_card(config.simulated_vram_gib, device=device)
        MemoryManager.set_wddm_cap_strict(config.wddm_cap_strict)
        allocator_cap.apply_wddm_hard_allocator_cap(
            device, config.legacy.wddm_hard_gib, log_prefix="[ArenaOffload]"
        )
        # The fp8 Linear kernels read this as a process-global. Bind it from the
        # config here so the arena path cannot disagree with what the job asked
        # for -- an unbound config field is how the flag silently went dead.
        MemoryManager.set_fp8_grad_input_enabled(config.fp8_backward)

        transformer.requires_grad_(False)

        blocks = adapter.execution_blocks(transformer)
        entries_by_block = {
            adapter.block_key(transformer, index): list(adapter.leaf_entries(block))
            for index, block in enumerate(blocks)
        }
        arena = CanonicalArena()
        arena.canonicalize(entries_by_block)

        canonical_modules = []
        for entries in entries_by_block.values():
            for _name, child in entries:
                child._mm_canonical_leaf = True
                canonical_modules.append(child)

        legacy = config.legacy
        # PHASE-2: keep_last -> pinned resident block keys.
        pinned_keys = MemoryManager.training_pinned_keys_for_keep_last(
            transformer, legacy.checkpoint_keep_last
        )
        try:
            # PHASE-2: block-key-native planning lands in policy.py.
            smart_plan = MemoryManager.attach_smart_training_immutable(
                transformer,
                device,
                canonical_modules=canonical_modules,
                working_reserve_gib=legacy.working_reserve_gib,
                wddm_margin_gib=legacy.wddm_margin_gib,
                wddm_hard_gib=legacy.wddm_hard_gib,
                ignore_modules=ignore_modules,
                pinned_resident_keys=pinned_keys,
                block_stream_only=legacy.block_stream_only,
                wddm_spill_reserve_pct=legacy.wddm_spill_reserve_pct,
                fp8_training_forward=config.fp8_forward,
                eager_promote_free_gib=legacy.eager_promote_free_gib,
                eager_promote_max_blocks=legacy.eager_promote_max_blocks,
            )
        except Exception:
            arena.release()
            for child in canonical_modules:
                if hasattr(child, "_mm_canonical_leaf"):
                    del child._mm_canonical_leaf
            raise

        residency = ResidencyState(arena, device)
        training_plan = ResidencyPlan.from_smart_plan(arena, smart_plan, phase="train")
        residency.reconcile(training_plan)

        # Phase 1 still publishes the legacy `_mm_*` fields: the immutable
        # runtime and the legacy manager both read them today. Phase 2 (policy)
        # and Phase 7 (legacy branch removal) retire them; the facade below is
        # what shared code is allowed to use in the meantime.
        transformer._mm_canonical_arena = arena
        transformer._mm_residency_state = residency
        transformer._mm_immutable_training_plan = training_plan
        must_resident_names = set(smart_plan.get("must_resident_layer_keys", ()))
        pinned_resident_blocks = set(smart_plan.get("pinned_resident_keys", ()))
        transformer._mm_immutable_protected_training_leaf_keys = frozenset(
            key
            for key in training_plan.resident_leaf_keys
            if key[0] in pinned_resident_blocks
            or f"{key[0]}.{key[1]}" in must_resident_names
        )
        transformer._mm_immutable_smart_plan = smart_plan
        transformer._mm_immutable_canonical_modules = tuple(canonical_modules)
        transformer._mm_immutable_backend = True

        executor = prepare_immutable_runtime(
            transformer,
            residency,
            architecture_adapter=adapter,
            depth=legacy.prefetch_depth,
            compile_blocks=config.compile_blocks,
            compile_dynamic=config.compile_dynamic,
            compile_dynamic_hints=config.compile_dynamic_hints,
        )

        runtime = cls(
            transformer,
            device=device,
            adapter=adapter,
            config=config,
            arena=arena,
            residency=residency,
            executor=executor,
            training_plan=training_plan,
            smart_plan=smart_plan,
            canonical_modules=tuple(canonical_modules),
        )
        setattr(transformer, RUNTIME_ATTR, runtime)
        return runtime

    # ------------------------------------------------------------------
    # read-only state
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def config(self):
        return self._config

    @property
    def block_count(self) -> int:
        return len(self._arena.block_keys())

    @property
    def finalized(self) -> bool:
        return bool(getattr(self._executor, "finalized", False))

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def set_compile_dynamic_hints(self, hints) -> None:
        """Install mark_dynamic hints on the block kernels (see ImmutableRuntime).

        The trainer derives sequence bounds from the datasets, which do not exist
        when the runtime is prepared. Must be called before the first forward.
        """
        self._executor.set_compile_dynamic_hints(hints)
        self._config = replace(
            self._config,
            compile_dynamic_hints=self._executor.compile_dynamic_hints,
        )

    def finalize(self, network=None):
        """Build the permanent train/sample programs, then activate TRAIN.

        Must run AFTER the training network is applied: the programs capture the
        installed adapter leaves. `network` is accepted for the eventual generic
        adapter protocol (see GENERIC_ADAPTER_IMMUTABLE_RUNTIME_PLAN.md); today
        the model collects its own adapters.
        """
        self._require_open()
        self._bind_training_cap()
        finalize_fn = getattr(self._model, "finalize_immutable_runtime", None)
        if finalize_fn is None:
            raise RuntimeError(
                "the model prepared an arena runtime but does not expose "
                "finalize_immutable_runtime."
            )
        finalize_fn()
        if self._config.fp8_forward:
            from ..manager import MemoryManager

            (
                self._training_fp8_restores,
                self._training_fp8_singletons,
            ) = MemoryManager._enable_fp8_training_compile(
                self._model,
                include_ids=self._singleton_runtime_ids(),
            )
        self._executor.activate(self._executor.TRAIN, self._training_plan)
        return self

    def close(self) -> None:
        if self._closed:
            return
        if getattr(self, "_training_fp8_restores", None):
            from ..manager import MemoryManager

            MemoryManager._disable_fp8_training_compile(
                self._model, self._training_fp8_restores
            )
            self._training_fp8_restores = []
        disable = getattr(self._model, "disable_immutable_runtime", None)
        if disable is not None:
            disable()
        else:
            self._executor.close()
        if getattr(self._model, RUNTIME_ATTR, None) is self:
            delattr(self._model, RUNTIME_ATTR)
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("arena_offload_runtime_closed")

    # ------------------------------------------------------------------
    # execution contexts
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def training_step(self, *, shape_key: tuple | None = None, step_num: int | None = None):
        """The training phase boundary. Spans forward AND backward.

        Backward must be inside: checkpoint recomputation re-enters the block
        runtime, so the source snapshot has to stay pinned for the whole step.

        This is the two-timescale residency controller's phase-boundary hook:
        enter = plan/act, exit = observe.
        """
        self._require_open()
        self._last_shape_key = shape_key
        self._last_step_num = step_num
        try:
            self._apply_training_policy()
        except BaseException as error:
            try:
                self._handle_training_failure(
                    error, shape_key=shape_key, step_num=step_num
                )
            except Exception as cleanup_error:
                self._last_policy_error = (
                    "training_policy_cleanup: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self._device)
        except Exception:
            pass
        started_at = time.perf_counter()
        succeeded = False
        try:
            with self._executor.execution(self._executor.TRAIN):
                yield self
            succeeded = True
        except BaseException as error:
            try:
                self._handle_training_failure(
                    error, shape_key=shape_key, step_num=step_num
                )
            except Exception as cleanup_error:
                self._last_policy_error = (
                    "training_failure_cleanup: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        finally:
            if succeeded:
                try:
                    self._observe_training_step(
                        shape_key=shape_key,
                        step_num=step_num,
                        step_wall_ms=(time.perf_counter() - started_at) * 1000.0,
                    )
                    self._last_policy_error = None
                except Exception as error:
                    # Diagnostics must never mask a successful training step.
                    self._last_policy_error = f"{type(error).__name__}: {error}"

    def _handle_training_failure(self, error, *, shape_key, step_num):
        """Clean arena-owned state after the executor has unwound."""
        import torch

        from .. import ingraph_stream
        from ..vram_budget import device_free_bytes

        abandoned = ingraph_stream.drain_fetch_runtime()
        text = str(error)
        allocation_failure = (
            isinstance(error, torch.cuda.OutOfMemoryError)
            or "out of memory" in text.lower()
        )
        rollback = None
        if allocation_failure:
            decision = self._policy.allocation_failure()
            if decision.action == "rollback" and decision.block_key is not None:
                rollback_keys = tuple(decision.block_keys or ())
                if rollback_keys:
                    self.transition_training_blocks(
                        rollback_keys, resident=False
                    )
                else:
                    self.transition_training_block(
                        decision.block_key, resident=False
                    )
                if decision.target_cap_bytes is not None:
                    allocator_cap.apply_wddm_hard_allocator_cap(
                        self._device,
                        self._config.legacy.wddm_hard_gib,
                        target_cap_bytes=decision.target_cap_bytes,
                        log_prefix="[ArenaOffload]",
                    )
                    self._last_training_cap_target_bytes = (
                        decision.target_cap_bytes
                    )
                rollback = (
                    list(decision.block_keys)
                    if decision.block_keys
                    else decision.block_key
                )
        try:
            stats = torch.cuda.memory_stats(self._device)
            peak_allocated = int(torch.cuda.max_memory_allocated(self._device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self._device))
            physical_free = int(device_free_bytes(self._device))
        except Exception:
            stats = {}
            peak_allocated = 0
            peak_reserved = 0
            physical_free = 0
        active_cap = allocator_cap.applied_cap_bytes(self._device)
        legacy = getattr(getattr(self, "_config", None), "legacy", None)
        hard_gib = getattr(legacy, "wddm_hard_gib", None)
        hard_bytes = int(max(1.0, float(hard_gib or 1.0)) * GIB)
        classification = "non_allocation_failure"
        if allocation_failure:
            classification = (
                "capped_allocator_rejection"
                if active_cap is not None and physical_free > hard_bytes / 2
                else "cuda_allocation_failure_unknown"
            )
        self._last_failure_event = {
            "event": "training_allocation_failure",
            "classification": classification,
            "exception_type": type(error).__name__,
            "exception": text,
            "shape_key": shape_key,
            "step_num": step_num,
            "active_cap_bytes": active_cap,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "physical_free_bytes": physical_free,
            "allocator_retry_delta": max(
                0,
                int(stats.get("num_alloc_retries", 0) or 0)
                - int(
                    (
                        getattr(self._signals, "_allocator_previous", None)
                        or {}
                    ).get("num_alloc_retries", 0)
                    or 0
                ),
            ),
            "rollback_block": rollback,
            "rejected_residency_bytes": (
                self._policy.last_rejected_residency_bytes
            ),
            "abandoned_fetch_tickets": int(abandoned),
        }

    def _singleton_runtime_ids(self):
        manager = getattr(self._model, "_memory_manager", None)
        return set(
            getattr(manager, "_training_runtime_candidate_ids", ())
        )

    @contextlib.contextmanager
    def sampling_session(self):
        """Wraps a whole sampling run (all images), restoring TRAIN at the end.

        The TRAIN program is restored once per session, not once per image:
        re-activating it between images would reconcile residency back to the
        training plan and churn the sidecars for nothing.
        """
        self._require_open()
        sampling_restores = []
        if self._config.fp8_sampling:
            from ..manager import MemoryManager

            (
                sampling_restores,
                self._sampling_fp8_singletons,
                _streamed,
            ) = MemoryManager._enable_fp8_sampling(
                self._model,
                include_ids=self._singleton_runtime_ids(),
            )
        try:
            yield self
        finally:
            if sampling_restores:
                from ..manager import MemoryManager

                MemoryManager._disable_fp8_sampling(
                    self._model, sampling_restores
                )
            self._bind_training_cap()
            self._executor.activate(self._executor.TRAIN, self._training_plan)

    @contextlib.contextmanager
    def sampling_image(self, *, shape_key: tuple, cold_working_bytes: int):
        """The sampling phase boundary for ONE image.

        Switches to the permanent SAMPLE program (forward-only, no
        checkpointing) over the same arena. TRAIN is restored by the enclosing
        `sampling_session()`.
        """
        self._require_open()
        legacy = self._config.legacy

        fixed_working_bytes = _fixed_working_bytes(legacy.sampling_working_reserve_gib)
        hard_gib = (
            1.0
            if legacy.sampling_wddm_hard_gib is None
            else float(legacy.sampling_wddm_hard_gib)
        )
        allocator_cap.apply_wddm_hard_allocator_cap(
            self._device, hard_gib, log_prefix="[ArenaOffload]"
        )
        # PHASE-2: margin resolution moves into policy.py.
        from ..manager import MemoryManager

        margin_gib = MemoryManager._resolve_wddm_margin_gib(
            self._device,
            legacy.sampling_wddm_margin_gib,
            hard_gib=hard_gib,
            env_name="AI_TOOLKIT_SAMPLING_WDDM_MARGIN_GIB",
        )
        dequant_reserve = (
            0
            if self._config.fp8_sampling
            else int(
                (self._smart_plan or {}).get(
                    "largest_singleton_bf16_dequant_bytes", 0
                )
            )
        )
        with self._executor.sampling(
            shape_key=shape_key,
            cold_working_bytes=int(cold_working_bytes),
            fixed_working_bytes=fixed_working_bytes,
            cold_floor_bytes=int(margin_gib * GIB) + dequant_reserve,
            hot_floor_bytes=int((hard_gib + 0.25) * GIB) + dequant_reserve,
        ):
            yield self

    # ------------------------------------------------------------------
    def record_training_physical_free_min(self, free_bytes) -> None:
        """Publish one successful step's physical high-water for bootstrap."""
        if self._bootstrap_complete or free_bytes is None:
            return
        value = max(0, int(free_bytes))
        self._bootstrap_min_free_bytes = (
            value
            if self._bootstrap_min_free_bytes is None
            else min(self._bootstrap_min_free_bytes, value)
        )

    def _bootstrap_training_residency(self, active_cap_bytes) -> bool:
        if (
            self._bootstrap_complete
            or self._bootstrap_min_free_bytes is None
            or int(self._last_step_num or 0) < BOOTSTRAP_MIN_STEP
        ):
            return False
        hard_gib = self._config.legacy.wddm_hard_gib
        hard_bytes = int(
            (1.0 if hard_gib is None else max(1.0, float(hard_gib))) * GIB
        )
        budget = max(
            0,
            self._bootstrap_min_free_bytes
            - hard_bytes
            - BOOTSTRAP_MARGIN_BYTES,
        )
        self._bootstrap_budget_bytes = budget
        candidates = []
        protected = self._protected_training_blocks()
        plan = getattr(self._residency, "plan", None) or self._training_plan
        for order, block_key in enumerate(self._arena.block_keys()):
            record = self._arena.block_record(block_key)
            keys = tuple((block_key, name) for name in record.leaf_names)
            if block_key in protected or any(
                key in plan.resident_leaf_keys for key in keys
            ):
                continue
            candidates.append(
                (int(record.committed_bytes), order, str(block_key))
            )
        selected = []
        used = 0
        for block_bytes, _order, block_key in sorted(candidates):
            if used + block_bytes > budget:
                continue
            selected.append(block_key)
            used += block_bytes
        if not selected:
            self._bootstrap_complete = True
            return False
        resident_before = (
            int(self._residency.resident_bytes())
            + int((self._smart_plan or {}).get("singleton_resident_bytes", 0))
        )
        result = self.transition_training_blocks(selected, resident=True)
        if not result.get("changed"):
            return False
        self._bootstrap_complete = True
        self._bootstrap_block_keys = tuple(result["block_keys"])
        self._policy.begin_bootstrap_promotion(
            self._bootstrap_block_keys,
            used,
            resident_before,
            active_cap_bytes,
        )
        return True

    def _protected_training_blocks(self):
        return frozenset(
            str(block)
            for block, _leaf in getattr(
                self._model,
                "_mm_immutable_protected_training_leaf_keys",
                (),
            )
        )

    def _promotion_candidate(self):
        plan = getattr(self._residency, "plan", None) or self._training_plan
        protected = self._protected_training_blocks()
        candidates = []
        for order, block_key in enumerate(self._arena.block_keys()):
            record = self._arena.block_record(block_key)
            keys = tuple((block_key, name) for name in record.leaf_names)
            if block_key in protected or any(
                key in plan.resident_leaf_keys for key in keys
            ):
                continue
            candidates.append(
                (int(record.committed_bytes), order, str(block_key))
            )
        if not candidates:
            return None
        block_bytes, _order, block_key = min(candidates)
        return {"block_key": block_key, "block_bytes": block_bytes}

    def _demotion_candidate(self):
        plan = getattr(self._residency, "plan", None) or self._training_plan
        protected = self._protected_training_blocks()
        candidates = []
        for order, block_key in enumerate(self._arena.block_keys()):
            record = self._arena.block_record(block_key)
            keys = tuple((block_key, name) for name in record.leaf_names)
            if block_key in protected or not all(
                key in plan.resident_leaf_keys for key in keys
            ):
                continue
            actual = sum(
                self._residency.resident_leaf_bytes(key) for key in keys
            )
            candidates.append(
                (actual or int(record.committed_bytes), -order, str(block_key))
            )
        if not candidates:
            return None
        block_bytes, _order, block_key = max(candidates)
        return {"block_key": block_key, "block_bytes": block_bytes}

    def _worst_shape_candidate_margin_bytes(self, candidate):
        if candidate is None:
            return 0
        signal = self._signals.last_signal
        peaks = self._signals.shape_peaks
        if signal is None or not peaks:
            return 0
        from .. import vram_budget

        total = int(vram_budget.device_total_bytes(self._device))
        worst_working = max(
            int(peak.working_peak_bytes)
            for peak in peaks.values()
            if peak.steps > 0
        ) if any(peak.steps > 0 for peak in peaks.values()) else 0
        current_resident = (
            int(self._residency.resident_bytes())
            + int((self._smart_plan or {}).get("singleton_resident_bytes", 0))
        )
        current_ring = self._training_ring_bytes()
        non_torch = max(
            0,
            total
            - int(signal.get("device_free_bytes", 0) or 0)
            - int(signal.get("peak_reserved_bytes", 0) or 0),
        )
        hard_gib = self._config.legacy.wddm_hard_gib
        hard_bytes = int(
            (1.0 if hard_gib is None else max(1.0, float(hard_gib))) * GIB
        )
        predicted_free = total - (
            worst_working
            + current_resident
            + current_ring
            + non_torch
            + int(candidate["block_bytes"])
        )
        return int(predicted_free - hard_bytes)

    def _worst_shape_allocator_slack_bytes(self, current_cap_bytes):
        peaks = self._signals.shape_peaks
        if not any(peak.steps > 0 for peak in peaks.values()):
            return 0
        from .. import vram_budget

        worst_working = max(
            int(peak.working_peak_bytes)
            for peak in peaks.values()
            if peak.steps > 0
        )
        current_resident = (
            int(self._residency.resident_bytes())
            + int((self._smart_plan or {}).get("singleton_resident_bytes", 0))
        )
        predicted_live = (
            worst_working
            + current_resident
            + self._training_ring_bytes()
        )
        return vram_budget.allocator_allowance_bytes(
            current_cap_bytes, predicted_live
        )

    def _apply_training_policy(self):
        import torch

        if torch.device(self._device).type != "cuda" or not torch.cuda.is_available():
            return
        candidate = self._promotion_candidate()
        demote_candidate = self._demotion_candidate()
        cliff_cap = allocator_cap.wddm_cliff_cap_bytes(
            self._device, self._config.legacy.wddm_hard_gib
        )
        signal = self._signals.last_signal
        current_cap = min(
            cliff_cap,
            int(self._last_training_cap_target_bytes or cliff_cap),
        )
        if self._bootstrap_training_residency(current_cap):
            return
        decision = self._policy.step(
            self._signals.last_signal,
            candidate=candidate,
            demote_candidate=demote_candidate,
            cliff_cap_bytes=cliff_cap,
            current_cap_bytes=current_cap,
            worst_shape_free_bytes=self._worst_shape_candidate_margin_bytes(
                candidate
            ),
            worst_shape_allocator_slack_bytes=(
                self._worst_shape_allocator_slack_bytes(current_cap)
            ),
        )
        if decision.action == "promote":
            self.transition_training_block(decision.block_key, resident=True)
        elif decision.action in ("demote", "rollback"):
            rollback_keys = tuple(decision.block_keys or ())
            if rollback_keys:
                self.transition_training_blocks(
                    rollback_keys, resident=False
                )
            else:
                self.transition_training_block(
                    decision.block_key, resident=False
                )
            if (
                decision.action == "rollback"
                and decision.target_cap_bytes is not None
            ):
                allocator_cap.apply_wddm_hard_allocator_cap(
                    self._device,
                    self._config.legacy.wddm_hard_gib,
                    target_cap_bytes=decision.target_cap_bytes,
                    log_prefix="[ArenaOffload]",
                )
                self._last_training_cap_target_bytes = (
                    decision.target_cap_bytes
                )
        elif decision.action == "raise_cap":
            allocator_cap.apply_wddm_hard_allocator_cap(
                self._device,
                self._config.legacy.wddm_hard_gib,
                target_cap_bytes=decision.target_cap_bytes,
                log_prefix="[ArenaOffload]",
            )
            self._last_training_cap_target_bytes = decision.target_cap_bytes

    def transition_training_blocks(self, block_keys, *, resident: bool) -> dict:
        """Apply one executor-owned multi-block transaction at a boundary."""
        self._require_open()
        result = self._executor.transition_training_blocks(
            tuple(block_keys), resident=bool(resident)
        )
        if result.get("changed"):
            self._training_plan = result["plan"]
        return result

    def transition_training_block(self, block_key: str, *, resident: bool) -> dict:
        """Apply one executor-owned whole-block transaction at a boundary."""
        self._require_open()
        result = self._executor.transition_training_block(
            str(block_key), resident=bool(resident)
        )
        if result.get("changed"):
            self._training_plan = result["plan"]
        return result

    def _bind_training_cap(self) -> None:
        allocator_cap.apply_wddm_hard_allocator_cap(
            self._device,
            self._config.legacy.wddm_hard_gib,
            log_prefix="[ArenaOffload]",
        )

    # diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict:
        """One stable dict. Shared logging prints it; nobody reconstructs it."""
        active_plan = getattr(self._residency, "plan", None) or self._training_plan
        canonical_resident = int(self._residency.resident_bytes())
        singleton_resident = int(
            (self._smart_plan or {}).get("singleton_resident_bytes", 0)
        )
        return {
            "backend": "arena",
            "blocks": self.block_count,
            "finalized": self.finalized,
            "resident_bytes": singleton_resident + canonical_resident,
            "singleton_resident_bytes": singleton_resident,
            "canonical_resident_bytes": canonical_resident,
            "total_weight_resident_bytes": singleton_resident + canonical_resident,
            "plan_fingerprint": getattr(active_plan, "fingerprint", None),
            "prefetch_depth": int(getattr(self._executor, "depth", 0)),
            "compile_blocks": bool(self._config.compile_blocks),
            "compile_dynamic": bool(self._config.compile_dynamic),
            "fp8_forward": bool(self._config.fp8_forward),
            "fp8_backward": bool(self._config.fp8_backward),
            "fp8_sampling": bool(self._config.fp8_sampling),
            "training_fp8_singletons": getattr(
                self, "_training_fp8_singletons", 0
            ),
            "sampling_fp8_singletons": getattr(
                self, "_sampling_fp8_singletons", 0
            ),
            "largest_singleton_bf16_dequant_bytes": int(
                (self._smart_plan or {}).get(
                    "largest_singleton_bf16_dequant_bytes", 0
                )
            ),
            "training_cap_target_bytes": getattr(
                self, "_last_training_cap_target_bytes", None
            ),
            "bootstrap_complete": self._bootstrap_complete,
            "bootstrap_min_free_bytes": self._bootstrap_min_free_bytes,
            "bootstrap_margin_bytes": BOOTSTRAP_MARGIN_BYTES,
            "bootstrap_min_step": BOOTSTRAP_MIN_STEP,
            "bootstrap_budget_bytes": self._bootstrap_budget_bytes,
            "bootstrap_block_keys": self._bootstrap_block_keys,
            "working_reserve_bytes": int(
                (self._smart_plan or {}).get("working_reserve_bytes", 0)
            ),
            "last_shape_key": self._last_shape_key,
            "last_step_num": self._last_step_num,
            "policy": {
                **self._signals.diagnostics(),
                "controller": self._policy.diagnostics(),
            },
            "policy_error": self._last_policy_error,
            "last_failure_event": self._last_failure_event,
        }

    def _observe_training_step(self, *, shape_key, step_num, step_wall_ms) -> None:
        """Collect the completed step's policy signals."""
        import torch

        from .. import ingraph_stream
        from ..vram_budget import device_free_bytes

        if torch.device(self._device).type != "cuda" or not torch.cuda.is_available():
            return
        try:
            stats = torch.cuda.memory_stats(self._device)
        except Exception:
            stats = {}
        allocator = {
            key: int(stats.get(key, 0) or 0)
            for key in ("num_alloc_retries", "num_device_alloc", "num_device_free")
        }
        transfer = (
            ingraph_stream.lifetime_fetch_stats()
            if self._signals.transfer_snapshot_due
            else None
        )
        self._signals.observe(
            shape_key=shape_key,
            step_num=step_num,
            allocator_counters=allocator,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self._device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(self._device),
            device_free_bytes=device_free_bytes(self._device),
            resident_bytes=(
                self._residency.resident_bytes()
                + int((self._smart_plan or {}).get("singleton_resident_bytes", 0))
            ),
            ring_bytes=self._training_ring_bytes(),
            compile_counters=_compile_counter_snapshot(torch),
            transfer_counters=transfer,
            step_wall_ms=step_wall_ms,
        )

    def _training_ring_bytes(self) -> int:
        from ..transfer_plan import build_transfer_plan

        largest = 0
        plan = getattr(self._residency, "plan", None) or self._training_plan
        for block_key in self._arena.block_keys():
            record = self._arena.block_record(block_key)
            streamed = tuple(
                name
                for name in record.leaf_names
                if (block_key, name) not in plan.resident_leaf_keys
            )
            if streamed:
                transfer = build_transfer_plan(record, streamed)
                largest = max(largest, transfer.compact_nbytes)
        depth = max(1, int(getattr(self._executor, "depth", 1)))
        return int(largest * depth)

    def report_foreign_vram_once(self, *, phase: str) -> None:
        """Say so, once, if another tenant on the GPU is why we are streaming."""
        self._executor.report_foreign_vram_once(
            self._residency.device,
            phase=phase,
            working_reserve_bytes=int(
                (self._smart_plan or {}).get("working_reserve_bytes", 0)
            ),
        )

    # ------------------------------------------------------------------
    # escape hatches (Phase 1 only -- each has a phase that removes it)
    # ------------------------------------------------------------------

    @property
    def _legacy_executor(self):
        """PHASE-2/5: direct executor access, for call sites not yet migrated."""
        return self._executor

    @property
    def _legacy_training_plan(self):
        """PHASE-2: the TRAIN ResidencyPlan, for call sites not yet migrated."""
        return self._training_plan


def _compile_counter_snapshot(torch_module):
    try:
        counters = torch_module._dynamo.utils.counters
    except AttributeError:
        return None
    frames = int(counters["frames"].get("total", 0) or 0)
    graphs = int(counters["stats"].get("unique_graphs", 0) or 0)
    if frames == 0 and graphs == 0:
        return None
    return {
        "frames": frames,
        "graphs": graphs,
        "graph_breaks": int(sum(counters["graph_break"].values())),
    }


def _fixed_working_bytes(value) -> int | None:
    """None when the sampling working reserve is auto (unset, negative, 'auto')."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        # Non-numeric (e.g. "auto") means auto-size, same as unset.
        return None
    if numeric < 0:
        return None
    return int(numeric * GIB)
