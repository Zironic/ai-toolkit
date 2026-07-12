import contextlib
import json
import sys
import pathlib
import re
import os
import time
import torch
from .manager_modules import (
    LinearLayerMemoryManager,
    ConvLayerMemoryManager,
    _DEVICE_STATE,
    _is_quantized_tensor,
    _profile_is_pinned,
    _unpin_inner_tensors,
    unpin_layer,
    fp8_linear_inference,
    fp8_sampling_qualifies,
    _fp8_linear_compiled,
    _fp8_linear_training,
    _FP8_STATS,
    PIPELINE_DEPTH,
    summarize_offload_profile,
    set_offload_profile_enabled,
    offload_step_begin,
    offload_step_end,
    offload_step_abort,
    offload_trace_report,
    offload_trace_schedule,
    offload_trace_schedule_confidence,
    offload_trace_version,
    mark_transfer_plan_dirty,
    invalidate_offload_trace_for_shape,
    invalidate_execution_trace,
    set_offload_trace_enabled,
    set_fp8_grad_input_enabled,
    record_weight_access,
    set_block_stream_enabled,
    stage_block_forward,
    block_forward_done,
    reset_block_stream,
)
from .ingraph_stream import fetch_report as ingraph_fetch_report
from .ingraph_stream import drain_fetch_runtime as ingraph_drain_fetch_runtime
from .ingraph_stream import _flatten_leaves, _rebuild_from_leaves
from . import bounce_pool
from . import pin_manager
from . import vram_budget



# The reserve vocabulary was renamed (headroom -> working_reserve; buffer_hard/
# buffer_stop/hold_high/target_free/vram_safety -> wddm_* margins). These env
# overrides are read through _env so the previous names keep working.
_ENV_ALIASES = {
    "AI_TOOLKIT_SAMPLING_WORKING_RESERVE_GIB": "AI_TOOLKIT_SAMPLING_HEADROOM_GIB",
    "AI_TOOLKIT_SAMPLING_WORKING_RESERVE_FLOOR_GIB": "AI_TOOLKIT_SAMPLING_HEADROOM_FLOOR_GIB",
    "AI_TOOLKIT_SAMPLING_WORKING_RESERVE_PAD_GIB": "AI_TOOLKIT_SAMPLING_HEADROOM_PAD_GIB",
    "AI_TOOLKIT_SAMPLING_WDDM_HARD_GIB": "AI_TOOLKIT_SAMPLING_BUFFER_HARD_GIB",
    "AI_TOOLKIT_TRAINING_AUTO_SEED_WORKING_RESERVE_GIB": "AI_TOOLKIT_TRAINING_AUTO_SEED_HEADROOM_GIB",
    "AI_TOOLKIT_TRAINING_WORKING_RESERVE_PAD_GIB": "AI_TOOLKIT_TRAINING_HEADROOM_PAD_GIB",
    "AI_TOOLKIT_TRAINING_WORKING_RESERVE_STEP_GIB": "AI_TOOLKIT_TRAINING_HEADROOM_STEP_GIB",
    "AI_TOOLKIT_TRAINING_MIN_WORKING_RESERVE_GIB": "AI_TOOLKIT_TRAINING_MIN_HEADROOM_GIB",
    "AI_TOOLKIT_TRAINING_STABLE_WORKING_RESERVE_STEPS": "AI_TOOLKIT_TRAINING_STABLE_HEADROOM_STEPS",
    "AI_TOOLKIT_TRAINING_WDDM_HARD_GIB": "AI_TOOLKIT_TRAINING_BUFFER_HARD_GIB",
    "AI_TOOLKIT_TRAINING_WDDM_STOP_GIB": "AI_TOOLKIT_TRAINING_BUFFER_STOP_GIB",
    "AI_TOOLKIT_TRAINING_WDDM_HOLD_HIGH_GIB": "AI_TOOLKIT_TRAINING_HOLD_HIGH_GIB",
    "AI_TOOLKIT_TRAINING_WDDM_MARGIN_GIB": "AI_TOOLKIT_TRAINING_TARGET_FREE_GIB",
    "AI_TOOLKIT_TRAINING_WDDM_SAFETY_GIB": "AI_TOOLKIT_TRAINING_VRAM_SAFETY_GIB",
}


def _env(name, default=None):
    """os.environ.get with backward-compatible fallback to the pre-rename name."""
    if name in os.environ:
        return os.environ[name]
    old = _ENV_ALIASES.get(name)
    if old is not None and old in os.environ:
        return os.environ[old]
    return default


LINEAR_MODULES = [
    "Linear",
    "LoRACompatibleLinear",
    "QLinear",
]
CONV_MODULES = [
    "Conv2d",
    "LoRACompatibleConv",
    "QConv2d",
]

UNMANAGED_MODULES = [
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "Embedding",
    "EmbeddingBag",
    "RNNBase",
    "LSTM",
    "GRU",
    "RNN",
    "Conv3d"
]

UNMANAGED_MODULES_INCLUDES = ["RotaryEmbedding", "Norm", "RotaryPosEmbed"]


# Slice 2B: when the bounce pool is active it does all the pinning from a
# bounded reusable pool, so the canonical weights stay pageable and we skip the
# old permanent per-weight pinning (which only ever covered ~1 GiB anyway).
_OFFLOAD_PREFETCH_ENABLED = _env(
    "AI_TOOLKIT_OFFLOAD_PREFETCH", "0"
).lower() not in ("0", "false", "no", "off", "")


def _dxgi_telemetry(cuda_device_index: int = 0, min_interval_s: float = 0.5) -> dict:
    dxgi = bounce_pool.get_dxgi_meminfo()
    if dxgi is None:
        return {}
    non_local = dxgi.query_non_local_video_memory_info(
        cuda_device_index=cuda_device_index,
        min_interval_s=min_interval_s,
    )
    local = dxgi.query_local_video_memory_info(
        cuda_device_index=cuda_device_index,
        min_interval_s=min_interval_s,
    )
    adapter = dxgi.selected_adapter_info()
    fields = {}
    if non_local is not None:
        spill_reserve = bounce_pool.dxgi_spill_reserve_bytes(non_local.budget_bytes)
        headroom = dxgi.compute_non_local_headroom_bytes(
            non_local.budget_bytes,
            non_local.current_usage_bytes,
            spill_reserve,
        )
        fields.update({
            "dxgi_non_local_budget_gb": non_local.budget_bytes / 1024 ** 3,
            "dxgi_non_local_usage_gb": non_local.current_usage_bytes / 1024 ** 3,
            "dxgi_non_local_headroom_gb": headroom / 1024 ** 3,
            "dxgi_spill_reserve_gb": spill_reserve / 1024 ** 3,
        })
    if local is not None:
        fields.update({
            "dxgi_local_budget_gb": local.budget_bytes / 1024 ** 3,
            "dxgi_local_usage_gb": local.current_usage_bytes / 1024 ** 3,
            "dxgi_local_headroom_gb": max(
                0, local.budget_bytes - local.current_usage_bytes
            ) / 1024 ** 3,
            "dxgi_local_available_for_reservation_gb": (
                local.available_for_reservation_bytes / 1024 ** 3
            ),
            "dxgi_local_current_reservation_gb": (
                local.current_reservation_bytes / 1024 ** 3
            ),
        })
    if adapter is not None:
        fields.update({
            "dxgi_adapter_index": adapter.index,
            "dxgi_adapter_description": adapter.description,
            "dxgi_adapter_vendor_id": adapter.vendor_id,
            "dxgi_adapter_device_id": adapter.device_id,
            "dxgi_adapter_luid": adapter.luid,
            "dxgi_match_method": adapter.match_method,
            "dxgi_safe_for_control": adapter.safe_for_control,
            "dxgi_manual_control": adapter.manual_control,
        })
    return fields

def _process_memory_telemetry() -> dict:
    psutil = bounce_pool._psutil
    if psutil is None:
        return {}
    try:
        proc = psutil.Process()
        mi = proc.memory_info()
    except Exception:
        return {}
    fields = {}
    rss = getattr(mi, "rss", None)
    private = getattr(mi, "private", None)
    if rss is not None:
        fields["proc_working_set_gb"] = int(rss) / 1024 ** 3
    if private is not None:
        fields["proc_private_commit_gb"] = int(private) / 1024 ** 3
    try:
        uss = getattr(proc.memory_full_info(), "uss", None)
    except Exception:
        uss = None
    if uss is not None:
        fields["proc_uss_gb"] = int(uss) / 1024 ** 3
    return fields


def _dxgi_attach_log_text(cuda_device_index: int = 0) -> str:
    fields = _dxgi_telemetry(cuda_device_index, min_interval_s=0.0)
    if "dxgi_non_local_budget_gb" in fields:
        dxgi_text = (
            f"dxgi_non_local_budget={fields['dxgi_non_local_budget_gb']:.2f} GiB "
            f"dxgi_non_local_usage={fields['dxgi_non_local_usage_gb']:.2f} GiB "
        )
    else:
        dxgi_text = "dxgi_non_local=unavailable "
    adapter_desc = fields.get("dxgi_adapter_description", "unavailable")
    return (
        dxgi_text
        + f"pinned_ledger_total={bounce_pool._pinned_bytes_total / 1024 ** 3:.2f} GiB "
        + pin_manager.format_snapshot(cuda_device_index) + " "
        + f"dxgi_adapter_index={fields.get('dxgi_adapter_index')} "
        + f"dxgi_adapter_description={adapter_desc!r} "
        + f"dxgi_adapter_vendor_id={fields.get('dxgi_adapter_vendor_id')} "
        + f"dxgi_adapter_device_id={fields.get('dxgi_adapter_device_id')} "
        + f"dxgi_adapter_luid={fields.get('dxgi_adapter_luid')} "
        + f"dxgi_match_method={fields.get('dxgi_match_method')}"
    )


# Per-device bytes the allocator cap has been permanently widened by, after the
# cap was violated in non-strict mode. Keyed by CUDA device index; see
# MemoryManager.relieve_wddm_cap_after_oom.
_WDDM_CAP_RELIEF_BYTES: dict = {}


class MemoryManager:
    def __init__(
        self,
        module: torch.nn.Module,
        process_device: torch.device = torch.device("cpu"),
        pinned_weight_gib: float | None = None,
    ):
        self.module: torch.nn.Module = module
        self.process_device: torch.device = process_device
        self.unmanaged_modules: list[torch.nn.Module] = []
        self.pinned_weight_bytes = 0
        # How much offloaded weight to keep page-locked (pinned) in CPU RAM.
        # Pinning stops weights being paged out and re-faulted on every fetch.
        # It does not create a second steady-state copy of the already-loaded
        # CPU weights, but on Windows/WDDM it does count against the GPU's
        # shared-memory budget --
        # exhausting either makes the next CUDA call fail with a raw
        # cudaErrorMemoryAllocation even with VRAM free. Auto budgets are
        # capped by _cap_auto_pin_budget; a job config value (>= 0) wins;
        # otherwise fall back to the env/default (0 with prefetch on, since
        # the pool pins on demand; 1 otherwise).
        if pinned_weight_gib is not None and float(pinned_weight_gib) >= 0:
            self.pinned_weight_budget_bytes = int(float(pinned_weight_gib) * 1024 ** 3)
        else:
            default_pin_gib = "0.0" if _OFFLOAD_PREFETCH_ENABLED else "1.0"
            self.pinned_weight_budget_bytes = int(
                float(_env("AI_TOOLKIT_PINNED_WEIGHT_GIB", default_pin_gib))
                * 1024 ** 3
            )
        self._prefetch_pool = None
        self._resident_trace_hooks = {}

    @staticmethod
    def _cap_auto_pin_budget(
        budget: int, reserve_bytes: int = 0, device=None
    ) -> int:
        """Cap an auto-sized pinned-weight budget by what the host can give.

        Pinned (cudaHostAlloc) memory is page-locked AND, under WDDM, counts
        against the GPU's shared-memory budget (~RAM/2 shared by every GPU
        process). Blowing that budget makes the next CUDA call of any size
        fail with a raw cudaErrorMemoryAllocation while VRAM sits mostly free.
        The useful cap is the process-wide pinned-bytes ledger's headroom (a fraction of
        total RAM as a WDDM shared-budget proxy, minus whatever this process
        has already pinned elsewhere -- e.g. the bounce pool's own buffers, or
        a prior manager's weight pins; the real DXGI budget is not visible
        through CUDA). Shared with bounce_pool.py's own allocation checks so
        neither subsystem can push the combined total past the ceiling blind
        to the other's consumption.
        """
        gib = 1024 ** 3
        reserve_bytes = max(0, int(reserve_bytes or 0))
        reserve_text = (
            f" reserve_for_bounce={reserve_bytes / gib:.2f} GiB;"
            if reserve_bytes
            else ""
        )
        device_index = bounce_pool._cuda_device_index(device)
        dxgi_headroom = bounce_pool.dxgi_pinned_headroom(device_index)
        if dxgi_headroom is not None:
            # The real shared-budget probe is authoritative: system-RAM numbers
            # ("available" especially) do not measure the resource pinning
            # spends and must not shrink the budget when DXGI is visible.
            capped = min(budget, max(0, dxgi_headroom - reserve_bytes))
            if capped < budget:
                print(
                    "[MemoryManager] pinned-weight auto-budget capped: "
                    f"want={budget / gib:.2f} GiB -> {capped / gib:.2f} GiB "
                    f"(dxgi_pinned_headroom={dxgi_headroom / gib:.2f} GiB;"
                    f"{reserve_text} "
                    "pinned memory commits against the WDDM shared GPU budget -- "
                    "set layer_offloading_pinned_weight_gb to override)"
                )
            return capped
        # Fallback (no DXGI probe): conservative system-RAM proxies.
        vm = None
        try:
            if bounce_pool._psutil is not None:
                vm = bounce_pool._psutil.virtual_memory()
        except Exception:
            vm = None
        if vm is None:
            return budget
        total_floor = int(
            float(_env("AI_TOOLKIT_PINNED_WEIGHT_RAM_FLOOR_GIB", "8.0")) * gib
        )
        ledger_headroom = bounce_pool.pinned_bytes_headroom(device_index)
        caps = [
            budget,
            max(0, int(vm.total) - total_floor - reserve_bytes),
        ]
        if ledger_headroom is not None:
            caps.append(max(0, ledger_headroom - reserve_bytes))
        capped = min(caps)
        if capped < budget:
            print(
                "[MemoryManager] pinned-weight auto-budget capped (RAM proxy, no DXGI probe): "
                f"want={budget / gib:.2f} GiB -> {capped / gib:.2f} GiB "
                f"(reported_available={vm.available / gib:.2f} GiB total={vm.total / gib:.2f} GiB;"
                f"{reserve_text} "
                "pinned memory is page-locked and counts against the WDDM shared "
                "GPU budget -- set layer_offloading_pinned_weight_gb to override)"
            )
        return capped

    @classmethod
    def _training_bounce_pool_budget_defaults(
        cls,
        module,
        offload_ids=None,
        *,
        block_stream_only=False,
        sources=None,
        history=None,
    ):
        """Return default ``(budget_gib, target_ready_gib, mode)`` for training prefetch.

        Bounce memory and permanent pinned weights draw from the same WDDM shared
        pinned-memory ceiling. The pool is only a temporary staging window, so
        default it from the current streaming unit instead of replaying old large
        history budgets. Explicit AI_TOOLKIT_BOUNCE_* env values still win.
        """
        gib = 1024 ** 3
        history = history or {}
        offload_ids = set(offload_ids or ())
        rows = []
        if sources is not None:
            for key, child in sources:
                rows.append((str(key), cls._training_stream_bytes(child)))
        elif module is not None:
            for name, child in module.named_modules():
                if offload_ids and id(child) not in offload_ids:
                    continue
                if (
                    child.__class__.__name__ not in LINEAR_MODULES
                    and child.__class__.__name__ not in CONV_MODULES
                ):
                    continue
                rows.append((name or child.__class__.__name__, cls._training_stream_bytes(child)))

        sizes = [max(0, int(n)) for _key, n in rows if int(n or 0) > 0]
        if not sizes:
            budget_gib = float(_env("AI_TOOLKIT_BOUNCE_LAYER_DEFAULT_GIB", "1.0"))
            target_gib = min(
                budget_gib,
                float(_env("AI_TOOLKIT_BOUNCE_LAYER_TARGET_GIB", "0.75")),
            )
            return budget_gib, target_gib, "empty"

        slack = max(1.0, float(_env("AI_TOOLKIT_BOUNCE_AUTO_SLACK", "1.20")))
        if block_stream_only:
            group_bytes = {}
            for key, n in rows:
                gk = cls._offload_group_key(key)
                group_bytes[gk] = group_bytes.get(gk, 0) + max(0, int(n))
            block_parents = cls._streaming_block_parents(group_bytes.keys())
            block_sizes = [
                n for gk, n in group_bytes.items()
                if cls._block_parent_of(gk) in block_parents
            ]
            unit_bytes = max(block_sizes or sizes)
            window = max(1, int(_env("AI_TOOLKIT_BOUNCE_BLOCK_WINDOW", "2")))
            floor_gib = float(_env("AI_TOOLKIT_BOUNCE_BLOCK_MIN_GIB", "0.75"))
            max_gib = float(_env("AI_TOOLKIT_BOUNCE_BLOCK_MAX_GIB", "1.50"))
            mode = "block"
        else:
            window = max(1, int(_env("AI_TOOLKIT_BOUNCE_LAYER_WINDOW", str(PIPELINE_DEPTH * 4))))
            unit_bytes = sum(sorted(sizes, reverse=True)[:window])
            floor_gib = float(_env("AI_TOOLKIT_BOUNCE_LAYER_MIN_GIB", "1.00"))
            max_gib = float(_env("AI_TOOLKIT_BOUNCE_LAYER_MAX_GIB", "2.00"))
            mode = "layer"

        raw_gib = (unit_bytes * window * slack) / gib if block_stream_only else (unit_bytes * slack) / gib
        budget_gib = min(max_gib, max(floor_gib, raw_gib))
        target_fraction = float(_env("AI_TOOLKIT_BOUNCE_TARGET_FRACTION", "0.60"))
        target_gib = min(budget_gib, max(floor_gib * 0.5, budget_gib * target_fraction))
        return budget_gib, target_gib, mode

    @classmethod
    def _desired_pin_bytes_for_offload_ids(cls, module, offload_ids, pinned_weight_gib):
        """Requested pin-budget bytes, scoped to the layers ACTUALLY selected
        for streaming (``offload_ids``) -- never the whole model.

        Single source of truth for the auto/explicit pin-sizing math: sum
        weight+bias bytes over only the selected offload ids (auto mode,
        ``*1.03`` slack) or honor an explicit ``pinned_weight_gib`` config
        value. ``attach_smart_training`` previously sized auto-pin from
        ``plan["model_bytes"]`` (the WHOLE model), over-requesting whenever a
        smart-training plan keeps some blocks resident -- this is the fix
        (ticket 534ea49 Phase 2 Slice A). ``attach`` uses this too, so the two
        entry points can never drift apart again.

        Bytes are PHYSICAL storage bytes (``_tensor_storage_bytes`` walks
        wrapper leaves): a quanto/torchao FP8 weight reports the LOGICAL
        dtype through ``numel()*element_size()`` (bf16, 2 bytes/elem) --
        exactly 2x its real 1-byte qdata -- which produced a nonsense
        want=20.42 GiB request for a ~9.9 GiB model on a live run.
        """
        auto_pin = pinned_weight_gib is None
        try:
            auto_pin = auto_pin or float(pinned_weight_gib) < 0
        except (TypeError, ValueError):
            auto_pin = True
        if not auto_pin:
            return int(max(0.0, float(pinned_weight_gib)) * (1024 ** 3))
        managed_bytes = 0
        for _name, child in module.named_modules():
            if id(child) not in offload_ids:
                continue
            for param_name in ("weight", "bias"):
                param = getattr(child, param_name, None)
                if isinstance(param, torch.nn.Parameter):
                    managed_bytes += cls._tensor_storage_bytes(param.data)
        return int(managed_bytes * 1.03)

    @classmethod
    def _planned_bounce_reserve_bytes(
        cls,
        module,
        offload_ids,
        device,
        *,
        block_stream_only=False,
    ):
        if not (_OFFLOAD_PREFETCH_ENABLED and torch.device(device).type == "cuda"):
            return 0
        gib = 1024 ** 3
        env_budget = _env("AI_TOOLKIT_BOUNCE_POOL_GIB", None)
        if env_budget is not None:
            bounce_budget_gib = float(env_budget)
        else:
            bounce_budget_gib, _target_gib, _mode = cls._training_bounce_pool_budget_defaults(
                module,
                offload_ids,
                block_stream_only=block_stream_only,
            )
        bounce_budget_gib = min(
            float(_env("AI_TOOLKIT_BOUNCE_MAX_POOL_GIB", "6.0")),
            bounce_budget_gib,
        )
        return int(max(0.0, bounce_budget_gib) * gib)

    def memory_managed_to(self, *args, **kwargs):
        # Parse dtype from the supported Module.to(...) forms.
        dtype = kwargs.get("dtype")
        if dtype is None:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break

        # Parse target device.
        target_device = kwargs.get("device")
        if target_device is None:
            for arg in args:
                if isinstance(arg, (torch.device, str)):
                    target_device = arg
                    break

        immutable_backend = bool(
            getattr(self.module, "_mm_immutable_backend", False)
        )

        if immutable_backend:
            # Canonical Parameters are permanent CPU arena views. A whole-model
            # dtype conversion would mutate or replace those views, so dtype is
            # intentionally ignored here. The checkpoint was already loaded and
            # quantized/cast to its canonical storage dtype before arena creation.
            if target_device is None:
                return self.module

            target_device = torch.device(target_device)

            # Move only noncanonical singleton state:
            #
            #   - norms, embeddings, modulation layers, buffers, etc.
            #   - singleton leaves selected as resident by the legacy singleton
            #     manager
            #
            # Canonical leaves and streamed singleton leaves are skipped by
            # _move_unmanaged_parameters().
            MemoryManager._move_unmanaged_parameters(
                self.module,
                target_device,
            )

            # The first CUDA placement realizes the immutable cold-start residency
            # plan. That plan was already computed from:
            #
            #   free VRAM
            #   - WDDM planning margin
            #   - cold-start working reserve
            #   - stream/ring requirement
            #
            # Do not recompute residency here and do not infer it from
            # unmanaged_modules.
            if (
                target_device.type == "cuda"
                and not getattr(
                    self,
                    "_immutable_initial_placement_done",
                    False,
                )
            ):
                residency = getattr(
                    self.module,
                    "_mm_residency_state",
                    None,
                )
                training_plan = getattr(
                    self.module,
                    "_mm_immutable_training_plan",
                    None,
                )

                if residency is None or training_plan is None:
                    raise RuntimeError(
                        "Immutable arena is active, but its residency state or "
                        "cold-start training plan is missing"
                    )

                # Another explicit phase boundary may already have activated a
                # training or sampling plan. Never overwrite such a phase merely
                # because generic framework code called model.to(cuda).
                if residency.plan.phase == "empty":
                    residency.reconcile(training_plan)

                self._immutable_initial_placement_done = True

            return self.module

        # ------------------------------------------------------------------
        # Legacy offload backends
        # ------------------------------------------------------------------

        # Device-only moves need special handling for TorchAO Parameters.
        if target_device is not None and dtype is None:
            MemoryManager._move_unmanaged_parameters(
                self.module,
                target_device,
            )
        else:
            for module in self.unmanaged_modules:
                if isinstance(module, torch.nn.Parameter):
                    module.data = module.data.to(*args, **kwargs)
                else:
                    module.to(*args, **kwargs)

        if dtype is not None:
            return self.module._mm_to(dtype=dtype)

        return self.module

    @classmethod
    def attach(
        cls,
        module: torch.nn.Module,
        device: torch.device,
        offload_percent: float = 1.0,  # fraction of streamable layers to offload; selected evenly-spaced over execution order (see attach loop)
        ignore_modules: list[torch.nn.Module] = [],
        _offload_module_ids: set[int] | None = None,
        training_strategy: str = "percent",
        pinned_weight_gib: float | None = None,
        use_pinned_arena: bool = False,
    ):
        if hasattr(module, "_memory_manager"):
            # already attached
            return

        module._memory_manager = cls(module, device, pinned_weight_gib=pinned_weight_gib)
        # remember how we were attached so we can re-attach identically after a temporary
        # detach (see inference_resident).
        module._memory_manager._attach_args = {
            "device": device,
            "offload_percent": offload_percent,
            "ignore_modules": list(ignore_modules),
            "training_strategy": training_strategy,
            "pinned_weight_gib": pinned_weight_gib,
            "use_pinned_arena": use_pinned_arena,
        }

        # override the to method to handle memory management
        module._mm_to = module.to
        module.to = module._memory_manager.memory_managed_to

        # add ignore modules to unmanaged list
        for im in ignore_modules:
            module._memory_manager.unmanaged_modules.append(im)

        # count ignore modules as processed
        modules_processed = [x for x in ignore_modules]

        # Decide which streamable leaf layers to offload (stream from CPU) vs
        # keep resident. If the caller pinned an explicit id set, honor it.
        # Otherwise pick an evenly-spaced subset over execution/registration
        # order sized to `offload_percent`:
        #   1.0 -> stream everything (default)
        #   0.5 -> every other layer (resident, streamed, resident, ...)
        #   0.0 -> keep everything resident
        # The 0.5 interleave deliberately places a resident layer before each
        # streamed one: that is the layout a future depth-1 prefetch would use
        # (kick the next layer's H2D off during the resident layer's compute).
        # NOTE: this is a count fraction, not a byte/VRAM budget; MLP layers are
        # larger than attn projections, so bytes-resident ~= count-resident only
        # approximately. Refine to a byte budget here if that matters later.
        selected_offload_ids = _offload_module_ids
        if selected_offload_ids is None:
            ignore_ids = {id(im) for im in ignore_modules}
            eligible = [
                child
                for _n, child in module.named_modules()
                if (
                    child.__class__.__name__ in LINEAR_MODULES
                    or child.__class__.__name__ in CONV_MODULES
                )
                and id(child) not in ignore_ids
            ]
            p = max(0.0, min(1.0, offload_percent))
            selected_offload_ids = set()
            for i, child in enumerate(eligible):
                # offload when the running quota crosses an integer boundary;
                # this spreads the offloaded layers evenly instead of clumping.
                if int((i + 1) * p) > int(i * p):
                    selected_offload_ids.add(id(child))

        # Pin budget: pin the offloaded weights, capped by the WDDM shared
        # pinned-memory proxy after reserving the planned bounce-pool window.
        # A positive config value is a requested budget, not permission to starve
        # bounce; set the bounce pool budget to 0 if the pool should get no share.
        desired_pin_bytes = cls._desired_pin_bytes_for_offload_ids(
            module, selected_offload_ids, pinned_weight_gib
        )
        # Re-attach with a live arena: request the FULL desired, not a
        # `desired - committed` delta. plan_budgets' headroom is already net
        # of the arena's committed bytes (they sit in DXGI usage), so
        # weight_budget = min(desired, usable_headroom) is exactly the room
        # available for NEW pins -- and the arena's build() only ever rebuilds
        # stale/pageable groups (current-pinned blocks are skipped), so a
        # larger-than-needed grant can never over-pin: build()'s own
        # budget_left caps each rebuild. The delta form UNDERSHOOTS: when the
        # streamed set grows a couple of blocks (142->144 layers at the
        # training->sampling boundary), `desired - committed` is a small
        # estimate that caps the grant below the actual bytes those blocks
        # need, forcing them pageable despite ~1.9 GiB of real free headroom
        # (observed live, 768px fp8). Do NOT reintroduce a committed
        # subtraction here OR in _build_pinned_arena -- subtracting on BOTH
        # sides is the original double-count that zeroed the budget.
        requested_pin_bytes = desired_pin_bytes
        if use_pinned_arena and requested_pin_bytes > 0:
            # The DXGI headroom plan_budgets measures still contains torch's
            # retained host-pin cache (every non_blocking staging buffer stays
            # page-locked for the process lifetime once freed). pin_alloc
            # reconciles on refusal, but the arena's per-block pin decision is
            # made from THIS grant -- without reclaiming the cache first the
            # grant undersizes and blocks flip pageable without ever
            # attempting a pin (observed live: 12.70 GiB DXGI usage against
            # an 8.86 GiB ledger -> free=0 -> 3 pageable blocks ->
            # non_pinned_pack under strict ingraph). allow_shrink=False:
            # weights are the lowest pin tier and must not evict bounce.
            pin_manager.reconcile(
                requested_pin_bytes, device=device, allow_shrink=False
            )
        if use_pinned_arena:
            # Ticket 534ea49: the arena pins the whole streamed set once, and
            # a pinned weight bypasses bounce staging entirely (the
            # profile_is_pinned fast path). Reserving a bounce window for the
            # SAME offloaded weights the arena is about to pin double-counts
            # the host-pin budget: plan_budgets hands the arena
            # `usable - bounce_reserve`, so the reservation clips the arena's
            # grant below the streamed set even when real headroom exists
            # (observed live: 1.73 GiB free ledger, yet 2 streamed blocks
            # forced pageable for want of ~0.8 GiB -> non_pinned_pack under
            # strict ingraph). The arena gets first claim on the headroom;
            # bounce still grows dynamically at runtime for the genuinely
            # pageable remainder (unmanaged params, any block the arena could
            # not fit), just without a pre-subtracted reservation.
            bounce_reserve_bytes = 0
        else:
            bounce_reserve_bytes = cls._planned_bounce_reserve_bytes(
                module,
                selected_offload_ids,
                device,
                block_stream_only=False,
            )
        pin_plan = pin_manager.plan_budgets(
            offloaded_weight_bytes=requested_pin_bytes,
            requested_bounce_bytes=bounce_reserve_bytes,
            device=device,
            mode="training",
        )
        budget = int(pin_plan["weight_budget_bytes"])
        if use_pinned_arena:
            # The arena is the SOLE pinner (bounce_reserve=0 above) and pins
            # under the "weights" tier, so size build() to the ACTUAL
            # weight-tier usable headroom -- NOT plan_budgets' headroom_bytes.
            # plan_budgets probes headroom with the generic "unknown" kind,
            # which applies the conservative pct spill reserve (~0.20 x budget,
            # ~3 GiB here); the weight tier's spill reserve is only the ~1 GiB
            # floor, so available_for_pin(kind="weights") is ~2 GiB larger.
            # Using the unknown-kind figure under-grants the weight tier and
            # forced the LAST block of a full 28-block ingraph-training set
            # pageable (11.0 GiB budget vs an ~11.4 GiB set, with ~4 GiB of
            # real DXGI headroom idle) -> non_pinned_pack. Per-block
            # pin_register(kind="weights", required=False) still enforces the
            # true per-block DXGI limit + reserve, so a generous budget can
            # never cross the cliff: build() only (re)builds stale/pageable
            # groups and a per-block pin over the reserve falls back to
            # pageable exactly as a tight budget_bytes would have. `desired`
            # (bytes x 1.03) also undercounts each flat's 256B leaf alignment +
            # 4096B register page-padding, so it is a floor, not a cap.
            weight_tier_usable = pin_manager.available_for_pin(
                kind="weights", device=device
            )
            if weight_tier_usable is not None:
                budget = max(budget, int(weight_tier_usable))
        # Ticket 534ea49 Phase 2 Slice B2: when the arena is active it is the
        # SOLE pinner. Give the per-layer deferred attach below a budget of 0
        # so it never cudaHostRegisters anything; _build_pinned_arena spends
        # the real `budget` itself afterward, once, per block -- otherwise
        # first attach would pin per-tensor, then unpin, then re-pin as flats
        # (the exact churn this arena exists to remove).
        module._memory_manager.pinned_weight_budget_bytes = 0 if use_pinned_arena else budget
        module._memory_manager._pin_plan = pin_plan

        # attach to all modules. The actual per-layer attach (which consumes the
        # pinned-weight budget greedily) is deferred to `deferred_attach` and run
        # afterward in an interleaved order -- see below.
        deferred_attach = []
        for name, sub_module in module.named_modules():
            for child_name, child_module in sub_module.named_modules():
                if (
                    child_module.__class__.__name__ in LINEAR_MODULES
                    and child_module not in modules_processed
                ):
                    skip = id(child_module) not in selected_offload_ids
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        deferred_attach.append(("linear", child_module))
                    modules_processed.append(child_module)
                elif (
                    child_module.__class__.__name__ in CONV_MODULES
                    and child_module not in modules_processed
                ):
                    skip = id(child_module) not in selected_offload_ids
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        deferred_attach.append(("conv", child_module))
                    modules_processed.append(child_module)
                elif child_module.__class__.__name__ in UNMANAGED_MODULES or any(
                    inc in child_module.__class__.__name__
                    for inc in UNMANAGED_MODULES_INCLUDES
                ):
                    # unmanaged
                    module._memory_manager.unmanaged_modules.append(child_module)
                else:
                    continue

        # Run the deferred attaches in interleaved (not execution) order. Pinning
        # is consumed greedily as each layer attaches, so processing strictly in
        # execution order would front-load every pinned layer at one end of the
        # stream and leave a long unpinned (bounce-only) tail with no resident/
        # pinned neighbor nearby. Repeated transformer blocks make same-role
        # layers byte-identical, so which specific layers get pinned first is
        # free to choose -- spread them evenly instead.
        gib = 1024 ** 3
        dxgi_pin_before = cls._dxgi_shared_budget_snapshot_bytes(device)
        ledger_pin_before = bounce_pool._pinned_bytes_total
        n_deferred = len(deferred_attach)
        for i, (kind, child_module) in sorted(
            enumerate(deferred_attach),
            key=lambda pair: cls._interleave_priority(pair[0], n_deferred),
        ):
            if kind == "linear":
                LinearLayerMemoryManager.attach(child_module, module._memory_manager)
                # attach to ARA as well
                if hasattr(child_module, "ara_lora_ref"):
                    ara = child_module.ara_lora_ref()
                    if ara not in modules_processed:
                        MemoryManager.attach(ara, device)
            else:
                ConvLayerMemoryManager.attach(child_module, module._memory_manager)
                # attach to ARA as well
                if hasattr(child_module, "ara_lora_ref"):
                    ara = child_module.ara_lora_ref()
                    if ara not in modules_processed:
                        MemoryManager.attach(ara, device)
                        modules_processed.append(ara)
        dxgi_pin_after = cls._dxgi_shared_budget_snapshot_bytes(device)
        if cls._diagnostics_enabled() and dxgi_pin_before is not None and dxgi_pin_after is not None:
            pinned_layers = sum(
                1 for _kind, child in deferred_attach
                if int(getattr(child, "_mm_pinned_bytes", 0) or 0) > 0
            )
            pinned_bytes = sum(
                int(getattr(child, "_mm_pinned_bytes", 0) or 0)
                for _kind, child in deferred_attach
            )
            print(
                "[MemoryManager] DXGI attach pin delta: "
                f"usage={dxgi_pin_before['usage_bytes'] / gib:.2f}->{dxgi_pin_after['usage_bytes'] / gib:.2f} GiB "
                f"headroom={dxgi_pin_before['usable_headroom_bytes'] / gib:.2f}->{dxgi_pin_after['usable_headroom_bytes'] / gib:.2f} GiB "
                f"pinned_layers={pinned_layers}/{len(deferred_attach)} "
                f"pinned_bytes={pinned_bytes / gib:.2f} GiB "
                f"ledger_delta={(bounce_pool._pinned_bytes_total - ledger_pin_before) / gib:.2f} GiB "
                f"match={dxgi_pin_after.get('match_method')}"
            )

        # Assign each streamable candidate a stable identity from its module path.
        # The trace scheduler keys on this rather than id(weight), which would not
        # survive the Parameter replacement that sampling detach/restore does.
        for name, child in module.named_modules():
            if child.__class__.__name__ in LINEAR_MODULES or child.__class__.__name__ in CONV_MODULES:
                child._mm_layer_key = name or child.__class__.__name__
        arena_stats = None
        if use_pinned_arena:
            arena_stats = cls._build_pinned_arena(
                module, budget_bytes=budget, priority_ids=selected_offload_ids,
            )
        cls._refresh_resident_trace_hooks(module, module._memory_manager)
        if arena_stats is not None and cls._diagnostics_enabled():
            gib = 1024 ** 3
            print(
                "[MemoryManager] pinned arena: "
                f"blocks={arena_stats.blocks} "
                f"pinned={arena_stats.pinned_bytes / gib:.2f} GiB "
                f"pageable_blocks={arena_stats.pageable_blocks} "
                f"pageable={arena_stats.pageable_bytes / gib:.2f} GiB"
            )
        if cls._diagnostics_enabled():
            gib = 1024 ** 3
            managed = sum(
                1 for child in module.modules()
                if hasattr(child, "_layer_memory_manager")
            )
            dxgi_text = _dxgi_attach_log_text(bounce_pool._cuda_device_index(device))
            print(
                f"[MemoryManager] training offload attached: "
                f"managed_layers={managed} "
                f"pinned_cpu={module._memory_manager.pinned_weight_bytes / gib:.2f} GiB "
                f"pin_budget={module._memory_manager.pinned_weight_budget_bytes / gib:.2f} GiB "
                f"{dxgi_text}"
            )

    @classmethod
    def _build_pinned_arena(
        cls, module: torch.nn.Module, budget_bytes: int = 0, priority_ids=None,
    ):
        """Fold newly-offloaded weights into module._mm_weight_arena (ticket
        534ea49): pin once into persistent per-block flat host buffers instead
        of per-tensor cudaHostRegister. Idempotent AND non-churning across
        re-attach: a child already arena-current (``arena.is_current``, i.e.
        built by an earlier attach and untouched since -- detach() leaves
        arena params alone) is skipped entirely, so re-attaching after a
        detach costs zero pin/repin work, only ever building blocks that are
        genuinely new or were invalidated. (Assumes a block's children move
        in and out of currency together, which holds as long as the whole
        block is built in one ``arena.build()`` call, as here.)

        ``budget_bytes`` is the ``plan_budgets`` weight grant: the room
        available for NEW pins right now (plan_budgets' headroom is already
        net of the arena's committed bytes, which sit in DXGI usage, so this
        is min(desired, usable_headroom) -- NOT a delta, and not to be
        reduced by committed again here; subtracting on both sides is the
        original double-count that zeroed the budget). build() only rebuilds
        stale/pageable groups, so a grant larger than the rebuild need can
        never over-pin -- its own budget_left caps each block. The arena is
        the SOLE pinner when active (Slice B2: the per-layer deferred attach
        above ran with a budget of 0), so no unpin pre-pass is needed here.

        ``priority_ids`` (the current attach's selected_offload_ids) are the
        blocks that will actually be STREAMED this phase and therefore
        borrowed by strict ingraph -- build() spends the budget in iteration
        order, so those groups go first. Strict ingraph is all-or-nothing on
        the streamed set: one pageable streamed block fails the whole
        compile, so under a tight budget the pageable fallback must land on
        resident-during-this-phase blocks (harmless -- they are not borrowed)
        rather than on a streamed one.
        """
        from .pinned_arena import PinnedWeightArena

        arena = getattr(module, "_mm_weight_arena", None)
        if arena is None:
            arena = PinnedWeightArena()
            module._mm_weight_arena = arena
        # Group ALL managed children by block key first, then rebuild any
        # group with at least one non-current member as a WHOLE. Rebuilding
        # with only the missing children (the original approach) bumps the
        # block's generation and strands its previously-built siblings as
        # stale -- observed live as `borrow refused: stale_modules=5/8` on
        # every block, because the smart-training plan splits blocks
        # per-layer (its resident-growth loops append individual linears),
        # so training built each block from a subset and the sampling
        # attach then "completed" it destructively.
        priority_ids = set(priority_ids or ())
        groups: dict = {}
        group_has_stale: dict = {}
        group_is_streamed: dict = {}
        for name, child in module.named_modules():
            if not hasattr(child, "_layer_memory_manager"):
                continue
            key = getattr(child, "_mm_layer_key", None) or name
            group_key = cls._offload_group_key(key)
            groups.setdefault(group_key, []).append((key, child))
            if id(child) in priority_ids:
                group_is_streamed[group_key] = True
            if not arena.is_current(child):
                group_has_stale[group_key] = True
            elif budget_bytes and budget_bytes > 0:
                # Self-healing retry: a block that fell back to a PAGEABLE
                # flat in an earlier tight-budget build stays current (its
                # params are valid views), so the stale check alone would
                # never revisit it -- strict ingraph then fails on it
                # forever even after headroom recovers. Rebuild pageable
                # groups whenever this attach actually has new budget.
                pack_key = arena.arena_block_of(child)
                if pack_key is not None:
                    pack = arena.block_pack(pack_key)
                    if pack is not None and not pack.pinned:
                        group_has_stale[group_key] = True
        # Streamed (to-be-borrowed) groups first so a tight budget spends its
        # pins where strict ingraph requires them; resident-this-phase groups
        # take any pageable fallback.
        rebuild_keys = [k for k in groups if group_has_stale.get(k)]
        rebuild_keys.sort(key=lambda k: 0 if group_is_streamed.get(k) else 1)
        entries_by_block = {k: groups[k] for k in rebuild_keys}
        if entries_by_block:
            # ``budget_bytes`` is ALREADY the new-bytes delta: attach()
            # subtracts the arena's committed bytes from the plan_budgets
            # request (they are already in DXGI usage, so the measured
            # headroom is net of them). Do NOT subtract committed again
            # here -- that double-count zeroed the budget on every sampling
            # re-attach and forced all block rebuilds pageable. Blocks being
            # rebuilt release their old flat, which build() credits back
            # internally.
            arena.build(entries_by_block, budget_bytes=int(budget_bytes))
        module._memory_manager.pinned_weight_budget_bytes = arena.committed_pinned_bytes()
        module._memory_manager.pinned_weight_bytes = arena.committed_pinned_bytes()
        return arena.stats()

    @classmethod
    def _destroy_pinned_arena(cls, module: torch.nn.Module) -> None:
        """Explicitly tear down ``module._mm_weight_arena`` (ticket 534ea49
        Phase 2 Slice D).

        NOT called from ``detach``/sampling boundaries -- the whole point of
        the arena is that it persists across those (see pinned_arena.py's
        module docstring); tearing it down there would reintroduce the
        unpin/repin churn this ticket removes. Only for genuine model
        unload or test teardown.

        Order matters: every arena-backed param is first detached onto
        standalone (non-arena) storage via a clone, THEN the arena's packs
        are released. Releasing pinned bytes while live params still view
        that storage would decrement the ledger for memory that is still
        page-locked and referenced -- a real leak dressed up as a clean
        teardown.
        """
        arena = getattr(module, "_mm_weight_arena", None)
        if arena is None:
            return
        for child in module.modules():
            if getattr(child, "_mm_arena_block", None) is None:
                continue
            for name in ("weight", "bias"):
                param = getattr(child, name, None)
                if not isinstance(param, torch.nn.Parameter):
                    continue
                data = param.data
                if _is_quantized_tensor(data) or hasattr(data, "__tensor_flatten__"):
                    cloned = _rebuild_from_leaves(
                        data, (leaf.clone() for leaf in _flatten_leaves(data))
                    )
                else:
                    cloned = data.clone()
                setattr(
                    child, name,
                    torch.nn.Parameter(cloned, requires_grad=param.requires_grad),
                )
            del child._mm_arena_block
            if hasattr(child, "_mm_arena_generation"):
                del child._mm_arena_generation
        arena.release()
        del module._mm_weight_arena

    @classmethod
    def detach(cls, module: torch.nn.Module):
        """
        Reverse of attach(). Moves unmanaged modules back to CPU, restores the
        original .to() and forward methods on all child layers, unpins CPU weight
        tensors, and clears the global CUDA device state.

        Call this before unloading/replacing a module that had attach() applied.
        """
        if not hasattr(module, "_memory_manager"):
            return

        pool = getattr(module._memory_manager, "_prefetch_pool", None)
        if pool is not None:
            bounce_pool.destroy_pool(pool.device)
            module._memory_manager._prefetch_pool = None
        cls._clear_resident_trace_hooks(module._memory_manager)

        for unmanaged in module._memory_manager.unmanaged_modules:
            try:
                if isinstance(unmanaged, torch.nn.Parameter):
                    unmanaged.data = unmanaged.data.to('cpu')
                else:
                    unmanaged.to('cpu')
            except Exception:
                pass

        if hasattr(module, "_mm_to"):
            module.to = module._mm_to
            del module._mm_to

        del module._memory_manager

        for child in module.modules():
            lmm = getattr(child, "_layer_memory_manager", None)
            if lmm is None:
                continue

            original_forward = getattr(lmm, "_original_forward", None)
            if original_forward is not None:
                # Unwind our streaming forward from wherever it lives NOW: a
                # LoRA applied after attach moved it into the LoRA's
                # org_forward slot, and writing the original into the
                # attach-time slot would overwrite the LoRA hijack
                # (lora_hijack_missing at the sampling boundary).
                uninstall = getattr(lmm, "_uninstall_base_forward", None)
                if uninstall is not None:
                    uninstall()
                elif hasattr(child, "ara_lora_ref"):
                    ara = child.ara_lora_ref()
                    if ara is not None:
                        ara.org_forward = original_forward
                else:
                    child.forward = original_forward

            try:
                unpin_layer(child)
            except Exception:
                pass
            # Arena-backed children (ticket 534ea49): the weight/bias views
            # belong to a persistent per-block flat the arena still owns.
            # Cloning them here (like the loop below does for ordinary
            # per-tensor pins) would detach the param from the arena AND
            # double-release its bytes from the "weights" ledger, since the
            # arena's own bytes were never counted against child._mm_pinned_bytes
            # in the first place. Leave arena params untouched; the arena
            # persists across this detach/attach cycle by design.
            if getattr(child, "_mm_arena_block", None) is None:
                for param_name in ("weight", "bias"):
                    param = getattr(child, param_name, None)
                    if param is None or not isinstance(param, torch.nn.Parameter):
                        continue
                    try:
                        if _is_quantized_tensor(param.data):
                            _unpin_inner_tensors(param.data)
                        if param.data.is_pinned():
                            bounce_pool.release_pinned_bytes(
                                param.data.numel() * param.data.element_size(),
                                kind="weights",
                            )
                            object.__setattr__(
                                child,
                                param_name,
                                torch.nn.Parameter(
                                    param.data.clone(),
                                    requires_grad=param.requires_grad,
                                ),
                            )
                    except Exception:
                        pass
            child._mm_pinned_bytes = 0

            del child._layer_memory_manager
            if hasattr(child, "_memory_management_device"):
                del child._memory_management_device
            for param in child.parameters(recurse=False):
                if hasattr(param, "_is_memory_managed"):
                    del param._is_memory_managed

    @staticmethod
    def _move_tensor_subclass(tensor, device):
        """Rebuild a wrapper tensor with each flattened inner tensor moved."""
        try:
            names, attributes = tensor.__tensor_flatten__()
        except Exception:
            return tensor.to(device)
        moved = {}
        for name in names:
            inner = getattr(tensor, name)
            if inner is None:
                moved[name] = None
            elif hasattr(inner, "__tensor_flatten__"):
                moved[name] = MemoryManager._move_tensor_subclass(inner, device)
            else:
                moved[name] = inner.to(device)
        return type(tensor).__tensor_unflatten__(
            moved, attributes, tensor.size(), tensor.stride()
        )

    @staticmethod
    def _move_quantized_parameters(module, device):
        """Move tensor-subclass weights by replacing each complete Parameter."""
        target = torch.device(device)
        for child in module.modules():
            if getattr(child, "_mm_canonical_leaf", False):
                continue
            for name, param in list(child._parameters.items()):
                if param is None or not _is_quantized_tensor(param.data):
                    continue
                moved = MemoryManager._move_tensor_subclass(param.data, target)
                replacement = torch.nn.Parameter(
                    moved, requires_grad=param.requires_grad
                )
                child._parameters[name] = replacement
                # TorchAO installs a direct instance attribute that shadows Module._parameters.
                # Keep both references aligned or Linear.forward reads the stale CPU wrapper.
                if name in child.__dict__:
                    object.__setattr__(child, name, replacement)

    @staticmethod
    def _sync_shadowed_parameters(module):
        """TorchAO shadows Linear weight/bias with direct instance attributes."""
        for child in module.modules():
            for name, param in child._parameters.items():
                if param is not None and name in child.__dict__:
                    object.__setattr__(child, name, param)

    @staticmethod
    def _move_module_parameters(module, device):
        """Move a module without Module._apply swapping TorchAO Parameters."""
        target = torch.device(device)
        MemoryManager._move_quantized_parameters(module, target)
        for child in module.modules():
            if getattr(child, "_mm_canonical_leaf", False):
                continue
            for param in child._parameters.values():
                if param is None or _is_quantized_tensor(param.data):
                    continue
                param.data = param.data.to(target)
            for name, buffer in list(child._buffers.items()):
                if buffer is not None:
                    child._buffers[name] = buffer.to(target)
        MemoryManager._sync_shadowed_parameters(module)

    @staticmethod
    def _move_unmanaged_parameters(module, device):
        """Move only parameters not owned by a streaming layer manager."""
        target = torch.device(device)
        for child in module.modules():
            if hasattr(child, "_layer_memory_manager"):
                continue
            if getattr(child, "_mm_canonical_leaf", False):
                # Immutable-arena leaves are permanent CPU views. Device
                # residency lives in manager-owned sidecars; every whole-model
                # movement path must leave the canonical Parameter untouched.
                continue
            if getattr(child, "_mm_ingraph_pack_source", False):
                # Ingraph-streamed linear: its manager hijack was stripped for
                # the compile region, but its weights are pack sources that
                # must stay on CPU -- the trunk streams them from the pinned
                # pack. Moving them here silently hauls the whole model onto
                # the card (observed: 12.23 GiB and a WDDM spill). Only the
                # STREAMED leaves carry this mark; a block's resident leaves are
                # supposed to move, and a compiled trunk reads them here.
                continue
            for name, param in list(child._parameters.items()):
                if param is None:
                    continue
                if _is_quantized_tensor(param.data):
                    moved = MemoryManager._move_tensor_subclass(param.data, target)
                    replacement = torch.nn.Parameter(
                        moved, requires_grad=param.requires_grad
                    )
                    # A quantized Parameter cannot be moved in place, so it is
                    # swapped for a new object. Anything holding the old one --
                    # notably an in-graph trunk, whose resident leaves are
                    # captured at enable time -- keeps the old tensor. For a
                    # FROZEN base that is benign (identical values, and the old
                    # storage stays alive), but a cpu->cuda move here strands the
                    # trunk on the host: see _assert_ingraph_training_current.
                    child._parameters[name] = replacement
                    if name in child.__dict__:
                        object.__setattr__(child, name, replacement)
                else:
                    param.data = param.data.to(target)
            for name, buffer in list(child._buffers.items()):
                if buffer is not None:
                    child._buffers[name] = buffer.to(target)
        MemoryManager._sync_shadowed_parameters(module)

    @staticmethod
    def _tensor_storage_bytes(tensor):
        """Count physical leaves of wrapper tensors such as TorchAO weights."""
        if tensor is None:
            return 0
        try:
            names, _ = tensor.__tensor_flatten__()
        except Exception:
            return tensor.numel() * tensor.element_size()
        total = 0
        for name in names:
            inner = getattr(tensor, name, None)
            if inner is not None:
                total += MemoryManager._tensor_storage_bytes(inner)
        return total

    @classmethod
    def _direct_module_bytes(cls, module):
        total = sum(
            cls._tensor_storage_bytes(param.data)
            for param in module.parameters(recurse=False)
        )
        total += sum(
            cls._tensor_storage_bytes(buffer)
            for buffer in module.buffers(recurse=False)
        )
        return total

    @classmethod
    def _module_bytes(cls, module):
        return sum(cls._direct_module_bytes(child) for child in module.modules())

    @classmethod
    def _stream_bytes(cls, module):
        """Estimate the GPU staging bytes for one streamed layer."""
        total = 0
        for param in module.parameters(recurse=False):
            if _is_quantized_tensor(param.data):
                if (
                    module.__class__.__name__ in LINEAR_MODULES
                    and hasattr(param.data, "qdata")
                ):
                    # Sampling transfers FP8 bytes and computes directly in FP8.
                    total += cls._tensor_storage_bytes(param.data)
                else:
                    total += param.numel() * 2
            else:
                total += cls._tensor_storage_bytes(param.data)
        return total

    @classmethod
    def _training_stream_bytes(cls, module):
        """GPU bytes used when a training fetch materializes this layer."""
        total = 0
        for param in module.parameters(recurse=False):
            if _is_quantized_tensor(param.data):
                total += param.numel() * 2
            else:
                total += cls._tensor_storage_bytes(param.data)
        return total

    @staticmethod
    def _first_tensor_output(output):
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)):
            for item in output:
                found = MemoryManager._first_tensor_output(item)
                if found is not None:
                    return found
        if isinstance(output, dict):
            for item in output.values():
                found = MemoryManager._first_tensor_output(item)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _trace_bytes_for_module(module):
        weight = getattr(module, "weight", None)
        if weight is None:
            return 0, 0
        try:
            storage = MemoryManager._tensor_storage_bytes(weight.data)
            materialized = weight.numel() * 2 if _is_quantized_tensor(weight.data) else storage
            return storage, materialized
        except Exception:
            return 0, 0

    @classmethod
    def _install_resident_trace_hook(cls, module, key):
        fp8_bytes, materialized_bytes = cls._trace_bytes_for_module(module)

        def _pre_hook(_module, _inputs, _key=key, _fp8=fp8_bytes, _mat=materialized_bytes):
            record_weight_access(_key, "forward", _fp8, _mat)

        def _post_hook(_module, _inputs, output, _key=key, _fp8=fp8_bytes, _mat=materialized_bytes):
            tensor = MemoryManager._first_tensor_output(output)
            if tensor is None or not getattr(tensor, "requires_grad", False):
                return

            def _backward_hook(grad):
                record_weight_access(_key, "backward", _fp8, _mat)
                return grad

            try:
                tensor.register_hook(_backward_hook)
            except Exception:
                pass

        return (
            module.register_forward_pre_hook(_pre_hook),
            module.register_forward_hook(_post_hook),
        )

    @classmethod
    def _clear_resident_trace_hooks(cls, mm):
        hooks = getattr(mm, "_resident_trace_hooks", None)
        if not hooks:
            return
        for handles in list(hooks.values()):
            for handle in handles:
                try:
                    handle.remove()
                except Exception:
                    pass
        hooks.clear()

    @classmethod
    def _refresh_resident_trace_hooks(cls, module, mm):
        """Trace resident streamable layers so future demotion can reuse order."""
        if mm is None:
            return
        if getattr(module, "_mm_sampling_disable_resident_trace_hooks", False):
            cls._clear_resident_trace_hooks(mm)
            return
        args = getattr(mm, "_attach_args", {}) or {}
        pinned_keys = set(getattr(mm, "_training_pinned_resident_keys", set()))
        current = getattr(mm, "_resident_trace_hooks", None)
        if current is None:
            current = {}
            mm._resident_trace_hooks = current
        wanted = {}
        for item in cls._training_layout_candidates(
            module, args.get("ignore_modules", []), pinned_keys
        ):
            child = item["module"]
            if item["managed"]:
                continue
            key = item["name"]
            child._mm_layer_key = key
            wanted[id(child)] = (child, key)
        for child_id in list(current):
            if child_id in wanted:
                continue
            for handle in current.pop(child_id):
                try:
                    handle.remove()
                except Exception:
                    pass
        for child_id, (child, key) in wanted.items():
            existing_key = getattr(child, "_mm_resident_trace_key", None)
            if child_id in current and existing_key == key:
                continue
            if child_id in current:
                for handle in current.pop(child_id):
                    try:
                        handle.remove()
                    except Exception:
                        pass
            child._mm_resident_trace_key = key
            current[child_id] = cls._install_resident_trace_hook(child, key)
    @classmethod
    def _enable_fp8_sampling(cls, module):
        """Install native FP8 forwards without bypassing attached LoRA modules."""
        restores = []
        resident_layers = 0
        streamed_layers = 0
        for child in module.modules():
            if child.__class__.__name__ not in LINEAR_MODULES:
                continue
            weight = getattr(child, "weight", None)
            if (
                not isinstance(weight, torch.nn.Parameter)
                or not hasattr(weight.data, "qdata")
                or weight.data.qdata.dtype != torch.float8_e4m3fn
            ):
                continue
            if hasattr(child, "_layer_memory_manager"):
                child._memory_management_fp8_sampling = True
                streamed_layers += 1
                continue

            container = child
            attribute = "forward"
            if hasattr(child, "ara_lora_ref"):
                owner = child.ara_lora_ref()
                if owner is not None and hasattr(owner, "org_forward"):
                    container, attribute = owner, "org_forward"
            else:
                owner = getattr(getattr(child, "forward", None), "__self__", None)
                if (
                    owner is not None
                    and owner is not child
                    and hasattr(owner, "org_forward")
                ):
                    container, attribute = owner, "org_forward"

            # Validate once, here, so the installed forward can be pure tensor
            # math (no per-call capability query, try/except, stats, or Optional
            # return) and torch.compile traces it without graph breaks. Layers
            # that do not qualify keep their original (GPU dequant) forward.
            if not fp8_sampling_qualifies(child.weight):
                continue

            original_forward = getattr(container, attribute)

            # Capture the unpacked FP8 weight (already transposed to (K, N)) and
            # row scale as plain tensors. Sampling weights are frozen and resident
            # for the life of this context, so the closure constants stay valid
            # and the compiled forward never touches the tensor subclass.
            qdata_t = child.weight.qdata.t()
            scale_row = child.weight.scale
            bias_t = getattr(child, "bias", None)

            def _fp8_forward(
                x, *args,
                _qt=qdata_t, _sr=scale_row, _b=bias_t, _original=original_forward,
                **kwargs,
            ):
                # The args/kwargs guard folds at trace time (compiled blocks call
                # the layer as ``layer(x)``), so it is not a graph break.
                if args or kwargs:
                    return _original(x, *args, **kwargs)
                return _fp8_linear_compiled(x, _qt, _sr, _b)

            setattr(container, attribute, _fp8_forward)
            restores.append((container, attribute, original_forward, child))
            resident_layers += 1
        return restores, resident_layers, streamed_layers

    @classmethod
    def _enable_fp8_training_compile(cls, module):
        """Install grad-safe native FP8 forwards for resident training compile."""
        restores = []
        resident_layers = 0
        for child in module.modules():
            if child.__class__.__name__ not in LINEAR_MODULES:
                continue
            weight = getattr(child, "weight", None)
            if (
                not isinstance(weight, torch.nn.Parameter)
                or not hasattr(weight.data, "qdata")
                or weight.data.qdata.dtype != torch.float8_e4m3fn
                or weight.requires_grad
                or hasattr(child, "_layer_memory_manager")
            ):
                continue
            if not fp8_sampling_qualifies(child.weight):
                continue

            container = child
            attribute = "forward"
            if hasattr(child, "ara_lora_ref"):
                owner = child.ara_lora_ref()
                if owner is not None and hasattr(owner, "org_forward"):
                    container, attribute = owner, "org_forward"
            else:
                owner = getattr(getattr(child, "forward", None), "__self__", None)
                if (
                    owner is not None
                    and owner is not child
                    and hasattr(owner, "org_forward")
                ):
                    container, attribute = owner, "org_forward"

            original_forward = getattr(container, attribute)
            qdata_t = child.weight.qdata.t()
            scale_row = child.weight.scale
            bias_t = getattr(child, "bias", None)

            def _fp8_forward(
                x, *args,
                _qt=qdata_t, _sr=scale_row, _b=bias_t, _original=original_forward,
                **kwargs,
            ):
                if args or kwargs:
                    return _original(x, *args, **kwargs)
                return _fp8_linear_training(x, _qt, _sr, _b)

            setattr(container, attribute, _fp8_forward)
            child._memory_management_training_compile_fp8 = True
            restores.append(
                (
                    container,
                    attribute,
                    original_forward,
                    _fp8_forward,
                    child,
                )
            )
            resident_layers += 1
        return restores, resident_layers

    @staticmethod
    def _disable_fp8_training_compile(module, restores):
        for (
            container,
            attribute,
            original_forward,
            installed_forward,
            child,
        ) in reversed(restores):
            current_forward = getattr(container, attribute, None)

            # Restore only if our compiled wrapper still owns the slot.
            if current_forward is installed_forward:
                setattr(container, attribute, original_forward)

            if hasattr(child, "_memory_management_training_compile_fp8"):
                del child._memory_management_training_compile_fp8

        for child in module.modules():
            if hasattr(child, "_memory_management_training_compile_fp8"):
                del child._memory_management_training_compile_fp8

    @staticmethod
    def _disable_fp8_sampling(module, restores):
        for container, attribute, original_forward, _child in reversed(restores):
            setattr(container, attribute, original_forward)
        for child in module.modules():
            if hasattr(child, "_memory_management_fp8_sampling"):
                del child._memory_management_fp8_sampling

    @staticmethod
    def _release_fp8_sampling_for(child_ids, restores):
        """Restore the original forwards of specific layers, in place.

        The resident FP8 sampling forwards hold the GPU qdata/scale as closure
        constants (a deliberate compile optimization -- see _enable_fp8_sampling).
        That closure is an untracked pin: demoting the layer moves param.data to
        CPU but the closure view keeps the GPU storage alive, so the demote
        frees NOTHING (observed at 2000px: 12 demote rounds, zero device-free
        gain). Any demote of an fp8-sampling layer must drop its closure first.
        Removes the released entries from ``restores`` so teardown does not
        re-restore over the streaming hijack installed afterwards.
        """
        kept = []
        released = 0
        for entry in restores:
            container, attribute, original_forward, child = entry
            if id(child) in child_ids:
                setattr(container, attribute, original_forward)
                released += 1
            else:
                kept.append(entry)
        restores[:] = kept
        return released

    @classmethod
    def _sampling_candidates(cls, module, ignore_modules):
        ignored = {id(item) for item in ignore_modules}
        candidates = []
        seen = set()
        for child in module.modules():
            if id(child) in ignored or id(child) in seen:
                continue
            if (
                child.__class__.__name__ in LINEAR_MODULES
                or child.__class__.__name__ in CONV_MODULES
            ):
                seen.add(id(child))
                candidates.append(
                    (child, cls._direct_module_bytes(child), cls._stream_bytes(child))
                )
        return candidates

    @staticmethod
    def _offload_group_key(name: str) -> str:
        """Collapse a layer's module path to its offload group key.

        The key is the path prefix up to and including the first numeric
        (ModuleList index) segment, so every Linear inside one repeated
        transformer block (e.g. ``blocks.7.attn.wq`` and ``blocks.7.mlp.down``)
        shares the key ``blocks.7`` and is offloaded or kept resident together.
        Layers with no numeric segment (``first``, ``last.linear``) are their
        own singleton group.

        Whole-block residency is what makes a block torch.compile-able for the
        sampler: a block with even one streamed Linear carries an offload hook
        (_BouncingLinearFn) and must stay eager. It also makes the runtime
        stream predictable block-sized groups instead of scattered layers.
        """
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part.isdigit():
                return ".".join(parts[: i + 1])
        return name

    @staticmethod
    def _block_parent_of(group_key: str):
        """Parent prefix of a ``prefix.N`` block group key, else ``None``.

        ``blocks.7`` -> ``blocks``; ``final_layer.proj`` (no trailing index)
        -> ``None``. Used to tell a repeated-block layer from a one-off layer.
        """
        head, _, tail = group_key.rpartition(".")
        if head and tail.isdigit():
            return head
        return None

    @classmethod
    def _streaming_block_parents(cls, group_keys) -> set:
        """Parents that own >= 2 indexed children (i.e. a ModuleList of blocks).

        A repeated transformer block (``blocks.0`` .. ``blocks.37``) shows up as
        a parent (``blocks``) with many numeric children. A one-off layer that
        merely sits at a Sequential index (``final_layer.adaLN_modulation.1``,
        the only candidate under that parent) does not, so it is not treated as
        a streaming block.
        """
        children: dict = {}
        for gk in group_keys:
            parent = cls._block_parent_of(gk)
            if parent is not None:
                children.setdefault(parent, set()).add(gk)
        return {p for p, kids in children.items() if len(kids) >= 2}

    @staticmethod
    def _make_block_stage_prehook(device, block_key, members):
        """Forward pre-hook: stage the whole block's streamed weights in one
        transfer-stream burst before its Linears run. Best-effort — any failure
        leaves the per-Linear staging path untouched."""
        def _prehook(_mod, args):
            try:
                compute_dtype = torch.bfloat16
                if args and torch.is_tensor(args[0]):
                    dt = args[0].dtype
                    if dt in (torch.bfloat16, torch.float16, torch.float32):
                        compute_dtype = dt
                linears = [
                    (lk, child.weight, getattr(child, "bias", None))
                    for (lk, child) in members
                ]
                stage_block_forward(device, block_key, linears, compute_dtype)
            except Exception:
                pass
            return None
        return _prehook

    @staticmethod
    def _make_block_done_hook(device, block_key):
        """Forward post-hook: mark the block's compute done so its staged buffers
        can be reclaimed once the matmuls that read them complete."""
        def _hook(_mod, _args, _output):
            try:
                block_forward_done(device, block_key)
            except Exception:
                pass
            return None
        return _hook

    @classmethod
    def _wire_block_stream_forward_hooks(cls, module, device):
        """Register per-block forward pre/post hooks on the streamed transformer
        blocks so each block's weights are staged together. Returns block count.

        Idempotent: removes any previously registered handles first.
        """
        device = torch.device(device)
        if device.type != "cuda":
            return 0
        for handle in getattr(module, "_mm_block_stream_handles", []) or []:
            try:
                handle.remove()
            except Exception:
                pass
        # Group streamed (managed) Linears by their block group key.
        blocks: dict = {}
        for name, child in module.named_modules():
            if not hasattr(child, "_layer_memory_manager"):
                continue
            key = getattr(child, "_mm_layer_key", None) or name
            blocks.setdefault(cls._offload_group_key(key), []).append((key, child))
        block_parents = cls._streaming_block_parents(blocks.keys())
        handles = []
        wired = 0
        for gk, linears in blocks.items():
            if cls._block_parent_of(gk) not in block_parents:
                continue  # singleton/non-block layer — never block-staged
            try:
                block_module = module.get_submodule(gk)
            except AttributeError:
                continue
            # fp8-native forward keeps the per-Linear path; don't block-stage it.
            members = [
                (lk, child) for (lk, child) in linears
                if not getattr(child, "_memory_management_fp8_training", False)
                and not getattr(child, "_memory_management_fp8_sampling", False)
            ]
            if not members:
                continue
            handles.append(block_module.register_forward_pre_hook(
                cls._make_block_stage_prehook(device, gk, members)
            ))
            handles.append(block_module.register_forward_hook(
                cls._make_block_done_hook(device, gk)
            ))
            wired += 1
        module._mm_block_stream_handles = handles
        return wired

    @classmethod
    def _smart_sampling_plan(
        cls, module, free_bytes, working_reserve_bytes, ignore_modules,
        wddm_margin_bytes=0, wddm_hard_bytes=0,
    ):
        """Comfy-style byte budget, but offloading whole blocks rather than
        individual Linears.

        The byte accounting (resident weights + async transfer ring + working_reserve)
        is unchanged; only the unit of offload is coarsened to a block. We
        offload the most expensive blocks first until the resident set plus the
        ring fits. The ring still streams one Linear at a time, so it is sized
        from the largest PIPELINE_DEPTH individual layers among everything
        offloaded — block grouping changes *which* layers stream, not how the
        runtime ring works.
        """
        ignored = {id(item) for item in ignore_modules}
        seen = set()
        # key -> {"resident_bytes", "stream_layers": [bytes...], "ids": [int...]}
        groups: dict = {}
        for name, child in module.named_modules():
            if id(child) in ignored or id(child) in seen:
                continue
            if (
                child.__class__.__name__ in LINEAR_MODULES
                or child.__class__.__name__ in CONV_MODULES
            ):
                seen.add(id(child))
                key = cls._offload_group_key(name)
                group = groups.setdefault(
                    key, {"resident_bytes": 0, "stream_layers": [], "ids": []}
                )
                group["resident_bytes"] += cls._direct_module_bytes(child)
                group["stream_layers"].append(cls._stream_bytes(child))
                group["ids"].append(id(child))

        total_model_bytes = cls._module_bytes(module)
        wddm_hard_bytes = max(0, int(wddm_hard_bytes or 0))
        wddm_margin_bytes = max(wddm_hard_bytes, int(wddm_margin_bytes or 0))
        usable_bytes = max(0, free_bytes - working_reserve_bytes - wddm_margin_bytes)
        resident_bytes = total_model_bytes
        offload_ids: set = set()
        offloaded_stream_layers: list = []
        offloaded_blocks = 0

        for key, group in sorted(
            groups.items(), key=lambda kv: kv[1]["resident_bytes"], reverse=True
        ):
            ring_bytes = sum(
                sorted(offloaded_stream_layers, reverse=True)[:PIPELINE_DEPTH]
            )
            if resident_bytes + ring_bytes <= usable_bytes:
                break
            offload_ids.update(group["ids"])
            offloaded_stream_layers.extend(group["stream_layers"])
            resident_bytes -= group["resident_bytes"]
            offloaded_blocks += 1

        ring_bytes = sum(
            sorted(offloaded_stream_layers, reverse=True)[:PIPELINE_DEPTH]
        )
        fits = resident_bytes + ring_bytes <= usable_bytes
        return {
            "offload_ids": offload_ids,
            "offloaded_layers": len(offloaded_stream_layers),
            "offloaded_blocks": offloaded_blocks,
            "total_blocks": len(groups),
            "resident_bytes": resident_bytes,
            "ring_bytes": ring_bytes,
            "model_bytes": total_model_bytes,
            "working_reserve_bytes": working_reserve_bytes,
            "wddm_margin_bytes": wddm_margin_bytes,
            "wddm_hard_bytes": wddm_hard_bytes,
            "usable_bytes": usable_bytes,
            "fits": fits,
        }

    @staticmethod
    def _interleave_priority(index, count):
        """Bit-reversal (van der Corput) rank of ``index`` among ``count`` slots.

        Sorting a list by this key (as a tie-break under an equal primary key)
        makes every prefix of the sorted order spread evenly across the original
        0..count-1 range, whatever the prefix length turns out to be. Used to
        pick which same-size residency/pin candidates a partial budget covers,
        so the unselected remainder does not cluster into one contiguous
        streaming/bounce-only run — see the depth-1 interleave in ``attach``.
        """
        if count <= 1:
            return 0.0
        bits = (count - 1).bit_length()
        rev = 0
        x = index
        for _ in range(bits):
            rev = (rev << 1) | (x & 1)
            x >>= 1
        return rev / float(1 << bits)

    _auto_wddm_margin_gib = staticmethod(vram_budget.auto_margin_gib)

    @classmethod
    def _resolve_wddm_margin_gib(
        cls,
        device,
        value,
        *,
        hard_gib=0.0,
        env_name="AI_TOOLKIT_TRAINING_WDDM_MARGIN_GIB",
    ):
        raw = _env(env_name, "-1.0") if value is None else value
        try:
            margin = float(raw)
            auto = margin < 0
        except (TypeError, ValueError):
            auto = str(raw).strip().lower() == "auto"
            margin = -1.0
        if auto:
            margin = cls._auto_wddm_margin_gib(device)
        return max(float(margin), float(hard_gib or 0.0))

    @classmethod
    def smart_training_plan(
        cls,
        module,
        device,
        working_reserve_gib=2.0,
        ignore_modules=None,
        must_resident_keys=("tproj.1",),
        resident_floor_gib=2.0,
        wddm_margin_gib=-1.0,
        wddm_hard_gib=None,
        prefetch_healthy=False,
        pinned_resident_keys=None,
        cold_growth=False,
        block_stream_only=False,
    ):
        """Choose training-resident layers with stream buffers before growth.

        Residency is chosen per-Linear, so a transformer block is routinely part
        streamed / part resident. That is fine for the compiled in-graph trunk:
        it packs only a block's streamed leaves (one coalesced fetch over a
        smaller flat) and reads the resident ones straight off their Parameters.
        See ``ingraph_stream.build_block_leaf_plans``."""
        ignore_modules = list(ignore_modules or [])
        device = torch.device(device)
        free_bytes, total_bytes = vram_budget.device_mem_info(device)
        # Measure the uncontrolled (not-ours) VRAM already on the card: CUDA
        # context, cuDNN/cublas workspaces, cudagraph constants, Windows/WDDM/
        # display, and other processes. This is the system_reserve bucket. The
        # driver-level ``free`` already excludes it, so it is recorded (not
        # re-subtracted) — surfacing it lets manual mode reason about true
        # available VRAM the same way the live controller does.
        try:
            device_used_bytes = torch.cuda.device_memory_used(device)
        except Exception:
            device_used_bytes = total_bytes - free_bytes
        system_reserve_bytes = max(
            0, device_used_bytes - torch.cuda.memory_reserved(device)
        )
        working_reserve_bytes = int(float(working_reserve_gib) * 1024 ** 3)
        wddm_hard_gib = (
            float(_env("AI_TOOLKIT_TRAINING_WDDM_HARD_GIB", "1.0"))
            if wddm_hard_gib is None
            else float(wddm_hard_gib)
        )
        wddm_margin_gib = cls._resolve_wddm_margin_gib(
            device,
            wddm_margin_gib,
            hard_gib=wddm_hard_gib,
        )
        wddm_margin_bytes = int(wddm_margin_gib * 1024 ** 3)
        wddm_hard_bytes = int(wddm_hard_gib * 1024 ** 3)
        usable_bytes = max(0, free_bytes - wddm_margin_bytes - working_reserve_bytes)
        total_model_bytes = cls._module_bytes(module)
        names = {id(child): name for name, child in module.named_modules()}
        candidates = []
        for child, resident, _ in cls._sampling_candidates(module, ignore_modules):
            key = names.get(id(child), child.__class__.__name__)
            candidates.append(
                {
                    "module": child,
                    "key": key,
                    "resident_bytes": resident,
                    "stream_bytes": cls._training_stream_bytes(child),
                }
            )

        candidate_resident_bytes = sum(item["resident_bytes"] for item in candidates)
        non_candidate_bytes = max(0, total_model_bytes - candidate_resident_bytes)
        must_tokens = tuple(must_resident_keys or ())
        pinned_keys = set(pinned_resident_keys or ())
        must_resident_layer_keys = {
            item["key"]
            for item in candidates
            if any(token and token in item["key"] for token in must_tokens)
        }
        # Block-only streaming: in ``block_stream_only`` mode every layer that is
        # NOT part of a repeated transformer block (a ModuleList of indexed
        # entries, e.g. ``blocks.0`` .. ``blocks.37``) is forced resident, so the
        # streaming ring only ever moves uniform block-sized groups. This trades
        # a little resident VRAM for far fewer scattered small transfers — one-off
        # layers (embedders, the final projection, standalone Sequential Linears)
        # otherwise each submit a tiny copy and flood the offload worker with
        # high-frequency requests.
        block_parents = cls._streaming_block_parents(
            cls._offload_group_key(item["key"]) for item in candidates
        )

        def _is_streaming_block(group_key):
            parent = cls._block_parent_of(group_key)
            return parent is not None and parent in block_parents

        resident = []
        offloaded = []
        for item in candidates:
            group_key = cls._offload_group_key(item["key"])
            item["group_key"] = group_key
            item["is_streaming_block"] = _is_streaming_block(group_key)
            item["pinned_resident"] = group_key in pinned_keys
            item["block_stream_resident"] = bool(
                block_stream_only and not _is_streaming_block(group_key)
            )
            if (
                item["pinned_resident"]
                or item["block_stream_resident"]
                or item["key"] in must_resident_layer_keys
            ):
                resident.append(item)
            else:
                offloaded.append(item)

        # Repeated transformer blocks make most offloaded candidates byte-identical,
        # so the size-sort below ties constantly; break ties by execution-order
        # spread instead of leaving them at stable-sort (= list/definition) order,
        # which would otherwise cluster every fill at one end of the block stack and
        # leave the other end streaming with no resident/pinned neighbor nearby.
        offload_priority = {
            id(item["module"]): cls._interleave_priority(i, len(offloaded))
            for i, item in enumerate(offloaded)
        }

        pinned_resident_bytes = sum(
            item["resident_bytes"] for item in resident
            if item.get("pinned_resident")
        )
        must_resident_bytes = non_candidate_bytes + sum(
            item["resident_bytes"] for item in resident
        )
        resident_floor_bytes = int(float(resident_floor_gib) * 1024 ** 3)
        resident_bytes = must_resident_bytes
        remaining = max(0, usable_bytes - resident_bytes)

        if resident_bytes < resident_floor_bytes and remaining > 0:
            for item in sorted(
                offloaded,
                key=lambda row: (
                    row["resident_bytes"],
                    offload_priority[id(row["module"])],
                ),
            ):
                need = item["resident_bytes"]
                if resident_bytes >= resident_floor_bytes or need > remaining:
                    continue
                resident.append(item)
                offloaded.remove(item)
                resident_bytes += need
                must_resident_bytes += need
                remaining -= need

        stream_need_bytes = sum(
            sorted(
                (item["stream_bytes"] for item in offloaded), reverse=True
            )[:PIPELINE_DEPTH]
        )
        gpu_stream_budget_bytes = min(remaining, stream_need_bytes)
        remaining -= gpu_stream_budget_bytes

        generic_resident_bytes = 0
        # Grow resident up from the floor into the surplus. ``remaining`` is
        # already net of the ring, working_reserve, and target-free reserves, so the
        # growth never eats the activation/backward budget — keeping a block
        # resident only swaps a stream for the same demand-load path it already
        # used, so it is always safe.
        #
        # The live AIMD loop (auto mode) is gated on ``prefetch_healthy`` so it
        # does not pile on resident pressure step-to-step before the current
        # layout proves stable. But that signal can only be known AFTER a step
        # has streamed, so at attach time it is structurally False and the cold
        # plan would otherwise never leave the floor. ``cold_growth`` lets the
        # initial plan fill the surplus once, up front: manual working_reserve means
        # "use the number I picked," so we grow to the byte budget immediately;
        # auto working_reserve leaves the final climb to the live loop.
        resident_growth_allowed = bool((prefetch_healthy or cold_growth) and remaining > 0)
        blocked_reason = None
        if resident_growth_allowed:
            for item in sorted(
                list(offloaded),
                key=lambda row: (
                    row["resident_bytes"],
                    -offload_priority[id(row["module"])],
                ),
                reverse=True,
            ):
                need = item["resident_bytes"]
                if need <= remaining:
                    resident.append(item)
                    offloaded.remove(item)
                    resident_bytes += need
                    generic_resident_bytes += need
                    remaining -= need
        elif remaining > 0:
            blocked_reason = "prefetch_unhealthy"
        else:
            blocked_reason = "no_surplus"

        fits = resident_bytes + stream_need_bytes <= usable_bytes
        compile_readiness = None
        if hasattr(module, "training_compile_readiness"):
            try:
                compile_readiness = module.training_compile_readiness(pinned_keys)
            except Exception:
                compile_readiness = None
        return {
            "offload_ids": {id(item["module"]) for item in offloaded},
            "offloaded_layers": len(offloaded),
            "candidate_layers": len(candidates),
            "model_bytes": total_model_bytes,
            "resident_bytes": resident_bytes,
            "must_resident_bytes": must_resident_bytes,
            "must_resident_layer_keys": set(must_resident_layer_keys),
            "pinned_resident_bytes": pinned_resident_bytes,
            "pinned_resident_keys": set(pinned_keys),
            "block_stream_only": bool(block_stream_only),
            "block_stream_resident_bytes": sum(
                item["resident_bytes"]
                for item in resident
                if item.get("block_stream_resident")
            ),
            "training_compile_readiness": compile_readiness,
            "generic_resident_bytes": generic_resident_bytes,
            "ring_bytes": gpu_stream_budget_bytes,
            "gpu_stream_need_bytes": stream_need_bytes,
            "gpu_stream_budget_bytes": gpu_stream_budget_bytes,
            "working_reserve_bytes": working_reserve_bytes,
            "wddm_margin_bytes": wddm_margin_bytes,
            "wddm_hard_bytes": wddm_hard_bytes,
            "system_reserve_bytes": system_reserve_bytes,
            "usable_bytes": usable_bytes,
            "free_bytes": free_bytes,
            "resident_growth_allowed": resident_growth_allowed,
            "resident_growth_blocked_reason": blocked_reason,
            "fits": fits,
        }

    @staticmethod
    def _training_working_reserve_decision(
        current_gib,
        measured_peak_gib,
        min_device_free_gib,
        danger_gib,
        *,
        wddm_hard_gib=1.0,
        wddm_stop_gib=1.5,
        pad_gib=0.5,
        step_gib=0.5,
        retreat_gib=1.0,
    ):
        """Decide the next training working_reserve reservation.

        The working reserve is a measurement, not a safety controller. It should
        track the observed activation/dequant/temporary peak plus a small pad.
        WDDM free-space cliffs, OOMs, and external pressure are handled by layout
        demotion/promotion; feeding them back into this number made auto reserve
        worse than a fixed manual guess by inventing panic reserve.
        """
        del min_device_free_gib, danger_gib, wddm_hard_gib, wddm_stop_gib, step_gib, retreat_gib
        target_gib = max(0.0, float(measured_peak_gib)) + max(0.0, float(pad_gib))
        current_gib = max(0.0, float(current_gib))
        if abs(current_gib - target_gib) <= 1e-6:
            return target_gib, None, "hold"
        if current_gib < target_gib:
            return target_gib, None, "grow"
        return target_gib, None, "shrink"

    @staticmethod
    def _prefetch_trace_invalid(
        *,
        schedule_len: int,
        consume_pos: int,
        lookahead: int,
        hard_miss_rate: float,
        mismatch_rate: float,
        duplicate_key_block_rate: float,
        hard_miss_threshold: float = 0.25,
        mismatch_threshold: float = 0.10,
    ) -> bool:
        if schedule_len <= 0:
            return False
        if consume_pos > schedule_len + max(1, lookahead) and hard_miss_rate > hard_miss_threshold:
            return True
        if mismatch_rate > mismatch_threshold:
            return True
        if duplicate_key_block_rate > mismatch_threshold:
            return True
        return False

    @staticmethod
    def _prefetch_recovery_action(*, prefetch_missing: bool, prefetch_invalid: bool):
        if prefetch_invalid:
            return "reset_prefetch_trace"
        if prefetch_missing:
            return "seed_prefetch_schedule"
        return None

    @staticmethod
    def _prefetch_allows_resident_growth(
        *,
        pool_present: bool,
        schedule_confidence: str,
        prefetch_healthy: bool,
    ) -> bool:
        """Whether prefetch evidence is strong enough to grow residency.

        Compatible traces are useful hints for hiding transfers, but they are not
        proof that the current execution/memory profile is stable enough to add
        resident pressure.
        """
        if not pool_present:
            return True
        if schedule_confidence not in ("exact", "observed"):
            return False
        return bool(prefetch_healthy)

    @staticmethod
    def _training_working_reserve_signal(
        measured_peak_gib,
        working_ema_gib,
        *,
        steps,
        stable_windows,
        min_working_reserve_gib,
        pad_gib,
    ):
        """Per-step target signal for the working_reserve decision (pure, CPU-testable).

        ``measured_peak_gib`` MUST be the truthful within-step activation peak
        (``max_memory_allocated - resident - ring``), not the step-end residual.
        Feeding the residual here was the bug that made auto-working_reserve "not
        work": the residual reads the trough (~1 GiB, after the backward graph
        frees) while the real peak is several GiB, so the controller shrank the
        budget below the activation footprint and spilled.

        Once a bucket is warmed up (``steps >= stable_windows``) we drive off the
        smoothed peak so single-step jitter does not move the reserve; before that,
        off the raw peak. Floored at ``min_working_reserve - pad`` so a quiet
        bucket cannot starve the reserve below the configured minimum.
        """
        if steps >= stable_windows and working_ema_gib is not None:
            return max(
                working_ema_gib, measured_peak_gib, min_working_reserve_gib - pad_gib
            )
        return measured_peak_gib

    @staticmethod
    def _training_timing_spill_floor(
        step_time_s,
        best_step_time_s,
        min_device_free_gib,
        *,
        steps,
        warmup_steps=4,
        slowdown_ratio=3.0,
        max_signal_free_gib=2.0,
        pad_gib=0.25,
    ):
        """Infer a learned WDDM hard floor from a catastrophic timing cliff.

        WDDM spill is often silent: no OOM, just a sudden multi-x slowdown once
        the committed footprint crosses the driver cliff. Treat timing as WDDM
        evidence only after a bucket has a stable baseline and the measured
        peak-free signal is already near the configured safety band.
        """
        if step_time_s is None or best_step_time_s is None:
            return None
        try:
            step_time_s = float(step_time_s)
            best_step_time_s = float(best_step_time_s)
            min_device_free_gib = float(min_device_free_gib)
        except (TypeError, ValueError):
            return None
        if steps < int(warmup_steps) or best_step_time_s <= 0.0:
            return None
        if step_time_s < best_step_time_s * float(slowdown_ratio):
            return None
        if min_device_free_gib > float(max_signal_free_gib):
            return None
        return max(0.0, min_device_free_gib + float(pad_gib))

    @staticmethod
    def _training_layout_action(
        governing_free_gib,
        *,
        wddm_hard_gib=1.0,
        wddm_hold_high_gib=2.0,
        did_oom=False,
    ):
        """Free-VRAM deadband: which way to move the resident set (pure, CPU-testable).

        The user-specified policy: keep free VRAM inside a hold band so the
        controller stops moving layers — and therefore stops resetting the
        prefetch trace — once it is in range. That hold is what lets auto-working_reserve
        converge to the same steady state as a hand-picked smart_working_reserve value.

            free < wddm_hard_gib        -> "down"  (give VRAM back)
            free > wddm_hold_high_gib   -> "up"    (use more VRAM)
            hard <= free <= high   -> "hold"  (no move, no trace reset)

        ``governing_free_gib`` is the worst-case (minimum) recent free margin
        across resolution buckets, so a generous low-res step never authorizes a
        layout that spills at high-res.
        """
        if did_oom or governing_free_gib < wddm_hard_gib:
            return "down"
        if governing_free_gib > wddm_hold_high_gib:
            return "up"
        return "hold"

    @staticmethod
    def _gc_counter_deltas(prev, current):
        """Per-window deltas of allocator GC counters (pure, CPU-testable).

        ``num_alloc_retries`` ticks once per OOM-retry reclaim (the
        gc_threshold proactive sweep does not tick it); ``num_device_free`` /
        ``num_device_alloc`` count cudaFree/cudaMalloc calls and expose GC
        churn. Counters are monotonic per process; a smaller current value
        means the stats were reset externally, in which case the current
        value is the best available delta.
        """
        deltas = {}
        for key in ("num_alloc_retries", "num_device_alloc", "num_device_free"):
            cur = int((current or {}).get(key, 0) or 0)
            prv = int((prev or {}).get(key, 0) or 0)
            deltas[key] = cur - prv if cur >= prv else cur
        return deltas

    @classmethod
    def _training_layout_move(
        cls,
        demote_governing_free_gib,
        current_free_gib,
        *,
        wddm_hard_gib,
        wddm_hold_high_gib,
        did_oom,
    ):
        """Demote and promote deliberately watch different signals.

        Demote stays conservative across every resolution bucket seen so far
        (``demote_governing_free_gib``, the worst-case recent free margin): a
        generous low-res step must never license a layout that later spills at
        high-res. Promote is gated on the CURRENT bucket's own headroom only
        (``current_free_gib``) -- requiring every OTHER bucket to also be
        comfortable meant a chronically tight high-res bucket (activation-bound,
        not fixable by shedding resident weight bytes) permanently vetoed
        promotion even during a roomy low-res step, so resident VRAM only ever
        ratcheted down and never recovered. A promotion that turns out unsafe at
        another resolution is caught and reversed by the demote check above the
        very next time that resolution runs.
        """
        demote_move = cls._training_layout_action(
            demote_governing_free_gib,
            wddm_hard_gib=wddm_hard_gib,
            wddm_hold_high_gib=wddm_hold_high_gib,
            did_oom=did_oom,
        )
        if demote_move == "down":
            return "down"
        promote_move = cls._training_layout_action(
            current_free_gib,
            wddm_hard_gib=wddm_hard_gib,
            wddm_hold_high_gib=wddm_hold_high_gib,
            did_oom=False,
        )
        return "up" if promote_move == "up" else "hold"

    @staticmethod
    def _shared_cliff_relief_decision(
        shared_raw_headroom_gib,
        shared_margin_gib,
        dedicated_free_gib,
        dedicated_promote_free_gib,
    ):
        """Choose the shared-budget pressure response.

        ``shared_raw_headroom_gib`` is NON_LOCAL Budget - CurrentUsage, before
        subtracting the reserved spill margin. When it falls below the margin,
        prefer unpinning a streamed layer back to pageable CPU. Promotion is only
        allowed when dedicated VRAM is already roomy enough that spending more
        resident memory will not deepen the dedicated/WDDM spill cliff.
        """
        if shared_raw_headroom_gib is None or shared_margin_gib is None:
            return "hold"
        try:
            shared_raw = float(shared_raw_headroom_gib)
            margin = float(shared_margin_gib)
        except (TypeError, ValueError):
            return "hold"
        if shared_raw >= margin:
            return "hold"
        try:
            dedicated_free = float(dedicated_free_gib)
            promote_free = float(dedicated_promote_free_gib)
        except (TypeError, ValueError):
            return "unpin"
        return "promote" if dedicated_free >= promote_free else "unpin"

    @staticmethod
    def _dxgi_shared_budget_snapshot(device):
        raw = MemoryManager._dxgi_shared_budget_snapshot_bytes(device)
        if raw is None:
            return None
        return {
            "budget_gib": raw["budget_bytes"] / 1024 ** 3,
            "usage_gib": raw["usage_bytes"] / 1024 ** 3,
            "raw_headroom_gib": raw["raw_headroom_bytes"] / 1024 ** 3,
            "margin_gib": raw["margin_bytes"] / 1024 ** 3,
            "usable_headroom_gib": raw["usable_headroom_bytes"] / 1024 ** 3,
            "match_method": raw.get("match_method"),
            "manual_control": bool(raw.get("manual_control", False)),
        }

    @staticmethod
    def _dxgi_shared_budget_snapshot_bytes(device):
        dxgi = bounce_pool.get_dxgi_meminfo()
        if dxgi is None:
            return None
        adapter = dxgi.selected_adapter_info()
        if adapter is None or not getattr(adapter, "safe_for_control", False):
            return None
        info = dxgi.query_non_local_video_memory_info(
            cuda_device_index=bounce_pool._cuda_device_index(device),
            min_interval_s=0.0,
        )
        if info is None:
            return None
        budget_bytes = int(info.budget_bytes)
        usage_bytes = int(info.current_usage_bytes)
        margin_bytes = bounce_pool.dxgi_spill_reserve_bytes(budget_bytes)
        raw_headroom_bytes = max(0, budget_bytes - usage_bytes)
        return {
            "budget_bytes": budget_bytes,
            "usage_bytes": usage_bytes,
            "raw_headroom_bytes": raw_headroom_bytes,
            "margin_bytes": margin_bytes,
            "usable_headroom_bytes": raw_headroom_bytes - margin_bytes,
            "match_method": getattr(adapter, "match_method", None),
            "manual_control": bool(getattr(adapter, "manual_control", False)),
        }

    @staticmethod
    def _dxgi_local_budget_snapshot_bytes(device):
        dxgi = bounce_pool.get_dxgi_meminfo()
        if dxgi is None:
            return None
        adapter = dxgi.selected_adapter_info()
        if adapter is None or not getattr(adapter, "safe_for_control", False):
            return None
        info = dxgi.query_local_video_memory_info(
            cuda_device_index=bounce_pool._cuda_device_index(device),
            min_interval_s=0.0,
        )
        if info is None:
            return None
        budget_bytes = int(info.budget_bytes)
        usage_bytes = int(info.current_usage_bytes)
        return {
            "budget_bytes": budget_bytes,
            "usage_bytes": usage_bytes,
            "raw_headroom_bytes": max(0, budget_bytes - usage_bytes),
            "match_method": getattr(adapter, "match_method", None),
            "manual_control": bool(getattr(adapter, "manual_control", False)),
        }

    @staticmethod
    def _predict_dxgi_local_peak_bytes(
        snapshot,
        *,
        current_reserved_bytes,
        current_allocated_bytes,
        peak_reserved_bytes,
        peak_allocated_bytes,
    ):
        """Predict required LOCAL usage from live allocation demand.

        Historical allocator reservation is cache appetite, not required live
        memory. Keep the currently observed non-allocator footprint and add
        the learned live allocation peak. peak_reserved_bytes remains in the
        signature for compatibility with callers and diagnostics only.
        """
        if snapshot is None:
            return None
        usage = int(snapshot.get("usage_bytes", 0) or 0)
        current_reserved = max(0, int(current_reserved_bytes or 0))
        current_allocated = max(0, int(current_allocated_bytes or 0))
        peak_allocated = max(current_allocated, int(peak_allocated_bytes or 0))
        non_torch = max(0, usage - current_reserved)
        return non_torch + peak_allocated

    @staticmethod
    def _dxgi_sampling_settle_status(snapshot):
        if snapshot is None:
            return "unavailable"
        return "settled" if snapshot.get("raw_headroom_bytes", 0) >= snapshot.get("margin_bytes", 0) else "pressure"

    @classmethod
    def _wait_for_sampling_dxgi_settle(cls, device, *, timeout_s=None, poll_s=None):
        timeout_s = float(
            _env("AI_TOOLKIT_SAMPLING_DXGI_SETTLE_TIMEOUT_S", "2.0")
            if timeout_s is None
            else timeout_s
        )
        poll_s = float(
            _env("AI_TOOLKIT_SAMPLING_DXGI_SETTLE_POLL_S", "0.05")
            if poll_s is None
            else poll_s
        )
        deadline = time.monotonic() + max(0.0, timeout_s)
        attempts = 0
        last = None
        while True:
            attempts += 1
            last = cls._dxgi_shared_budget_snapshot_bytes(device)
            status = cls._dxgi_sampling_settle_status(last)
            if status in ("settled", "unavailable"):
                return {"status": status, "attempts": attempts, "snapshot": last}
            if time.monotonic() >= deadline:
                return {"status": "timeout", "attempts": attempts, "snapshot": last}
            time.sleep(max(0.0, poll_s))

    @staticmethod
    def _available_vram_gib(
        total_gib,
        device_used_gib,
        torch_reserved_gib,
        peak_reserved_gib,
        *,
        safety_gib=0.5,
    ):
        """Reserved working_reserve we can still grow into before spilling (pure, CPU-testable).

        Accounts for residents that are not in our caching allocator:

            other = (total - free) - reserved == device_used - reserved

        which captures both other CUDA processes and our own non-allocator
        overhead (CUDA context, cuDNN workspaces, compiled-graph constants). Then:

            max_reserved_we_can_hold = total - safety - other
            available                = max_reserved_we_can_hold - peak_reserved

        ``available`` is what the deadband governs: grow resident while
        ``peak_reserved + Δ < max_reserved_we_can_hold``. Shared by the live
        controller and the offline memory simulator so they cannot drift.
        """
        other = max(0.0, device_used_gib - torch_reserved_gib)
        max_reserved_we_can_hold = max(0.0, total_gib - safety_gib - other)
        return max(0.0, max_reserved_we_can_hold - peak_reserved_gib)

    @staticmethod
    def _training_governing_free_gib(estimated_free_gib, diagnostics):
        """Use observed driver-free minima when present; otherwise keep estimate."""
        result = float(estimated_free_gib)
        if (diagnostics or {}).get("device_peak_source") != "observed":
            return result
        try:
            observed = max(0.0, float(diagnostics.get("device_free_peak_gb", result)))
        except (TypeError, ValueError):
            return result
        return min(result, observed)

    @staticmethod
    def _training_cliff_guard_action(
        device_free_gib, *, wddm_hard_gib=1.0, did_oom=False
    ):
        """Manual-mode cliff guard trigger (pure, CPU-testable).

        Manual working_reserve does not auto-tune the budget, but it must still
        refuse to sit on the WDDM spill cliff. This is the single predicate the
        live safety net escalates on and the offline simulator reuses, so the two
        cannot drift:

            free < wddm_hard  (or an OOM was seen)  -> "reclaim"
            otherwise                               -> "ok"

        "reclaim" means: first return the allocator's idle cache to the driver,
        then — only if free is still under the floor — demote resident layers.
        """
        if did_oom or device_free_gib < wddm_hard_gib:
            return "reclaim"
        return "ok"

    @staticmethod
    def _training_shape_state_key(shape_key):
        if shape_key is None:
            return ("default",)
        try:
            hash(shape_key)
            return shape_key
        except TypeError:
            return repr(shape_key)

    @classmethod
    def _manual_training_safety_state(cls, mm):
        state = getattr(mm, "_training_manual_safety_state", None)
        if not isinstance(state, dict):
            state = {}
            mm._training_manual_safety_state = state
        state.setdefault("shape_peaks", {})
        return state

    @classmethod
    def _record_manual_training_shape_peak(
        cls, mm, shape_key, *, peak_allocated_gib, peak_reserved_gib=0.0
    ):
        state = cls._manual_training_safety_state(mm)
        peaks = state.setdefault("shape_peaks", {})
        key = cls._training_shape_state_key(shape_key)
        bucket = peaks.setdefault(key, {"steps": 0, "warmup_steps": 0})
        if int(bucket.get("warmup_steps", 0)) < 1:
            # The first execution of a shape, and the first execution after a
            # residency/layout change, includes compile/retrace and pipeline
            # warmup. It is not a steady-state safety requirement.
            bucket["warmup_steps"] = int(bucket.get("warmup_steps", 0)) + 1
            return None
        bucket["steps"] = int(bucket.get("steps", 0)) + 1
        bucket["peak_allocated_gib"] = max(
            float(bucket.get("peak_allocated_gib", 0.0)), float(peak_allocated_gib or 0.0)
        )
        bucket["peak_reserved_gib"] = max(
            float(bucket.get("peak_reserved_gib", 0.0)), float(peak_reserved_gib or 0.0)
        )
        return bucket

    @classmethod
    def _invalidate_manual_training_shape_peaks(cls, mm):
        """Discard peaks learned for an obsolete residency/layout."""
        state = cls._manual_training_safety_state(mm)
        state["shape_peaks"] = {}

    @classmethod
    def _manual_training_shape_peak(cls, mm, shape_key):
        state = cls._manual_training_safety_state(mm)
        bucket = state.get("shape_peaks", {}).get(cls._training_shape_state_key(shape_key))
        if not bucket:
            return None
        peak = float(bucket.get("peak_allocated_gib", 0.0))
        return peak if peak > 0.0 else None

    @classmethod
    def _training_shape_peak_bucket(cls, mm, shape_key):
        state = cls._manual_training_safety_state(mm)
        return state.get("shape_peaks", {}).get(cls._training_shape_state_key(shape_key))

    _training_cliff_predicted_peak_free_gib = staticmethod(
        vram_budget.training_cliff_predicted_peak_free_gib
    )
    _training_guard_pressure = staticmethod(vram_budget.training_guard_pressure)

    @classmethod
    def promote_layer(cls, child):
        """Make one streamed layer resident, in place (offloaded -> resident).

        Tensor-subclass weights cannot be promoted by assigning CUDA storage into
        the existing CPU Parameter. Replace those Parameters, keep TorchAO shadow
        attributes aligned, and only restore the native forward after every move
        succeeds. On failure the layer remains streamed.
        """
        lmm = getattr(child, "_layer_memory_manager", None)
        if lmm is None:
            return False
        device = lmm.manager.process_device
        original_params = dict(child._parameters)
        original_data = {
            name: param.data for name, param in child._parameters.items()
            if param is not None
        }
        original_managed = {
            name: hasattr(param, "_is_memory_managed")
            for name, param in child._parameters.items()
            if param is not None
        }
        try:
            for name, param in list(child._parameters.items()):
                if param is None:
                    continue
                if _is_quantized_tensor(param.data):
                    moved = cls._move_tensor_subclass(param.data, device)
                    replacement = torch.nn.Parameter(
                        moved, requires_grad=param.requires_grad
                    )
                    child._parameters[name] = replacement
                    # TorchAO may shadow Module._parameters with a direct
                    # instance attribute; Linear.forward reads that attribute.
                    if name in child.__dict__:
                        object.__setattr__(child, name, replacement)
                    param = replacement
                else:
                    param.data = param.data.to(device)
                if hasattr(param, "_is_memory_managed"):
                    del param._is_memory_managed
        except Exception:
            for name, param in original_params.items():
                if param is not None and name in original_data:
                    try:
                        param.data = original_data[name]
                    except Exception:
                        pass
                    if original_managed.get(name, False):
                        param._is_memory_managed = True
                child._parameters[name] = param
                if param is not None and name in child.__dict__:
                    object.__setattr__(child, name, param)
            raise

        # Same chain-aware unwind as detach: the recorded slot may hold a LoRA
        # hijack applied after attach; only our own forward gets removed.
        lmm._uninstall_base_forward()
        if hasattr(child, "_memory_management_device"):
            del child._memory_management_device
        # Return this layer's pinned budget: its CPU copy is now on GPU, so the
        # pinned bytes it held are free for other (e.g. later-demoted) layers.
        freed = getattr(child, "_mm_pinned_bytes", 0)
        if freed:
            lmm.manager.pinned_weight_bytes = max(
                0, lmm.manager.pinned_weight_bytes - freed
            )
            bounce_pool.release_pinned_bytes(freed, kind="weights")
            child._mm_pinned_bytes = 0
        del child._layer_memory_manager
        cls._refresh_resident_trace_hooks(lmm.manager.module, lmm.manager)
        return True

    @classmethod
    def demote_layer(cls, child, manager, layer_key=None):
        if child is None or hasattr(child, "_layer_memory_manager"):
            return False

        if (
            getattr(manager, "_attach_args", {}).get("training_strategy")
            == "smart_immutable"
            and any(
                parameter.requires_grad
                for parameter in child.parameters(recurse=False)
            )
        ):
            raise RuntimeError(
                "immutable training attempted to demote a trainable module: "
                f"{layer_key or child.__class__.__name__}"
            )

        name = child.__class__.__name__
        if name in LINEAR_MODULES:
            LinearLayerMemoryManager.attach(child, manager)
        elif name in CONV_MODULES:
            ConvLayerMemoryManager.attach(child, manager)
        else:
            return False
        child._mm_layer_key = (
            layer_key
            or getattr(child, "_mm_layer_key", None)
            or name
        )
        cls._refresh_resident_trace_hooks(manager.module, manager)
        return True

    @staticmethod
    def _training_auto_seed_working_reserve_gib():
        """Cold-start reserve-space assumption before the first measured step."""
        return float(_env("AI_TOOLKIT_TRAINING_AUTO_SEED_WORKING_RESERVE_GIB", "5.0"))

    _wddm_hard_cap_applied: dict = {}
    # Cap violations are not fatal by default. The cap exists as a *tuning
    # lever* (it recycles idle cache on demand and is the residency
    # controller's cheap inner lever), not as a kill switch: a training run
    # that overshoots it should give the memory back and keep going, degrading
    # toward WDDM paging, rather than die. Each violation permanently widens
    # the cap for this device by WDDM_CAP_RELIEF_BYTES, so the same overshoot
    # cannot cost a second step. Strict mode (validation, simulated cards,
    # attributing an OOM to its true culprit) turns the violation back into a
    # real OutOfMemoryError.
    _wddm_cap_strict: bool = False
    WDDM_CAP_RELIEF_BYTES = int(0.5 * 1024 ** 3)

    # Pure dedicated-cliff arithmetic lives in vram_budget; these wrappers keep
    # the historical MemoryManager.* names for tests and extensions.
    _wddm_cap_fraction = staticmethod(vram_budget.cap_fraction)

    @classmethod
    def set_wddm_cap_strict(cls, strict: bool) -> None:
        """Strict = a cap violation raises. Default (False) = relax and survive."""
        cls._wddm_cap_strict = bool(strict)

    @classmethod
    def relieve_wddm_cap_after_oom(cls, device, *, context="training step") -> bool:
        """Widen the cap after it was violated. True when the run may continue.

        Returns False when nothing can be relieved -- strict mode, no cap
        applied, or the cap is already at the whole card, which means the OOM
        was physical and not ours to forgive.
        """
        if cls._wddm_cap_strict:
            return False
        if sys.platform != "win32" or not torch.cuda.is_available():
            return False
        dev = torch.device(device if device is not None else "cuda")
        if dev.type != "cuda":
            return False
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        applied = cls._wddm_hard_cap_applied.get(index)
        if applied is None or applied >= 1.0:
            return False

        real_total = vram_budget.real_device_total_bytes(index)
        relief = _WDDM_CAP_RELIEF_BYTES.get(index, 0) + cls.WDDM_CAP_RELIEF_BYTES
        _WDDM_CAP_RELIEF_BYTES[index] = relief
        widened = min(1.0, applied + cls.WDDM_CAP_RELIEF_BYTES / float(real_total))
        torch.cuda.set_per_process_memory_fraction(widened, index)
        cls._wddm_hard_cap_applied[index] = widened
        gib = 1024 ** 3
        print(
            f"[MemoryManager] WDDM allocator cap violated during {context}: "
            f"widening {applied * real_total / gib:.2f} -> "
            f"{widened * real_total / gib:.2f} GiB "
            f"(total relief {relief / gib:.2f} GiB above the cliff bound). "
            "The run continues; past the dedicated ceiling WDDM pages to system "
            "RAM, which is slow but not fatal. Set "
            "layer_offloading_wddm_cap_strict to make this an OOM instead."
        )
        return True

    @classmethod
    def _apply_wddm_hard_allocator_cap(
        cls, device, wddm_hard_gib=None, *, target_cap_bytes=None
    ):
        """Hard-cap torch's allocator below the WDDM dedicated ceiling.

        Crossing the dedicated-VRAM ceiling on Windows does not OOM -- WDDM
        silently pages GPU memory to system RAM (catastrophic slowdown, no
        error; we have observed torch_allocated=12.23 GiB on an 11.99 GiB
        card). set_per_process_memory_fraction makes the caching allocator
        recycle its cache and, failing that, raise a real OOM at the cap
        instead, so the failure is loud, attributable, and never a silent
        30x slowdown. The cap accounts for non-torch device usage (see
        _wddm_cap_fraction); it is re-measured at every call, and each call
        site is a phase boundary (training attach / sampling start), so the
        latest measurement wins.

        ``target_cap_bytes`` optionally requests a *reclaim* cap below the cliff
        bound: the caller has planned its live footprint and wants the cap to sit
        just above it (banking the unused dedicated VRAM as the DXGI overflow
        valve + tier-1 climb headroom) rather than at the ceiling. It is clamped
        to the cliff bound above, so the worst case is exactly the default
        behavior; the reclaim only ever tightens, never loosens past the ceiling.
        """
        if sys.platform != "win32" or not torch.cuda.is_available():
            return
        dev = torch.device(device if device is not None else "cuda")
        if dev.type != "cuda":
            return
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        try:
            hard_gib = float(wddm_hard_gib) if wddm_hard_gib is not None else 1.0
        except (TypeError, ValueError):
            hard_gib = 1.0
        if hard_gib <= 0:
            hard_gib = 1.0
        # `total`/`free` are the *governing* card, which a simulated smaller card
        # shrinks (vram_budget.set_simulated_card_bytes). torch enforces the
        # fraction against the physical card, so the cap is planned in governing
        # bytes and converted back to a real-card fraction at the last moment.
        total = vram_budget.device_total_bytes(index)
        real_total = vram_budget.real_device_total_bytes(index)
        free_bytes, _governing_total = vram_budget.device_mem_info(index)
        reserved_bytes = torch.cuda.memory_reserved(index)
        cliff_fraction = cls._wddm_cap_fraction(total, free_bytes, reserved_bytes, hard_gib)
        fraction = cliff_fraction
        reclaimed = False
        if target_cap_bytes is not None:
            target_fraction = float(target_cap_bytes) / float(total)
            # Never loosen past the cliff, never collapse below a sane floor.
            fraction = max(0.1, min(cliff_fraction, target_fraction))
            reclaimed = fraction < cliff_fraction - 1e-9
        # Relief granted after a past violation survives phase boundaries: a cap
        # that snapped back to the cliff bound every phase would re-crash on the
        # very footprint it just forgave.
        relief_bytes = _WDDM_CAP_RELIEF_BYTES.get(index, 0)
        if relief_bytes:
            fraction = min(1.0, fraction + relief_bytes / float(total))
        applied = fraction * total / float(real_total)
        previous = cls._wddm_hard_cap_applied.get(index)
        # 64 MiB tolerance: non_torch jitters a little every step (the per-step
        # trim re-measures); only real shifts (phase changes, kernel-code
        # growth) are worth a reset and a log line.
        if previous is not None and abs(previous - applied) < (64 * 1024 ** 2) / real_total:
            return
        torch.cuda.set_per_process_memory_fraction(applied, index)
        cls._wddm_hard_cap_applied[index] = applied
        gib = 1024 ** 3
        non_torch = max(0, (total - free_bytes) - reserved_bytes)
        source = (
            f"reclaim target, cliff {cliff_fraction * total / gib:.2f} GiB"
            if reclaimed
            else "cliff bound"
        )
        if relief_bytes:
            source += f"; +{relief_bytes / gib:.2f} GiB post-violation relief"
        if total != real_total:
            source += f"; SIMULATED {total / gib:.2f} GiB card"
        print(
            "[MemoryManager] WDDM hard allocator cap: "
            f"{fraction * total / gib:.2f}/{total / gib:.2f} GiB "
            f"({source}; margin {hard_gib:.2f} GiB, non_torch {non_torch / gib:.2f} GiB; "
            "allocation beyond this recycles cache or raises OOM "
            "instead of silently paging)"
        )

    @classmethod
    def attach_smart_training(
        cls, module, device, working_reserve_gib=2.0, ignore_modules=None,
        wddm_margin_gib=None,
        wddm_hard_gib=None,
        fp8_training_forward=False,
        pinned_resident_keys=None,
        block_stream_only=False,
        pinned_weight_gib=None,
        wddm_spill_reserve_pct=None,
        use_pinned_arena=False,
    ):
        if getattr(module, "_mm_immutable_backend", False):
            mm = getattr(module, "_memory_manager", None)
            if mm is None or getattr(mm, "_smart_training_plan", None) is None:
                raise RuntimeError("immutable backend is missing its smart plan")
            return mm._smart_training_plan
        cls._apply_wddm_hard_allocator_cap(device, wddm_hard_gib)
        ignore_modules = list(ignore_modules or [])
        pinned_resident_keys = set(pinned_resident_keys or ())
        auto_working_reserve = False
        try:
            auto_working_reserve = float(working_reserve_gib) < 0
        except (TypeError, ValueError):
            auto_working_reserve = str(working_reserve_gib).lower() == "auto"
        if auto_working_reserve:
            # The working set is not controllable; before the first measurement,
            # reserve enough resident-weight space for the observed Krea/WDDM peak
            # instead of optimistically starting near 2-3 GiB and spilling step 1.
            working_reserve_gib = cls._training_auto_seed_working_reserve_gib()
        resolved_wddm_hard_gib = (
            float(_env("AI_TOOLKIT_TRAINING_WDDM_HARD_GIB", "1.0"))
            if wddm_hard_gib is None
            else float(wddm_hard_gib)
        )
        resolved_wddm_margin_gib = cls._resolve_wddm_margin_gib(
            device,
            wddm_margin_gib,
            hard_gib=resolved_wddm_hard_gib,
        )
        # Deliberate borrow across the two cliffs: the SHARED-budget (DXGI
        # NON_LOCAL, pinned-memory) spill-reserve floor is seeded from the
        # dedicated-cliff planning margin so one config knob scales both
        # conservatively. They are otherwise unrelated quantities -- see
        # vram_budget's module docstring for the two-cliff model.
        bounce_pool.set_spill_reserve_policy(
            floor_gib=resolved_wddm_margin_gib,
            pct=wddm_spill_reserve_pct,
        )
        plan = cls.smart_training_plan(
            module, device, working_reserve_gib, ignore_modules,
            wddm_margin_gib=resolved_wddm_margin_gib,
            wddm_hard_gib=resolved_wddm_hard_gib,
            pinned_resident_keys=pinned_resident_keys,
            # Manual working_reserve has no live loop to climb later, so fill the
            # surplus at attach. Auto working_reserve leaves the climb to the live loop.
            cold_growth=not auto_working_reserve,
            block_stream_only=block_stream_only,
        )
        # Pin budget: size to the layers ACTUALLY selected for streaming (auto
        # mode), or honor a positive config value as a requested budget, then
        # cap it after reserving the planned bounce-pool window. This keeps
        # explicit pin budgets from consuming the pool's share of the same
        # WDDM pinned-memory ceiling. Uses the same scoped helper `attach`
        # itself uses (ticket 534ea49 Phase 2 Slice A) -- sizing auto-pin from
        # plan["model_bytes"] (the WHOLE model) over-requested whenever a
        # smart-training plan keeps some blocks resident.
        gib = 1024 ** 3
        desired_pin_bytes = cls._desired_pin_bytes_for_offload_ids(
            module, plan["offload_ids"], pinned_weight_gib
        )
        if use_pinned_arena:
            # See the matching rationale in attach(): the arena pins the
            # streamed set and pinned weights bypass bounce staging, so
            # reserving a bounce window for the same weights double-counts
            # the host-pin budget and clips the arena's cap below the
            # streamed set. resolved_pin_gib below is recorded in
            # _attach_args and becomes every later re-attach's explicit
            # budget, so the reservation must be dropped here too or the
            # whole run inherits the undersized cap.
            bounce_reserve_bytes = 0
        else:
            bounce_reserve_bytes = cls._planned_bounce_reserve_bytes(
                module,
                plan["offload_ids"],
                device,
                block_stream_only=block_stream_only,
            )
        if use_pinned_arena and desired_pin_bytes > 0:
            # Reclaim the retained torch host-pin cache before measuring the
            # cap: the resolved gib below is recorded in _attach_args and
            # becomes the EXPLICIT desired budget of every later sampling
            # re-attach -- capping it against cache-clogged headroom
            # undersizes the arena for the whole run (see the matching
            # reconcile in attach()).
            pin_manager.reconcile(desired_pin_bytes, device=device, allow_shrink=False)
        pin_bytes = cls._cap_auto_pin_budget(
            desired_pin_bytes,
            reserve_bytes=bounce_reserve_bytes,
            device=device,
        )
        resolved_pin_gib = pin_bytes / gib
        cls.attach(
            module,
            device,
            offload_percent=0.0,
            ignore_modules=ignore_modules,
            _offload_module_ids=plan["offload_ids"],
            training_strategy="smart",
            pinned_weight_gib=resolved_pin_gib,
            use_pinned_arena=use_pinned_arena,
        )
        module._memory_manager._smart_training_plan = plan
        module._memory_manager._training_must_resident_keys = set(
            plan.get("must_resident_layer_keys", ())
        )
        module._memory_manager._training_pinned_resident_keys = set(pinned_resident_keys)
        module._memory_manager._training_block_stream_only = bool(block_stream_only)
        # Slice 2 (per-block GPU forward staging) is OFF by default and gated
        # behind an explicit env flag. The naive forward-pre-hook stages a block
        # right before it runs and waits on it, which serializes transfer with
        # compute and measured ~2.4x SLOWER than the depth-4 per-Linear ring
        # (which overlaps transfer and compute). It needs cross-block prefetch
        # (stage block i+depth during block i) to be a win — see
        # BLOCK_STREAM_PLAN.md. block_stream_only itself keeps only the proven
        # Slice 1 worker-fill batching.
        gpu_ring = _env("AI_TOOLKIT_BLOCK_STREAM_GPU_RING", "0").lower() not in (
            "0", "false", "no", "",
        )
        if block_stream_only and gpu_ring and torch.device(device).type == "cuda":
            depth = max(1, int(_env("AI_TOOLKIT_BLOCK_STREAM_DEPTH", "2")))
            set_block_stream_enabled(device, True, depth=depth)
            wired = cls._wire_block_stream_forward_hooks(module, device)
            print(
                f"[MemoryManager] block-stream forward staging (EXPERIMENTAL, "
                f"may regress): blocks_hooked={wired} ring_depth={depth}"
            )
        module._memory_manager._training_autotune_enabled = auto_working_reserve
        module._memory_manager._training_autotune_state = {
            "current_working_reserve_gib": plan["working_reserve_bytes"] / (1024 ** 3),
            "danger_working_reserve_gib": None,
            "buckets": {},
            "last_step": -1,
            "last_action": "init_auto" if auto_working_reserve else "manual",
            "stopped": False,
        }
        fp8_training_requested = False
        if fp8_training_forward and torch.device(device).type == "cuda":
            fp8_training_requested = torch.cuda.get_device_capability(device) >= (8, 9)
        module._memory_manager._fp8_training_requested = fp8_training_requested
        fp8_training_layers = 0
        if fp8_training_forward and torch.device(device).type == "cuda":
            fp8_supported = torch.cuda.get_device_capability(device) >= (8, 9)
            if fp8_supported:
                for child in module.modules():
                    weight = getattr(child, "weight", None)
                    if (
                        hasattr(child, "_layer_memory_manager")
                        and child.__class__.__name__ in LINEAR_MODULES
                        and isinstance(weight, torch.nn.Parameter)
                        and hasattr(weight.data, "qdata")
                        and weight.data.qdata.dtype == torch.float8_e4m3fn
                        and not weight.requires_grad
                    ):
                        child._memory_management_fp8_training = True
                        fp8_training_layers += 1
        _FP8_STATS["training_enabled"] = fp8_training_layers > 0
        _FP8_STATS["kernel_calls"] = 0
        _FP8_STATS["fallback_calls"] = 0
        module._memory_manager._fp8_training_layers = fp8_training_layers
        gib = 1024 ** 3
        print(
            f"[MemoryManager] smart training plan: "
            f"model={plan['model_bytes'] / gib:.2f} GiB "
            f"resident={plan['resident_bytes'] / gib:.2f} GiB "
            f"must_resident={plan['must_resident_bytes'] / gib:.2f} GiB "
            f"pinned_resident={plan.get('pinned_resident_bytes', 0) / gib:.2f} GiB "
            f"compile_ready={(plan.get('training_compile_readiness') or {}).get('ready_blocks', 0)}/"
            f"{(plan.get('training_compile_readiness') or {}).get('pinned_blocks', 0)} blocks "
            f"generic_resident={plan['generic_resident_bytes'] / gib:.2f} GiB "
            + (
                f"block_stream_only_resident={plan.get('block_stream_resident_bytes', 0) / gib:.2f} GiB "
                if plan.get("block_stream_only") else ""
            )
            + f"streamed_layers={plan['offloaded_layers']}/{plan['candidate_layers']} "
            f"gpu_stream={plan['gpu_stream_budget_bytes'] / gib:.2f}/"
            f"{plan['gpu_stream_need_bytes'] / gib:.2f} GiB "
            f"training_working_reserve={plan['working_reserve_bytes'] / gib:.2f} GiB "
            f"wddm_margin={plan['wddm_margin_bytes'] / gib:.2f} GiB "
            f"wddm_hard={plan.get('wddm_hard_bytes', 0) / gib:.2f} GiB "
            f"free={plan['free_bytes'] / gib:.2f} GiB "
            f"resident_growth_allowed={plan['resident_growth_allowed']} "
            f"blocked={plan['resident_growth_blocked_reason']} "
            f"fits={plan['fits']}"
        )
        if fp8_training_forward:
            print(
                f"[MemoryManager] native FP8 training: "
                f"{'enabled' if fp8_training_layers else 'unavailable'} "
                f"({fp8_training_layers} streamed linear layers)"
            )
        if _OFFLOAD_PREFETCH_ENABLED and torch.device(device).type == "cuda":
            cls._attach_prefetch_pool(module, device)
        return plan


    @classmethod
    def attach_smart_training_immutable(
        cls,
        module,
        device,
        *,
        canonical_modules,
        working_reserve_gib=2.0,
        ignore_modules=None,
        wddm_margin_gib=None,
        wddm_hard_gib=None,
        fp8_training_forward=False,
        pinned_resident_keys=None,
        block_stream_only=False,
        wddm_spill_reserve_pct=None,
        eager_promote_free_gib=0.0,
        eager_promote_max_blocks=4,
    ):
        """Attach the legacy manager only to non-canonical singleton leaves.

        The planner still sees every Linear, so its per-Linear residency
        decision feeds ``ResidencyPlan.from_smart_plan`` for canonical block
        sidecars. The mutation mechanism is split: canonical leaves are ignored
        by legacy attach/move paths, while singleton layers keep the established
        manager behavior. Compiled canonical layouts are fixed after this cold
        plan, so live legacy autotune is deliberately disabled for this backend.
        """
        cls._apply_wddm_hard_allocator_cap(device, wddm_hard_gib)
        planner_ignore = list(ignore_modules or [])
        canonical_modules = list(canonical_modules or [])
        canonical_ids = {id(child) for child in canonical_modules}
        pinned_resident_keys = set(pinned_resident_keys or ())
        try:
            auto_working_reserve = float(working_reserve_gib) < 0
        except (TypeError, ValueError):
            auto_working_reserve = str(working_reserve_gib).lower() == "auto"
        if auto_working_reserve:
            working_reserve_gib = cls._training_auto_seed_working_reserve_gib()
        resolved_hard = (
            float(_env("AI_TOOLKIT_TRAINING_WDDM_HARD_GIB", "1.0"))
            if wddm_hard_gib is None
            else float(wddm_hard_gib)
        )
        resolved_margin = cls._resolve_wddm_margin_gib(
            device, wddm_margin_gib, hard_gib=resolved_hard
        )
        bounce_pool.set_spill_reserve_policy(
            floor_gib=resolved_margin, pct=wddm_spill_reserve_pct
        )
        plan = cls.smart_training_plan(
            module,
            device,
            working_reserve_gib,
            planner_ignore,
            wddm_margin_gib=resolved_margin,
            wddm_hard_gib=resolved_hard,
            pinned_resident_keys=pinned_resident_keys,
            cold_growth=not auto_working_reserve,
            block_stream_only=block_stream_only,
        )
        legacy_ignore = list(
            dict.fromkeys(planner_ignore + canonical_modules)
        )
        legacy_ignore_ids = {id(child) for child in legacy_ignore}

        runtime_candidate_ids = {
            id(child)
            for _name, child in module.named_modules()
            if id(child) not in legacy_ignore_ids
            and (
                child.__class__.__name__ in LINEAR_MODULES
                or child.__class__.__name__ in CONV_MODULES
            )
        }
        # The immutable runtime is the sole backend: the canonical blocks
        # stream through the arena, and every non-canonical singleton module
        # (tmlp, tproj, first/last projections, text fusion, ...) stays
        # resident. Offloading singletons would install per-Linear streaming
        # forwards (LinearLayerMemoryManager._mm_forward) that both defeat the
        # resident-singleton design and crash compiled sampling. Keep them
        # resident by streaming nothing here; the planner output still drives
        # canonical block sidecars via ResidencyPlan.from_smart_plan.
        if hasattr(module, "_memory_manager"):
            raise RuntimeError(
                "immutable backend requires exclusive memory-manager attachment"
            )
        mm = cls(module, device, pinned_weight_gib=0.0)
        module._memory_manager = mm
        mm._attach_args = {
            "device": device,
            "offload_percent": 0.0,
            "ignore_modules": list(legacy_ignore),
            "training_strategy": "smart_immutable",
            "pinned_weight_gib": 0.0,
            "use_pinned_arena": False,
        }
        module._mm_to = module.to
        module.to = mm.memory_managed_to
        mm.unmanaged_modules.extend(legacy_ignore)
        mm._training_runtime_candidate_ids = set(runtime_candidate_ids)
        mm._smart_training_plan = plan
        mm._training_must_resident_keys = set(
            plan.get("must_resident_layer_keys", ())
        )

        mm._training_pinned_resident_keys = pinned_resident_keys
        mm._training_block_stream_only = bool(block_stream_only)
        mm._training_autotune_enabled = bool(auto_working_reserve)
        mm._training_eager_promote_free_gib = max(
            0.0, float(eager_promote_free_gib or 0.0)
        )
        mm._training_eager_promote_max_blocks = max(
            1, int(eager_promote_max_blocks or 1)
        )
        mm._immutable_planner_ignore_modules = planner_ignore
        mm._canonical_leaf_ids = canonical_ids
        module._mm_canonical_leaf_ids = canonical_ids
        module._mm_immutable_planner_ignore_modules = planner_ignore
        fp8_requested = bool(
            fp8_training_forward
            and torch.device(device).type == "cuda"
            and torch.cuda.get_device_capability(device) >= (8, 9)
        )
        mm._fp8_training_requested = fp8_requested
        cls._refresh_training_fp8_flags(module, mm)
        if _OFFLOAD_PREFETCH_ENABLED and torch.device(device).type == "cuda":
            cls._attach_prefetch_pool(module, device)
        return plan

    @classmethod
    def _training_layout_candidates(cls, module, ignore_modules=None, pinned_resident_keys=None):
        ignored = {id(item) for item in (ignore_modules or [])}
        pinned_keys = set(pinned_resident_keys or ())
        seen = set()
        for name, child in module.named_modules():
            if id(child) in ignored or id(child) in seen:
                continue
            if (
                child.__class__.__name__ not in LINEAR_MODULES
                and child.__class__.__name__ not in CONV_MODULES
            ):
                continue
            seen.add(id(child))
            group_key = cls._offload_group_key(name)
            yield {
                "name": name,
                "group_key": group_key,
                "pinned_resident": group_key in pinned_keys,
                "module": child,
                "resident_bytes": cls._direct_module_bytes(child),
                "stream_bytes": cls._training_stream_bytes(child),
                "managed": hasattr(child, "_layer_memory_manager"),
            }

    @classmethod
    def _refresh_training_plan_from_layout(
        cls,
        module,
        mm,
        working_reserve_gib=None,
    ):
        old_plan = getattr(mm, "_smart_training_plan", {}) or {}
        args = getattr(mm, "_attach_args", {}) or {}
        ignore = args.get("ignore_modules", [])
        pinned_keys = set(
            getattr(mm, "_training_pinned_resident_keys", set())
        )
        device = torch.device(args.get("device", mm.process_device))
        gib = 1024 ** 3
        if working_reserve_gib is None:
            working_reserve_bytes = int(old_plan.get("working_reserve_bytes", 0))
        else:
            working_reserve_bytes = int(float(working_reserve_gib) * gib)
        wddm_hard_bytes = int(
            old_plan.get(
                "wddm_hard_bytes",
                float(_env("AI_TOOLKIT_TRAINING_WDDM_HARD_GIB", "1.0")) * gib,
            )
        )
        wddm_margin_bytes = max(
            wddm_hard_bytes,
            int(
                old_plan.get(
                    "wddm_margin_bytes",
                    float(_env("AI_TOOLKIT_TRAINING_WDDM_MARGIN_GIB", "1.0"))
                    * gib,
                )
            ),
        )
        try:
            free_bytes = cls._torch_allocatable_bytes(device)
        except Exception:
            free_bytes = int(old_plan.get("free_bytes", 0))

        candidates = list(cls._training_layout_candidates(module, ignore, pinned_keys))
        offloaded = [item for item in candidates if item["managed"]]
        offloaded_ids = {id(item["module"]) for item in offloaded}
        offloaded_resident = sum(item["resident_bytes"] for item in offloaded)
        pinned_resident_bytes = sum(
            item["resident_bytes"] for item in candidates
            if item.get("pinned_resident") and not item["managed"]
        )
        total_model_bytes = cls._module_bytes(module)
        resident_bytes = max(0, total_model_bytes - offloaded_resident)
        stream_need_bytes = sum(
            sorted((item["stream_bytes"] for item in offloaded), reverse=True)[
                :PIPELINE_DEPTH
            ]
        )
        old_budget = int(
            old_plan.get("gpu_stream_budget_bytes", old_plan.get("ring_bytes", 0))
        )
        gpu_stream_budget_bytes = min(stream_need_bytes, max(old_budget, 0))
        usable_bytes = max(0, free_bytes - wddm_margin_bytes - working_reserve_bytes)
        compile_readiness = None
        if hasattr(module, "training_compile_readiness"):
            try:
                compile_readiness = module.training_compile_readiness(pinned_keys)
            except Exception:
                compile_readiness = None
        old_plan.update(
            {
                "offload_ids": offloaded_ids,
                "offloaded_layers": len(offloaded),
                "candidate_layers": len(candidates),
                "model_bytes": total_model_bytes,
                "resident_bytes": resident_bytes,
                "pinned_resident_bytes": pinned_resident_bytes,
                "pinned_resident_keys": set(pinned_keys),
                "training_compile_readiness": compile_readiness,
                "must_resident_bytes": min(
                    old_plan.get("must_resident_bytes", resident_bytes),
                    resident_bytes,
                ),
                "generic_resident_bytes": max(
                    0,
                    resident_bytes - old_plan.get("must_resident_bytes", 0),
                ),
                "ring_bytes": gpu_stream_budget_bytes,
                "gpu_stream_need_bytes": stream_need_bytes,
                "gpu_stream_budget_bytes": gpu_stream_budget_bytes,
                "working_reserve_bytes": working_reserve_bytes,
                "wddm_margin_bytes": wddm_margin_bytes,
                "wddm_hard_bytes": wddm_hard_bytes,
                "usable_bytes": usable_bytes,
                "free_bytes": free_bytes,
                "fits": resident_bytes + stream_need_bytes <= usable_bytes,
            }
        )
        mm._smart_training_plan = old_plan
        return old_plan

    @classmethod
    def _refresh_training_fp8_flags(cls, module, mm):
        enabled = bool(getattr(mm, "_fp8_training_requested", False))
        fp8_layers = 0
        for child in module.modules():
            if hasattr(child, "_memory_management_fp8_training"):
                del child._memory_management_fp8_training
            if hasattr(child, "_memory_management_training_compile_fp8"):
                del child._memory_management_training_compile_fp8
            if not enabled or not hasattr(child, "_layer_memory_manager"):
                continue
            weight = getattr(child, "weight", None)
            if (
                child.__class__.__name__ in LINEAR_MODULES
                and isinstance(weight, torch.nn.Parameter)
                and hasattr(weight.data, "qdata")
                and weight.data.qdata.dtype == torch.float8_e4m3fn
                and not weight.requires_grad
            ):
                child._memory_management_fp8_training = True
                fp8_layers += 1
        mm._fp8_training_layers = fp8_layers
        _FP8_STATS["training_enabled"] = fp8_layers > 0

    @classmethod
    def _register_training_prefetch_sources(cls, module, mm):
        pool = getattr(mm, "_prefetch_pool", None)
        if pool is None:
            return
        sources = []
        for child in module.modules():
            key = getattr(child, "_mm_layer_key", None)
            if key is not None and hasattr(child, "_layer_memory_manager"):
                sources.append((key, child))
        if hasattr(pool, "sync_sources"):
            pool.sync_sources(sources)
        else:
            for key, child in sources:
                pool.register_source(key, child)
    @classmethod
    def _demote_training_layers(
        cls,
        module,
        mm,
        count,
        *,
        largest=True,
    ):
        args = getattr(mm, "_attach_args", {}) or {}
        pinned_keys = set(
            getattr(mm, "_training_pinned_resident_keys", set())
        )
        must_resident_keys = set(
            getattr(mm, "_training_must_resident_keys", set())
        )
        runtime_candidate_ids = getattr(
            mm,
            "_training_runtime_candidate_ids",
            None,
        )
        layout = list(
            cls._training_layout_candidates(
                module,
                args.get("ignore_modules", []),
                pinned_keys,
            )
        )
        # In block_stream_only mode, non-block resident layers are kept resident
        # by design — the live controller must not demote them back to streaming,
        # which would reintroduce the scattered small transfers this mode avoids.
        block_only = bool(getattr(mm, "_training_block_stream_only", False))
        block_parents = (
            cls._streaming_block_parents(item["group_key"] for item in layout)
            if block_only else set()
        )
        
        def _streamable(item):
            if not block_only:
                return True
            return cls._block_parent_of(item["group_key"]) in block_parents

        candidates = [
            item
            for item in layout
            if not item["managed"]
            and not item.get("pinned_resident")
            and item["name"] not in must_resident_keys
            and (
                runtime_candidate_ids is None
                or id(item["module"]) in runtime_candidate_ids
            )
            and not any(
                parameter.requires_grad
                for parameter in item["module"].parameters(recurse=False)
            )
            and _streamable(item)
        ]
        candidates.sort(
            key=lambda item: item["resident_bytes"], reverse=bool(largest)
        )
        changed = 0
        demoted_names = []

        for item in candidates[: max(0, int(count))]:
            if cls.demote_layer(
                item["module"],
                mm,
                layer_key=item["name"],
            ):
                changed += 1
                demoted_names.append(item["name"])

        if changed:
            cls._register_training_prefetch_sources(module, mm)
            cls._refresh_training_fp8_flags(module, mm)
            cls._refresh_training_plan_from_layout(module, mm)
            cls._invalidate_manual_training_shape_peaks(mm)
            cls.reset_trace_due_to_execution_shape_change()
            cls._clear_cuda_pipeline_state()

        if demoted_names and cls._diagnostics_enabled():
            print(
                "[MemoryManager] training demoted modules: "
                + ", ".join(demoted_names)
            )

        return changed

    @classmethod
    def _next_promotion_layer_bytes(cls, module, mm):
        """Return resident bytes for the next runtime block or legacy layer."""
        runtime = getattr(module, "_immutable_runtime", None)
        next_runtime_block = getattr(
            runtime,
            "next_training_promotion_bytes",
            None,
        )
        if next_runtime_block is not None:
            return int(next_runtime_block())

        args = getattr(mm, "_attach_args", {}) or {}
        pinned_keys = set(
            getattr(mm, "_training_pinned_resident_keys", set())
        )
        candidates = [
            item
            for item in cls._training_layout_candidates(
                module,
                args.get("ignore_modules", []),
                pinned_keys,
            )
            if item["managed"]
        ]
        if not candidates:
            return 0
        candidates.sort(
            key=lambda item: (
                0 if item.get("pinned_resident") else 1,
                item["resident_bytes"],
            )
        )
        return int(candidates[0]["resident_bytes"])

    @classmethod
    def _promote_training_layer(
        cls, module, mm, device, *, cache_pad_gib, wddm_stop_gib, max_blocks=1
    ):
        runtime = getattr(module, "_immutable_runtime", None)
        promote_runtime = getattr(
            runtime,
            "increase_training_residency",
            None,
        )
        if promote_runtime is not None:
            driver_free_bytes = vram_budget.device_free_bytes(device)
            allocatable_bytes = cls._torch_allocatable_bytes(device)
            allocator_cached_bytes = max(
                0,
                allocatable_bytes - driver_free_bytes,
            )
            gib = 1024 ** 3
            cache_pad_bytes = int(float(cache_pad_gib) * gib)
            stop_bytes = int(float(wddm_stop_gib) * gib)
            available_growth_bytes = max(
                0,
                allocator_cached_bytes
                + max(0, driver_free_bytes - stop_bytes)
                - cache_pad_bytes,
            )
            result = promote_runtime(
                available_growth_bytes,
                max_blocks=max(1, int(max_blocks)),
            )
            if not result.get("added_blocks"):
                return 0, "no_immutable_block_fits"

            try:
                torch.cuda.synchronize(device)
                free_after = vram_budget.device_free_bytes(device)
            except Exception:
                free_after = stop_bytes
            if free_after < stop_bytes:
                previous_plan = result["previous_plan"]
                runtime.set_residency_plan(previous_plan)
                module._mm_immutable_training_plan = previous_plan
                return 0, "validated_low_free"

            actual_growth = int(
                result.get("actual_growth_bytes", 0) or 0
            )
            updated_plan = dict(
                getattr(mm, "_smart_training_plan", {}) or {}
            )
            updated_plan["resident_bytes"] = (
                int(updated_plan.get("resident_bytes", 0))
                + actual_growth
            )
            updated_plan["generic_resident_bytes"] = (
                int(updated_plan.get("generic_resident_bytes", 0))
                + actual_growth
            )
            updated_plan["offloaded_layers"] = max(
                0,
                int(updated_plan.get("offloaded_layers", 0))
                - len(result.get("added_leaf_keys", ())),
            )
            mm._smart_training_plan = updated_plan
            cls._invalidate_manual_training_shape_peaks(mm)
            return len(result["added_blocks"]), "promote_immutable_block"

        args = getattr(mm, "_attach_args", {}) or {}
        pinned_keys = set(getattr(mm, "_training_pinned_resident_keys", set()))
        candidates = [
            item for item in cls._training_layout_candidates(
                module, args.get("ignore_modules", []), pinned_keys
            )
            if item["managed"]
        ]
        candidates.sort(
            key=lambda item: (
                0 if item.get("pinned_resident") else 1,
                item["resident_bytes"],
            )
        )
        if not candidates:
            return 0, "no_offloaded_layers"
        driver_free_bytes = vram_budget.device_free_bytes(device)
        allocatable_bytes = cls._torch_allocatable_bytes(device)
        allocator_cached_bytes = max(0, allocatable_bytes - driver_free_bytes)
        gib = 1024 ** 3
        cache_pad_bytes = int(float(cache_pad_gib) * gib)
        stop_bytes = int(float(wddm_stop_gib) * gib)
        for item in candidates:
            need = int(item["resident_bytes"] + cache_pad_bytes)
            if need > allocatable_bytes:
                continue
            driver_bytes_needed = max(0, need - allocator_cached_bytes)
            if driver_free_bytes - driver_bytes_needed < stop_bytes:
                continue
            child = item["module"]
            if not cls.promote_layer(child):
                continue
            try:
                torch.cuda.synchronize(device)
                free_after = vram_budget.device_free_bytes(device)
            except Exception:
                free_after = stop_bytes
            if free_after < stop_bytes:
                cls.demote_layer(child, mm, layer_key=item["name"])
                cls._refresh_training_fp8_flags(module, mm)
                cls._refresh_training_plan_from_layout(module, mm)
                cls._invalidate_manual_training_shape_peaks(mm)
                cls.reset_trace_due_to_execution_shape_change()
                cls._clear_cuda_pipeline_state()
                return 0, "validated_low_free"
            cls._refresh_training_fp8_flags(module, mm)
            cls._refresh_training_plan_from_layout(module, mm)
            cls._invalidate_manual_training_shape_peaks(mm)
            cls.reset_trace_due_to_execution_shape_change()
            cls._clear_cuda_pipeline_state()
            return 1, "promote"
        return 0, "stop_line"

    @classmethod
    def _unpin_training_layer_for_shared_relief(cls, module, mm):
        args = getattr(mm, "_attach_args", {}) or {}
        pinned_keys = set(getattr(mm, "_training_pinned_resident_keys", set()))
        candidates = [
            item for item in cls._training_layout_candidates(
                module, args.get("ignore_modules", []), pinned_keys
            )
            if item["managed"] and int(getattr(item["module"], "_mm_pinned_bytes", 0) or 0) > 0
        ]
        candidates.sort(
            key=lambda item: int(getattr(item["module"], "_mm_pinned_bytes", 0) or 0),
            reverse=True,
        )
        for item in candidates:
            released = unpin_layer(item["module"])
            if released > 0:
                return 1, "unpin_shared", released
        return 0, "unpin_unavailable", 0

    @classmethod
    def _relieve_shared_cliff(
        cls,
        module,
        mm,
        device,
        *,
        shared_snapshot,
        dedicated_free_gib,
        dedicated_promote_free_gib,
        cache_pad_gib,
        wddm_stop_gib,
    ):
        decision = cls._shared_cliff_relief_decision(
            (shared_snapshot or {}).get("raw_headroom_gib"),
            (shared_snapshot or {}).get("margin_gib"),
            dedicated_free_gib,
            dedicated_promote_free_gib,
        )
        if decision == "hold":
            return 0, "hold", 0
        if decision == "promote":
            changed, action = cls._promote_training_layer(
                module,
                mm,
                device,
                cache_pad_gib=cache_pad_gib,
                wddm_stop_gib=wddm_stop_gib,
            )
            if changed:
                return changed, "promote_shared", 0
        changed, action, released = cls._unpin_training_layer_for_shared_relief(module, mm)
        return changed, action, released

    @classmethod
    def _promote_training_layers(
        cls, module, mm, device, *, cache_pad_gib, wddm_stop_gib, budget_gib, max_count=0
    ):
        """Promote a batch of the smallest streamed layers in one shot.

        Each move is still validated against real free VRAM and the stop line
        (per-layer ``synchronize`` + ``mem_get_info``, exactly like
        ``_promote_training_layer``), but the expensive part — the prefetch trace
        reset and plan refresh — happens ONCE, after the whole batch. That turns
        the convergence from ~N trace resets (one per layer per cadence) into a
        single reset, which the offline simulator showed is the difference between
        ~225 churning steps and ~1. ``budget_gib`` caps how much free VRAM the
        batch may consume so we land mid-band rather than at the stop line.
        Returns ``(count, action)``.
        """
        args = getattr(mm, "_attach_args", {}) or {}
        pinned_keys = set(getattr(mm, "_training_pinned_resident_keys", set()))
        candidates = [
            item for item in cls._training_layout_candidates(
                module, args.get("ignore_modules", []), pinned_keys
            )
            if item["managed"]
        ]
        candidates.sort(
            key=lambda item: (
                0 if item.get("pinned_resident") else 1,
                item["resident_bytes"],
            )
        )
        if not candidates:
            return 0, "no_offloaded_layers"
        gib = 1024 ** 3
        cache_pad_bytes = int(float(cache_pad_gib) * gib)
        stop_bytes = int(float(wddm_stop_gib) * gib)
        budget_bytes = max(0, int(float(budget_gib) * gib))
        consumed_bytes = 0
        promoted = 0
        for item in candidates:
            if max_count and promoted >= max_count:
                break
            if consumed_bytes >= budget_bytes:
                break
            # Re-read real free each layer: every promote consumes VRAM, so the
            # batch cannot run on a single stale snapshot without risking an OOM.
            allocatable_bytes = cls._torch_allocatable_bytes(device)
            need = int(item["resident_bytes"] + cache_pad_bytes)
            if need > allocatable_bytes:
                continue
            driver_free_bytes = vram_budget.device_free_bytes(device)
            allocator_cached_bytes = max(0, allocatable_bytes - driver_free_bytes)
            driver_bytes_needed = max(0, need - allocator_cached_bytes)
            if driver_free_bytes - driver_bytes_needed < stop_bytes:
                break
            child = item["module"]
            if not cls.promote_layer(child):
                continue
            try:
                torch.cuda.synchronize(device)
                free_after = vram_budget.device_free_bytes(device)
            except Exception:
                free_after = stop_bytes
            if free_after < stop_bytes:
                # Overshot the stop line: undo this one and stop the batch.
                cls.demote_layer(child, mm, layer_key=item["name"])
                break
            promoted += 1
            consumed_bytes += int(item["resident_bytes"])
        if not promoted:
            return 0, "stop_line"
        cls._refresh_training_fp8_flags(module, mm)
        cls._refresh_training_plan_from_layout(module, mm)
        cls._invalidate_manual_training_shape_peaks(mm)
        cls.reset_trace_due_to_execution_shape_change()
        cls._clear_cuda_pipeline_state()
        return promoted, "promote_batch"

    @staticmethod
    def _torch_allocatable_bytes(device):
        """Driver-free plus allocator cache that PyTorch can reuse directly."""
        free_bytes = vram_budget.device_free_bytes(device)
        reserved_bytes = torch.cuda.memory_reserved(device)
        allocated_bytes = torch.cuda.memory_allocated(device)
        return int(free_bytes + max(0, reserved_bytes - allocated_bytes))

    @classmethod
    def prepare_training_memory_for_shape(cls, module, device=None, shape_key=None):
        """Pre-step guard using learned per-shape peaks and live DXGI LOCAL budget."""
        while module is not None and not hasattr(module, "_memory_manager"):
            wrapped = getattr(module, "module", None)
            if wrapped is None or wrapped is module:
                return None
            module = wrapped
        if module is None or not hasattr(module, "_memory_manager"):
            return None
        mm = module._memory_manager
        plan = getattr(mm, "_smart_training_plan", None)
        if plan is None:
            return None
        try:
            device = torch.device(device or mm.process_device)
        except (TypeError, ValueError, RuntimeError):
            return None
        if device.type != "cuda" or not torch.cuda.is_available():
            return None

        bucket = cls._training_shape_peak_bucket(mm, shape_key)
        if not bucket:
            return None
        learned_peak_gib = float(bucket.get("peak_allocated_gib", 0.0) or 0.0)
        learned_peak_reserved_gib = float(bucket.get("peak_reserved_gib", 0.0) or 0.0)
        if learned_peak_gib <= 0.0:
            return None

        gib = 1024 ** 3
        hard_gib = max(
            float(_env("AI_TOOLKIT_TRAINING_WDDM_HARD_GIB", "1.0")),
            float(plan.get("wddm_hard_bytes", 0)) / gib,
        )
        margin_gib = max(
            hard_gib,
            float(plan.get("wddm_margin_bytes", 0)) / gib,
        )
        fallback_target_gib = max(
            hard_gib,
            float(_env("AI_TOOLKIT_TRAINING_WDDM_STOP_GIB", str(hard_gib + 0.5))),
        )
        retreat_layers = max(1, int(_env("AI_TOOLKIT_TRAINING_RETREAT_LAYERS", "1")))
        max_demote = int(_env("AI_TOOLKIT_TRAINING_SAFETY_MAX_DEMOTE", "12"))

        def _pressure():
            local = cls._dxgi_local_budget_snapshot_bytes(device)
            current_reserved = int(torch.cuda.memory_reserved(device))
            current_allocated = int(torch.cuda.memory_allocated(device))
            peak_allocated = int(max(0.0, learned_peak_gib) * gib)
            predicted_local = cls._predict_dxgi_local_peak_bytes(
                local,
                current_reserved_bytes=current_reserved,
                current_allocated_bytes=current_allocated,
                peak_reserved_bytes=int(max(0.0, learned_peak_reserved_gib) * gib),
                peak_allocated_bytes=peak_allocated,
            )
            # The physical (NVML) signal ALWAYS participates: the DXGI LOCAL
            # budget is a per-process OS grant that can exceed what the dedicated
            # cliff tolerates once other processes' usage is counted, so DXGI
            # alone can bless a residency that overfills the card. Note
            # mem_get_info is NOT that physical signal -- it is a per-process
            # promise and over-reports just like DXGI (see vram_budget's module
            # docstring); device_mem_info is NVML-backed.
            free_bytes, total_bytes = vram_budget.device_mem_info(device)
            free_gib = free_bytes / gib
            total_gib = total_bytes / gib
            reserved_gib = current_reserved / gib
            physical_predicted_peak_free_gib = cls._training_cliff_predicted_peak_free_gib(
                total_gib, free_gib, reserved_gib, learned_peak_gib
            )
            physical = {
                "source": "cuda_free",
                "pressure": physical_predicted_peak_free_gib < fallback_target_gib,
                "predicted_peak_free_gib": physical_predicted_peak_free_gib,
                "target_free_gib": fallback_target_gib,
            }
            if local is None or predicted_local is None:
                return physical

            target_usage = int(local["budget_bytes"] - margin_gib * gib)
            dxgi = {
                "source": "dxgi_local",
                "pressure": predicted_local > target_usage,
                "predicted_local_usage_gib": predicted_local / gib,
                "target_local_usage_gib": target_usage / gib,
                "local_budget_gib": local["budget_bytes"] / gib,
                "local_usage_gib": local["usage_bytes"] / gib,
                "predicted_peak_free_gib": (target_usage - predicted_local) / gib,
            }
            return cls._training_guard_pressure(dxgi, physical)

        before = _pressure()
        if not before.get("pressure"):
            return None

        # First relief rung: return idle allocator cache. This never changes the
        # residency layout and therefore does not invalidate the learned peak.
        cached_before = max(
            0,
            int(torch.cuda.memory_reserved(device))
            - int(torch.cuda.memory_allocated(device)),
        )
        cache_reclaimed = 0
        pressure = before
        if cached_before:
            torch.cuda.empty_cache()
            cache_reclaimed = max(
                0,
                cached_before
                - max(
                    0,
                    int(torch.cuda.memory_reserved(device))
                    - int(torch.cuda.memory_allocated(device)),
                ),
            )
            pressure = _pressure()
            if not pressure.get("pressure"):
                return {
                    "manual_safety": not bool(
                        getattr(mm, "_training_autotune_enabled", False)
                    ),
                    "action": "prestep_empty_cache",
                    "demoted_layers": 0,
                    "source": before.get("source"),
                    "before": before,
                    "after": pressure,
                    "allocator_cache_reclaimed_bytes": cache_reclaimed,
                    "canonical_relief": None,
                    "shape_peak_invalidated": False,
                }

        dxgi_gap_gib = max(
            0.0,
            float(pressure.get("predicted_local_usage_gib", 0.0) or 0.0)
            - float(pressure.get("target_local_usage_gib", 0.0) or 0.0),
        )
        if pressure.get("source") == "cuda_free":
            physical_peak_free = pressure.get("predicted_peak_free_gib")
            physical_target_free = pressure.get("target_free_gib")
        else:
            physical_peak_free = pressure.get("physical_predicted_peak_free_gib")
            physical_target_free = pressure.get("physical_target_free_gib")
        physical_gap_gib = max(
            0.0,
            float(physical_target_free or 0.0)
            - float(physical_peak_free or 0.0),
        )
        required_relief_bytes = max(
            1,
            int(max(dxgi_gap_gib, physical_gap_gib) * gib),
        )

        # Second relief rung: immutable canonical sidecars. The extension-owned
        # executor reconciles a subset-only TRAIN plan and rebuilds its program;
        # legacy demote_layer must never touch canonical host Parameters.
        canonical_relief = None
        canonical_relieved_bytes = 0
        executor = getattr(module, "_immutable_runtime", None)
        reduce_canonical = getattr(
            executor, "reduce_training_residency", None
        )
        if reduce_canonical is not None and required_relief_bytes > 0:
            canonical_relief = reduce_canonical(required_relief_bytes)
            relieved_bytes = int(
                canonical_relief.get("relieved_bytes", 0) or 0
            )
            canonical_relieved_bytes = relieved_bytes
            if relieved_bytes:
                cls._invalidate_manual_training_shape_peaks(mm)
                updated_plan = dict(plan)
                updated_plan["resident_bytes"] = max(
                    0,
                    int(updated_plan.get("resident_bytes", 0))
                    - relieved_bytes,
                )
                updated_plan["generic_resident_bytes"] = max(
                    0,
                    int(updated_plan.get("generic_resident_bytes", 0))
                    - relieved_bytes,
                )
                updated_plan["offloaded_layers"] = int(
                    updated_plan.get("offloaded_layers", 0)
                ) + len(canonical_relief.get("removed_leaf_keys", ()))
                mm._smart_training_plan = updated_plan
                plan = updated_plan

                remaining_gap = max(
                    0, required_relief_bytes - relieved_bytes
                )
                remaining_adjustable = int(
                    canonical_relief.get(
                        "remaining_adjustable_bytes", 0
                    )
                    or 0
                )
                if remaining_gap == 0 or remaining_adjustable > 0:
                    pressure = {
                        "source": before.get("source"),
                        "pressure": None,
                        "prediction_valid": False,
                        "reason": "canonical_layout_changed_relearn_required",
                    }
                    return {
                        "manual_safety": not bool(
                            getattr(mm, "_training_autotune_enabled", False)
                        ),
                        "action": "prestep_reduce_canonical",
                        "demoted_layers": 0,
                        "source": before.get("source"),
                        "before": before,
                        "after": pressure,
                        "allocator_cache_reclaimed_bytes": cache_reclaimed,
                        "required_relief_bytes": required_relief_bytes,
                        "canonical_relief": canonical_relief,
                        "learned_peak_allocated_gib": learned_peak_gib,
                        "learned_peak_reserved_gib": learned_peak_reserved_gib,
                        "shape_peak_invalidated": True,
                    }

        # Final emergency rung: only non-canonical singleton leaves are eligible
        # for the legacy manager. This runs only when canonical relief is absent
        # or exhausted while a predicted hard-floor deficit remains.
        demoted = 0
        if max_demote > 0:
            demoted = cls._demote_training_layers(
                module, mm, retreat_layers, largest=True
            )
            if demoted:
                cls._invalidate_manual_training_shape_peaks(mm)
                pressure = {
                    "source": before.get("source"),
                    "pressure": None,
                    "prediction_valid": False,
                    "reason": "singleton_layout_changed_relearn_required",
                }

        if cls._diagnostics_enabled():
            print(
                "[MemoryManager] pre-step training guard: "
                f"cache_reclaimed={cache_reclaimed / gib:.2f} GiB "
                f"canonical_relieved={canonical_relieved_bytes / gib:.2f} GiB "
                f"singleton_demoted={demoted}; "
                + (
                    "layout changed, relearning"
                    if demoted or canonical_relieved_bytes
                    else "no relief available"
                )
            )
        return {
            "manual_safety": not bool(getattr(mm, "_training_autotune_enabled", False)),
            "action": (
                "prestep_reduce_canonical_and_demote_singleton"
                if demoted and canonical_relieved_bytes
                else ("prestep_demote_singleton" if demoted else "prestep_unavailable")
            ),
            "demoted_layers": demoted,
            "source": before.get("source"),
            "before": before,
            "after": pressure,
            "allocator_cache_reclaimed_bytes": cache_reclaimed,
            "required_relief_bytes": required_relief_bytes,
            "canonical_relief": canonical_relief,
            "learned_peak_allocated_gib": learned_peak_gib,
            "learned_peak_reserved_gib": learned_peak_reserved_gib,
            "shape_peak_invalidated": bool(demoted or canonical_relieved_bytes),
        }

    @classmethod
    def auto_tune_training_memory(
        cls,
        module,
        device=None,
        shape_key=None,
        step_num=None,
        step_time_s=None,
        did_oom=False,
        peak_allocated_override=None,
        peak_reserved_override=None,
        observed_driver_free_min_bytes=None,
        observed_driver_total_bytes=None,
        observed_driver_free_samples=None,
    ):
        """Conservative live tuning for smart training offload.

        Enabled by ``layer_offloading_smart_working_reserve_gb: -1``. Mutates only at
        step boundaries: OOM or hard-floor breaches demote resident layers;
        proven slack promotes one smallest streamed layer at a slow cadence.
        """
        while module is not None and not hasattr(module, "_memory_manager"):
            wrapped = getattr(module, "module", None)
            if wrapped is None or wrapped is module:
                return None
            module = wrapped
        if module is None or not hasattr(module, "_memory_manager"):
            return None
        mm = module._memory_manager
        if not getattr(mm, "_training_autotune_enabled", False):
            # Manual working_reserve still gets cliff protection. A fixed budget
            # chooses the activation reserve; it does not license driving into the
            # WDDM spill and staying there. Reclaim idle allocator cache (and only
            # if that is not enough, demote resident layers) at the step boundary.
            return cls._training_cliff_safety_net(
                module,
                mm,
                device,
                shape_key=shape_key,
                did_oom=did_oom,
                peak_allocated_override=peak_allocated_override,
                peak_reserved_override=peak_reserved_override,
            )
        plan = getattr(mm, "_smart_training_plan", None)
        if plan is None:
            return None
        device = torch.device(device or mm.process_device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return None

        gib = 1024 ** 3
        state = getattr(mm, "_training_autotune_state", None) or {}
        state.setdefault("buckets", {})
        state.setdefault("current_working_reserve_gib", plan["working_reserve_bytes"] / gib)
        state.setdefault("danger_working_reserve_gib", None)
        state.setdefault("learned_wddm_hard_gib", None)
        state.setdefault("last_step", -1)
        state.setdefault("stopped", False)
        mm._training_autotune_state = state

        wddm_hard_gib = max(
            float(_env("AI_TOOLKIT_TRAINING_WDDM_HARD_GIB", "1.0")),
            float(plan.get("wddm_hard_bytes", 0)) / gib,
        )
        wddm_stop_gib = float(_env("AI_TOOLKIT_TRAINING_WDDM_STOP_GIB", "1.5"))
        configured_wddm_margin_gib = float(plan.get("wddm_margin_bytes", 0)) / gib
        wddm_stop_gib = max(wddm_stop_gib, configured_wddm_margin_gib)
        pad_gib = float(_env("AI_TOOLKIT_TRAINING_WORKING_RESERVE_PAD_GIB", "0.5"))
        step_gib = float(_env("AI_TOOLKIT_TRAINING_WORKING_RESERVE_STEP_GIB", "0.5"))
        retreat_gib = float(_env("AI_TOOLKIT_TRAINING_RETREAT_GIB", "1.0"))
        retreat_layers = int(_env("AI_TOOLKIT_TRAINING_RETREAT_LAYERS", "1"))
        promote_interval = int(_env("AI_TOOLKIT_TRAINING_PROMOTE_INTERVAL", "4"))
        cache_pad_gib = float(_env("AI_TOOLKIT_TRAINING_CACHE_PAD_GIB", "0.25"))
        min_working_reserve_gib = float(_env("AI_TOOLKIT_TRAINING_MIN_WORKING_RESERVE_GIB", "1.5"))
        stable_windows = int(_env("AI_TOOLKIT_TRAINING_STABLE_WORKING_RESERVE_STEPS", "2"))
        unhealthy_promote_slack_gib = float(
            _env("AI_TOOLKIT_TRAINING_UNHEALTHY_PROMOTE_SLACK_GIB", "1.5")
        )
        timing_spill_ratio = float(
            _env("AI_TOOLKIT_TRAINING_WDDM_TIMING_SPILL_RATIO", "3.0")
        )
        timing_spill_warmup = int(
            _env("AI_TOOLKIT_TRAINING_WDDM_TIMING_SPILL_WARMUP_STEPS", "4")
        )
        timing_spill_pad_gib = float(
            _env("AI_TOOLKIT_TRAINING_WDDM_TIMING_SPILL_PAD_GIB", "0.25")
        )

        diagnostics = cls.training_runtime_diagnostics(
            module,
            device,
            peak_allocated_override=peak_allocated_override,
            peak_reserved_override=peak_reserved_override,
            observed_driver_free_min_bytes=observed_driver_free_min_bytes,
            observed_driver_total_bytes=observed_driver_total_bytes,
            observed_driver_free_samples=observed_driver_free_samples,
        )
        if diagnostics is None:
            return None
        cls._record_manual_training_shape_peak(
            mm,
            shape_key,
            peak_allocated_gib=diagnostics.get("peak_allocated_gb", 0.0),
            peak_reserved_gib=diagnostics.get("peak_reserved_gb", 0.0),
        )
        # Residents on the card that are NOT in our caching allocator: other CUDA
        # processes (browser, compositor, another job) AND our own non-allocator
        # overhead (CUDA context, cuDNN workspaces, compiled-graph constants —
        # ~1.9 GiB here). Measured, not guessed:
        #   other = (total - free) - reserved  ==  device_used - torch_reserved
        # Read at step end, so our own late-materializing overhead is already in.
        other_gib = max(
            0.0, diagnostics["device_used_gb"] - diagnostics["torch_reserved_gb"]
        )
        state["system_reserve_gib"] = other_gib
        safety_gib = float(
            _env("AI_TOOLKIT_TRAINING_WDDM_SAFETY_GIB", "0.5")
        )
        # Margin-to-spill is the "available VRAM" the deadband governs. The
        # allocator-derived estimate misses resolution-switch WDDM pressure, so
        # when the trainer sampled an observed driver-free minimum, let that
        # stricter value govern the bucket and retreat decision.
        estimated_free_gib = cls._available_vram_gib(
            diagnostics["device_total_gb"],
            diagnostics["device_used_gb"],
            diagnostics["torch_reserved_gb"],
            diagnostics["peak_reserved_gb"],
            safety_gib=safety_gib,
        )
        min_device_free_gib = cls._training_governing_free_gib(
            estimated_free_gib, diagnostics
        )
        working_peak_gib = max(
            0.0,
            float(
                diagnostics.get(
                    "working_reserve_peak_gb",
                    diagnostics.get("working_reserve_used_gb", 0.0),
                )
            ),
        )
        allocation_peak_gib = max(
            diagnostics["peak_allocated_gb"]
            - diagnostics["planned_resident_gb"]
            - diagnostics.get("ring_peak_gb", diagnostics["planned_ring_gb"]),
            0.0,
        )
        measured_peak_gib = max(working_peak_gib, allocation_peak_gib)

        bucket_key = shape_key if shape_key is not None else ("default",)
        bucket = state["buckets"].setdefault(
            bucket_key,
            {
                "steps": 0,
                "peak_working_gib": 0.0,
                "min_device_free_gib": 999.0,
                "best_step_time_s": None,
                "last_step_time_s": None,
                "settled": True,
                "no_improve": 0,
                "working_ema_gib": None,
            },
        )
        bucket["steps"] += 1
        # Smooth and signal off the truthful within-step PEAK, never the step-end
        # residual (the trough that previously starved the reserve into a spill).
        working_ema = bucket.get("working_ema_gib")
        bucket["working_ema_gib"] = (
            measured_peak_gib if working_ema is None
            else 0.8 * working_ema + 0.2 * measured_peak_gib
        )
        working_reserve_signal_gib = cls._training_working_reserve_signal(
            measured_peak_gib,
            bucket["working_ema_gib"],
            steps=bucket["steps"],
            stable_windows=stable_windows,
            min_working_reserve_gib=min_working_reserve_gib,
            pad_gib=pad_gib,
        )
        bucket["peak_working_gib"] = max(
            bucket["peak_working_gib"], working_reserve_signal_gib
        )
        bucket["latest_working_signal_gib"] = working_reserve_signal_gib
        bucket["min_device_free_gib"] = min(
            bucket["min_device_free_gib"], min_device_free_gib
        )
        # Most recent free margin for this bucket. Unlike the all-time min above
        # (kept for conservative reserve accounting), this recovers when the real
        # free margin recovers, so the deadband can authorize promotion again once
        # a transient dip passes instead of latching low forever.
        bucket["last_free_gib"] = min_device_free_gib
        previous_best_step_time_s = bucket.get("best_step_time_s")
        if step_time_s is not None:
            last_time = bucket.get("last_step_time_s")
            if last_time is None:
                bucket["last_step_time_s"] = float(step_time_s)
            else:
                bucket["last_step_time_s"] = 0.8 * last_time + 0.2 * float(step_time_s)
            best = bucket.get("best_step_time_s")
            if best is None or bucket["last_step_time_s"] < best * 0.98:
                bucket["best_step_time_s"] = bucket["last_step_time_s"]
                bucket["no_improve"] = 0
            elif bucket["steps"] > 3:
                bucket["no_improve"] += 1

        learned_floor = cls._training_timing_spill_floor(
            step_time_s,
            previous_best_step_time_s,
            min_device_free_gib,
            steps=bucket["steps"],
            warmup_steps=timing_spill_warmup,
            slowdown_ratio=timing_spill_ratio,
            max_signal_free_gib=wddm_stop_gib,
            pad_gib=timing_spill_pad_gib,
        )
        timing_spill = learned_floor is not None
        if timing_spill:
            state["learned_wddm_hard_gib"] = max(
                state.get("learned_wddm_hard_gib") or 0.0,
                learned_floor,
            )
        learned_wddm_hard_gib = state.get("learned_wddm_hard_gib")
        if learned_wddm_hard_gib is not None:
            wddm_hard_gib = max(wddm_hard_gib, float(learned_wddm_hard_gib))
            wddm_stop_gib = max(wddm_stop_gib, wddm_hard_gib)
            plan["wddm_hard_bytes"] = int(wddm_hard_gib * gib)
            if plan.get("wddm_margin_bytes", 0) < plan["wddm_hard_bytes"]:
                plan["wddm_margin_bytes"] = plan["wddm_hard_bytes"]
        current_gib = float(state["current_working_reserve_gib"])
        # Worst-case recent resolution signal governs the reserve: a quiet low-res
        # step must not shrink the budget below what the latest high-res bucket
        # measured. Do not feed WDDM cliffs or OOMs into this number; those are
        # layout-safety signals handled below by move == "down".
        governing_reserve_signal_gib = max(
            [working_reserve_signal_gib]
            + [
                b.get("latest_working_signal_gib", b.get("peak_working_gib", 0.0))
                for b in state["buckets"].values()
            ]
        )
        if did_oom:
            governing_reserve_signal_gib = max(
                governing_reserve_signal_gib,
                max(0.0, current_gib - pad_gib),
            )
        new_working_reserve, danger, action = cls._training_working_reserve_decision(
            current_gib,
            governing_reserve_signal_gib,
            min(bucket["min_device_free_gib"], min_device_free_gib),
            state.get("danger_working_reserve_gib"),
            wddm_hard_gib=wddm_hard_gib,
            wddm_stop_gib=wddm_stop_gib,
            pad_gib=pad_gib,
            step_gib=step_gib,
            retreat_gib=retreat_gib,
        )
        if did_oom and action == "hold":
            action = "oom_hold"
        state["danger_working_reserve_gib"] = danger

        # This is not an allocation knob. It is measured working-set demand,
        # translated into resident-weight space we must leave empty. Do not cap
        # it with an arbitrary max; under-measuring here drives WDDM spills.
        new_working_reserve = max(new_working_reserve, min_working_reserve_gib)

        # Govern layout moves off a free-VRAM deadband (user spec). Demote uses
        # the worst-case *recent* free margin across resolution buckets so
        # high-res safety binds; promote uses only the CURRENT bucket's margin
        # -- see _training_layout_move for why they must differ (a chronically
        # tight bucket must not veto promotion during a roomy one forever).
        demote_governing_free_gib = min(
            [min_device_free_gib]
            + [
                b.get("last_free_gib", min_device_free_gib)
                for b in state["buckets"].values()
            ]
        )
        wddm_hold_high_gib = float(
            _env(
                "AI_TOOLKIT_TRAINING_WDDM_HOLD_HIGH_GIB", str(wddm_stop_gib + step_gib)
            )
        )
        wddm_hold_high_gib = max(wddm_hold_high_gib, wddm_stop_gib + step_gib)
        move = cls._training_layout_move(
            demote_governing_free_gib,
            min_device_free_gib,
            wddm_hard_gib=wddm_hard_gib,
            wddm_hold_high_gib=wddm_hold_high_gib,
            did_oom=did_oom,
        )

        pool = getattr(mm, "_prefetch_pool", None)
        schedule_confidence = diagnostics.get("prefetch_schedule_confidence", "cold")
        prefetch_ok = cls._prefetch_allows_resident_growth(
            pool_present=pool is not None,
            schedule_confidence=schedule_confidence,
            prefetch_healthy=diagnostics.get("prefetch_healthy", False),
        )
        prefetch_reason = diagnostics.get("prefetch_reason")
        prefetch_missing = bool(diagnostics.get("prefetch_missing_schedule", False))
        prefetch_invalid = bool(diagnostics.get("prefetch_invalid_trace", False))
        prefetch_recovery_action = cls._prefetch_recovery_action(
            prefetch_missing=prefetch_missing,
            prefetch_invalid=prefetch_invalid,
        )
        grew_prefetch = False
        changed_layers = 0
        layout_action = "hold"
        shared_snapshot = cls._dxgi_shared_budget_snapshot(device)
        shared_relief = cls._shared_cliff_relief_decision(
            (shared_snapshot or {}).get("raw_headroom_gib"),
            (shared_snapshot or {}).get("margin_gib"),
            min_device_free_gib,
            wddm_hold_high_gib,
        )
        # Dedicated-VRAM safety demotion wins. Otherwise, when the measured DXGI
        # NON_LOCAL headroom is inside the reserved margin, relieve the shared
        # cliff before ordinary prefetch repair or resident growth. Default relief
        # is unpin-to-pageable; promotion is allowed only with roomy dedicated VRAM.
        if move == "down":
            runtime = getattr(module, "_immutable_runtime", None)
            reduce_runtime = getattr(
                runtime,
                "reduce_training_residency",
                None,
            )
            if reduce_runtime is not None:
                relief = reduce_runtime(
                    max(1, int(float(retreat_gib) * gib))
                )
                changed_layers = len(relief.get("removed_blocks", ()))
                relieved_bytes = int(
                    relief.get("relieved_bytes", 0) or 0
                )
                if relieved_bytes:
                    updated_plan = dict(
                        getattr(mm, "_smart_training_plan", {}) or {}
                    )
                    updated_plan["resident_bytes"] = max(
                        0,
                        int(updated_plan.get("resident_bytes", 0))
                        - relieved_bytes,
                    )
                    updated_plan["generic_resident_bytes"] = max(
                        0,
                        int(updated_plan.get("generic_resident_bytes", 0))
                        - relieved_bytes,
                    )
                    updated_plan["offloaded_layers"] = (
                        int(updated_plan.get("offloaded_layers", 0))
                        + len(relief.get("removed_leaf_keys", ()))
                    )
                    mm._smart_training_plan = updated_plan
                    cls._invalidate_manual_training_shape_peaks(mm)
                layout_action = (
                    "demote_immutable_block"
                    if changed_layers
                    else "demote_unavailable"
                )
            else:
                changed_layers = cls._demote_training_layers(
                    module, mm, retreat_layers, largest=True
                )
                layout_action = (
                    "demote"
                    if changed_layers
                    else "demote_unavailable"
                )
            state["stopped"] = False
        elif shared_relief != "hold":
            changed_layers, layout_action, released_bytes = cls._relieve_shared_cliff(
                module,
                mm,
                device,
                shared_snapshot=shared_snapshot,
                dedicated_free_gib=min_device_free_gib,
                dedicated_promote_free_gib=wddm_hold_high_gib,
                cache_pad_gib=cache_pad_gib,
                wddm_stop_gib=wddm_stop_gib,
            )
            if changed_layers and layout_action == "unpin_shared":
                cls._register_training_prefetch_sources(module, mm)
            if cls._diagnostics_enabled() and layout_action != "hold":
                print(
                    "[MemoryManager] shared-budget relief: "
                    f"action={layout_action} changed_layers={changed_layers} "
                    f"released={released_bytes / gib:.2f} GiB "
                    f"shared_raw_headroom={(shared_snapshot or {}).get('raw_headroom_gib')} GiB "
                    f"shared_margin={(shared_snapshot or {}).get('margin_gib')} GiB "
                    f"dedicated_free={min_device_free_gib:.2f} GiB"
                )
        elif (
            prefetch_recovery_action is not None
            and getattr(module, "_immutable_runtime", None) is None
        ):
            cls._invalidate_manual_training_shape_peaks(mm)
            if prefetch_invalid:
                invalidate_offload_trace_for_shape(shape_key)
            else:
                cls.reset_trace_due_to_execution_shape_change()
            if pool is not None and hasattr(pool, "seed_schedule_from_sources"):
                pool.seed_schedule_from_sources()
            layout_action = prefetch_recovery_action
        elif move == "up":
            last_step = int(state.get("last_step", -1))
            step_index = int(step_num if step_num is not None else bucket["steps"])
            # Eager fill (layer_offloading_eager_promote_free_gb). The default climb
            # is one block per cadence window and stops at the WDDM hold floor, which
            # strands GiBs of VRAM on a card that has room to spare. When a free-margin
            # target is configured, promote every block that still fits above THAT
            # margin, every step, up to a per-step block bound. The safety gates below
            # (worst-shape prediction, post-promote free re-check) are unchanged and a
            # target above the hold floor makes each promotion strictly more careful,
            # not less: eager buys speed of climb and reach, never headroom.
            eager_free_gib = float(
                getattr(mm, "_training_eager_promote_free_gib", 0.0) or 0.0
            )
            eager = eager_free_gib > 0.0
            eager_max_blocks = max(
                1, int(getattr(mm, "_training_eager_promote_max_blocks", 4) or 4)
            )
            promote_floor_gib = max(wddm_hold_high_gib, eager_free_gib)
            cadence_ready = (
                eager
                or last_step < 0
                or step_index - last_step >= promote_interval
            )
            # Promote on MEASUREMENT + headroom, NOT on prefetch health. move=="up"
            # already proved peak-free headroom (the deadband); we only require the
            # working set for this bucket to have been measured first — start
            # conservative (singletons only) -> measure -> THEN promote blocks. We do
            # NOT wait for prefetch_healthy: promoting only converts a streamed layer
            # into a resident one (same demand-load fallback) and REDUCES streaming,
            # so gating it on hit-rate was a deadlock — under multi-resolution
            # shuffling the trace never validates, so the controller streamed
            # everything forever despite free headroom. cadence + "stopped"
            # hysteresis stop thrash; the trace is re-seeded after each promote.
            measured = bucket["steps"] >= stable_windows
            promoting = cadence_ready and measured
            # Worst-resolution guard: a block promoted here stays resident for
            # EVERY resolution bucket, but the deadband above only proved headroom
            # for the CURRENT step. Predict the cohabitation peak on the worst
            # measured resolution (its working reserve is what the plan already
            # applies to all shapes) and veto the promotion if it would not leave
            # the promote floor there -- otherwise a roomy low-res step promotes a
            # block that silently pages the next high-res step. Unmeasured
            # resolutions are handled by demote-on-arrival, not this gate.
            worst_case_veto = False
            predicted_worst_free_gib = None
            promote_blocks = 1
            if promoting:
                promote_block_bytes = cls._next_promotion_layer_bytes(module, mm)
                if promote_block_bytes > 0:
                    block_gib = promote_block_bytes / gib
                    ring_gib = diagnostics.get(
                        "ring_peak_gb", diagnostics.get("planned_ring_gb", 0.0)
                    )
                    resident_gib = diagnostics.get("planned_resident_gb", 0.0)
                    total_gib = diagnostics["device_total_gb"]
                    if eager:
                        promote_blocks = vram_budget.training_eager_promote_blocks(
                            resident_gib=resident_gib,
                            block_gib=block_gib,
                            ring_gib=ring_gib,
                            worst_working_reserve_gib=new_working_reserve,
                            other_gib=other_gib,
                            total_gib=total_gib,
                            promote_floor_gib=promote_floor_gib,
                            max_blocks=eager_max_blocks,
                        )
                        if promote_blocks < 1:
                            promoting = False
                            worst_case_veto = True
                            promote_blocks = 1
                    predicted_worst_free_gib = (
                        vram_budget.training_promotion_worst_shape_free_gib(
                            resident_gib=resident_gib,
                            added_block_gib=block_gib * promote_blocks,
                            ring_gib=ring_gib,
                            worst_working_reserve_gib=new_working_reserve,
                            other_gib=other_gib,
                            total_gib=total_gib,
                        )
                    )
                    if promoting and predicted_worst_free_gib < promote_floor_gib:
                        promoting = False
                        worst_case_veto = True
            if pool is not None and not prefetch_ok and not promoting:
                # Not promoting this step (still measuring, or off-cadence) and the
                # trace is unhealthy: spend the move improving prefetch coverage so
                # the layers we are still streaming hide better.
                max_budget_gib = float(_env("AI_TOOLKIT_BOUNCE_MAX_POOL_GIB", "6.0"))
                grow_gib = float(_env("AI_TOOLKIT_BOUNCE_GROW_GIB", "0.5"))
                grow_floor_gib = float(_env("AI_TOOLKIT_BOUNCE_GROW_FREE_FLOOR_GIB", str(wddm_stop_gib)))
                current_budget = float(getattr(pool, "budget_bytes", 0)) / gib
                current_target = float(getattr(pool, "target_ready_bytes", 0)) / gib
                current_lookahead = int(getattr(pool, "max_lookahead_positions", diagnostics.get("prefetch_lookahead", 16)))
                has_free_slack = min_device_free_gib > grow_floor_gib + grow_gib
                can_grow_budget = current_budget < max_budget_gib
                can_grow_target = current_target < max_budget_gib
                if has_free_slack and (can_grow_budget or can_grow_target):
                    next_budget = min(max_budget_gib, current_budget + grow_gib)
                    next_target = min(next_budget, current_target + grow_gib)
                    next_lookahead = min(
                        int(_env("AI_TOOLKIT_BOUNCE_MAX_LOOKAHEAD", "64")),
                        max(1, current_lookahead) * 2,
                    )
                    pool.tune(
                        budget_bytes=int(next_budget * gib),
                        target_ready_bytes=int(next_target * gib),
                        lookahead=next_lookahead,
                    )
                    grew_prefetch = True
                    layout_action = f"grow_prefetch:{prefetch_reason}"
                else:
                    layout_action = f"prefetch_maxed:{prefetch_reason}"
            if not grew_prefetch and promoting:
                changed_layers, layout_action = cls._promote_training_layer(
                    module,
                    mm,
                    device,
                    cache_pad_gib=cache_pad_gib,
                    wddm_stop_gib=wddm_stop_gib,
                    max_blocks=promote_blocks,
                )
                if changed_layers:
                    state["last_step"] = step_index
                    if bucket.get("no_improve", 0) >= 2:
                        state["stopped"] = True
                elif layout_action in ("no_offloaded_layers", "stop_line"):
                    state["stopped"] = True
            elif grew_prefetch:
                pass
            elif worst_case_veto:
                layout_action = (
                    "worst_shape_hold:"
                    f"{predicted_worst_free_gib:.2f}<{promote_floor_gib:.2f}"
                )
            elif not measured:
                layout_action = f"measuring_working_set:{bucket['steps']}/{stable_windows}"
            else:
                layout_action = "wait_cadence"
        effective_working_reserve = new_working_reserve
        state["current_working_reserve_gib"] = float(effective_working_reserve)
        plan = cls._refresh_training_plan_from_layout(module, mm, effective_working_reserve)
        if changed_layers:
            cls._register_training_prefetch_sources(module, mm)
        result = {
            "enabled": True,
            "bucket_steps": bucket["steps"],
            "action": action,
            "layout_action": layout_action,
            "changed_layers": changed_layers,
            "working_reserve_gb": plan["working_reserve_bytes"] / gib,
            "resident_gb": plan["resident_bytes"] / gib,
            "streamed_layers": plan["offloaded_layers"],
            "min_device_free_gb": min_device_free_gib,
            "estimated_device_free_gb": estimated_free_gib,
            "observed_device_free_gb": (
                diagnostics.get("device_free_peak_gb")
                if diagnostics.get("device_peak_source") == "observed"
                else None
            ),
            "measured_peak_gb": measured_peak_gib,
            "danger_working_reserve_gb": state.get("danger_working_reserve_gib"),
            "learned_wddm_hard_gb": state.get("learned_wddm_hard_gib"),
            "timing_spill": timing_spill,
        }
        state["last_action"] = f"{action}:{layout_action}"
        if cls._diagnostics_enabled() and (
            action != "hold" or changed_layers or did_oom
        ):
            print(
                f"[MemoryManager] training autotune: action={action} "
                f"layout={layout_action} changed_layers={changed_layers} "
                f"reserve_space={result['working_reserve_gb']:.2f} GiB "
                f"resident={result['resident_gb']:.2f} GiB "
                f"streamed_layers={result['streamed_layers']} "
                f"min_free={min_device_free_gib:.2f} GiB "
                f"estimated_free={estimated_free_gib:.2f} GiB "
                f"observed_free={(result['observed_device_free_gb'] if result['observed_device_free_gb'] is not None else float('nan')):.2f} GiB "
                f"peak_working={measured_peak_gib:.2f} GiB "
                f"learned_wddm_hard={state.get('learned_wddm_hard_gib') or 0.0:.2f} GiB"
            )
        return result

    @classmethod
    def _training_cliff_safety_net(
        cls,
        module,
        mm,
        device=None,
        *,
        shape_key=None,
        did_oom=False,
        peak_allocated_override=None,
        peak_reserved_override=None,
    ):
        """Keep manual working_reserve runs off the WDDM spill cliff.

        The guard trims idle allocator cache first, but demotion is decided from
        the next step's expected live peak, not from the artificially healthy
        post-trim trough. Otherwise a stable run can loop forever:
        empty_cache -> lots of free -> same peak allocation returns -> cliff.
        """
        plan = getattr(mm, "_smart_training_plan", None)
        if plan is None:
            return None
        try:
            device = torch.device(device or mm.process_device)
        except (TypeError, ValueError, RuntimeError):
            return None
        if device.type != "cuda" or not torch.cuda.is_available():
            return None

        gib = 1024 ** 3
        hard_gib = max(
            float(_env("AI_TOOLKIT_TRAINING_WDDM_HARD_GIB", "1.0")),
            float(plan.get("wddm_hard_bytes", 0)) / gib,
        )
        retreat_layers = max(1, int(_env("AI_TOOLKIT_TRAINING_RETREAT_LAYERS", "1")))
        max_demote = int(_env("AI_TOOLKIT_TRAINING_SAFETY_MAX_DEMOTE", "12"))
        target_gib = max(
            hard_gib,
            float(_env("AI_TOOLKIT_TRAINING_WDDM_STOP_GIB", str(hard_gib + 0.5))),
        )

        free_bytes, total_bytes = vram_budget.device_mem_info(device)
        before_gib = free_bytes / gib
        needs_reclaim = (
            cls._training_cliff_guard_action(
                before_gib, wddm_hard_gib=hard_gib, did_oom=did_oom
            ) == "reclaim"
        )

        total_gib = total_bytes / gib
        reserved_before_gib = torch.cuda.memory_reserved(device) / gib
        allocated_before_gib = torch.cuda.memory_allocated(device) / gib
        cached_before_gib = max(0.0, reserved_before_gib - allocated_before_gib)
        peak_allocated_gib = (
            peak_allocated_override
            if peak_allocated_override is not None
            else torch.cuda.max_memory_allocated(device)
        ) / gib
        peak_reserved_gib = (
            peak_reserved_override
            if peak_reserved_override is not None
            else torch.cuda.max_memory_reserved(device)
        ) / gib
        cls._record_manual_training_shape_peak(
            mm,
            shape_key,
            peak_allocated_gib=peak_allocated_gib,
            peak_reserved_gib=peak_reserved_gib,
        )
        predicted_peak_free_gib = cls._training_cliff_predicted_peak_free_gib(
            total_gib,
            before_gib,
            reserved_before_gib,
            peak_allocated_gib,
        )
        peak_pressure = did_oom or predicted_peak_free_gib < target_gib
        if not needs_reclaim and not peak_pressure:
            return None

        free_gib = before_gib
        if needs_reclaim:
            # Return idle cached blocks to the driver. This fixes pure allocator
            # hoarding, but demotion is still driven by peak math below.
            try:
                torch.cuda.synchronize(device)
            except RuntimeError:
                pass
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
            free_gib = vram_budget.device_free_bytes(device) / gib

        demoted = 0
        if peak_pressure and max_demote > 0:
            demoted = cls._demote_training_layers(
                module, mm, retreat_layers, largest=True
            )
            if demoted:
                cls._invalidate_manual_training_shape_peaks(mm)
                free_gib = vram_budget.device_free_bytes(device) / gib

        action = "demote" if demoted else ("empty_cache" if needs_reclaim else "peak_pressure_unavailable")
        if cls._diagnostics_enabled():
            print(
                f"[MemoryManager] manual cliff guard: action={action} "
                f"demoted_layers={demoted} "
                f"free={before_gib:.2f}->{free_gib:.2f} GiB "
                f"peak_free={predicted_peak_free_gib:.2f} GiB "
                f"cached_before={cached_before_gib:.2f} GiB "
                f"peak_allocated={peak_allocated_gib:.2f} GiB "
                f"peak_reserved={peak_reserved_gib:.2f} GiB "
                f"(hard_floor={hard_gib:.2f} GiB target={target_gib:.2f} GiB)"
            )
        return {
            "manual_safety": True,
            "action": action,
            "demoted_layers": demoted,
            "device_free_gib": free_gib,
            "device_free_before_gib": before_gib,
            "predicted_peak_free_gib": predicted_peak_free_gib,
            "allocator_cached_before_gib": cached_before_gib,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
        }

    @staticmethod
    def _unwrap_memory_managed_module(module):
        while module is not None and not hasattr(module, "_memory_manager"):
            wrapped = getattr(module, "module", None)
            if wrapped is None or wrapped is module:
                return module
            module = wrapped
        return module

    @staticmethod
    def training_pinned_keys_for_keep_last(module, keep_last):
        """Return block-aligned permanent-resident keys for trailing blocks."""
        keep_last = max(0, int(keep_last or 0))
        blocks = getattr(module, "blocks", None)
        if blocks is None or keep_last <= 0:
            return set()
        total = len(blocks)
        start = max(0, total - keep_last)
        return {f"blocks.{i}" for i in range(start, total)}

    @classmethod
    def set_training_pinned_resident_blocks(cls, module, keep_last):
        """Sync the permanent resident block tier after keep_last changes.

        The live working_reserve controller may move layers between evictable-resident
        and streamed, but it must not demote these block keys. If a newly pinned
        block is currently streamed, promote its layers when the driver-free
        stop-line says there is room; otherwise leave it eager/streamed for now
        and the compile-safe check will reject it.
        """
        root = cls._unwrap_memory_managed_module(module)
        if root is None or not hasattr(root, "_memory_manager"):
            return None
        mm = root._memory_manager
        keys = cls.training_pinned_keys_for_keep_last(root, keep_last)
        mm._training_pinned_resident_keys = set(keys)
        plan = getattr(mm, "_smart_training_plan", None)
        if plan is None:
            return None

        device = torch.device(mm.process_device)
        promoted = 0
        skipped = 0
        if device.type == "cuda" and torch.cuda.is_available() and keys:
            stop_bytes = int(
                float(_env("AI_TOOLKIT_TRAINING_WDDM_STOP_GIB", "1.5"))
                * 1024 ** 3
            )
            args = getattr(mm, "_attach_args", {}) or {}
            candidates = [
                item for item in cls._training_layout_candidates(
                    root, args.get("ignore_modules", []), keys
                )
                if item["managed"] and item.get("pinned_resident")
            ]
            for item in sorted(candidates, key=lambda row: row["resident_bytes"]):
                free_bytes = vram_budget.device_free_bytes(device)
                if free_bytes - item["resident_bytes"] < stop_bytes:
                    skipped += 1
                    continue
                if cls.promote_layer(item["module"]):
                    promoted += 1
        cls._refresh_training_fp8_flags(root, mm)
        plan = cls._refresh_training_plan_from_layout(root, mm)
        if promoted:
            cls._invalidate_manual_training_shape_peaks(mm)
            cls.reset_trace_due_to_execution_shape_change()
            cls._clear_cuda_pipeline_state()
        return {
            "pinned_keys": sorted(keys),
            "promoted_layers": promoted,
            "skipped_layers": skipped,
            "pinned_resident_gb": plan.get("pinned_resident_bytes", 0) / 1024 ** 3,
        }
    @staticmethod
    def _historical_prefetch_defaults(source_count: int):
        """Infer cold-start prefetch defaults from previous local perf logs.

        Logs do not contain the exact layer order, so this only estimates how
        long the first schedule should be and how much pool coverage was useful.
        The bounce pool still learns exact ordering from the live step.
        """
        if source_count <= 0:
            return {}
        if _env("AI_TOOLKIT_BOUNCE_HISTORY", "1").lower() in (
            "0", "false", "no", "off", ""
        ):
            return {}
        output_root = pathlib.Path(
            _env("AI_TOOLKIT_BOUNCE_HISTORY_ROOT", "output")
        )
        if not output_root.exists():
            return {}
        max_files = int(_env("AI_TOOLKIT_BOUNCE_HISTORY_FILES", "24"))
        max_rows_per_file = int(_env("AI_TOOLKIT_BOUNCE_HISTORY_ROWS", "128"))
        tolerance = max(
            8,
            int(source_count * float(_env("AI_TOOLKIT_BOUNCE_HISTORY_TOL", "0.25"))),
        )
        files = sorted(
            output_root.rglob("performance_log.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:max_files]
        fetch_re = re.compile(r"fetches=(\d+) unique_layers=(\d+)")
        bp_num_re = re.compile(r"([a-zA-Z_]+)=([0-9.]+)")
        ratios = []
        healthy_ratios = []
        budgets = []
        targets = []
        lookaheads = []
        for path in files:
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines[-max_rows_per_file:]:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                mm = row.get("smart_training_offload") or row.get("memory_manager") or {}
                profile = row.get("offload_profile") or ""
                prefetch = row.get("offload_prefetch") or ""
                unique = mm.get("unique_layers")
                accesses = None
                match = fetch_re.search(profile)
                if match:
                    accesses = int(match.group(1))
                    unique = int(match.group(2))
                stats = {}
                if prefetch:
                    for key, value in bp_num_re.findall(prefetch):
                        try:
                            stats[key] = float(value)
                        except ValueError:
                            pass
                    accesses = max(accesses or 0, int(stats.get("acquires", 0)) or 0)
                schedule_len = mm.get("prefetch_schedule_len") or stats.get("schedule")
                if schedule_len:
                    accesses = max(accesses or 0, int(schedule_len))
                if not unique or abs(int(unique) - source_count) > tolerance:
                    continue
                if accesses and unique:
                    ratio = float(accesses) / max(1.0, float(unique))
                    ratios.append(ratio)
                    hit_rate = mm.get("bounce_hit_rate")
                    if hit_rate is None and stats.get("acquires"):
                        hit_rate = stats.get("hit", 0.0) / max(1.0, stats["acquires"])
                    if hit_rate is not None and hit_rate >= 0.90:
                        healthy_ratios.append(ratio)
                budget = mm.get("prefetch_pool_budget_gb") or stats.get("budget")
                target = mm.get("prefetch_target_ready_gb") or stats.get("target_ready")
                lookahead = mm.get("prefetch_lookahead") or stats.get("lookahead")
                if budget:
                    budgets.append(float(budget))
                if target:
                    targets.append(float(target))
                if lookahead:
                    lookaheads.append(int(lookahead))
        selected = healthy_ratios or ratios
        if not selected:
            return {}
        def median(values):
            values = sorted(values)
            return values[len(values) // 2]
        ratio = median(selected)
        schedule_len = int(source_count * max(1.0, ratio) + 0.999)
        schedule_len = min(
            int(_env("AI_TOOLKIT_BOUNCE_HISTORY_MAX_SCHEDULE", "4096")),
            max(source_count, schedule_len),
        )
        result = {"schedule_len": schedule_len, "ratio": ratio, "samples": len(selected)}
        if budgets:
            result["budget_gib"] = median(budgets)
        if targets:
            result["target_ready_gib"] = median(targets)
        if lookaheads:
            result["lookahead"] = median(lookaheads)
        return result
    @classmethod
    def _attach_prefetch_pool(cls, module, device):
        """Create a bounce pool and seed it from historical access counts."""
        gib = 1024 ** 3
        sources = []
        for child in module.modules():
            key = getattr(child, "_mm_layer_key", None)
            if key is not None and hasattr(child, "_layer_memory_manager"):
                sources.append((key, child))
        registered = len(sources)
        # Fully-pinned streamed set: every streamed layer's H2D runs async
        # straight from its page-locked weight (the pinned-source bypass in
        # manager_modules), so a bounce pool would never transfer anything --
        # it would only pin buffers out of the same finite WDDM shared budget
        # the weight pins already spent. Don't create one.
        def _layer_pinned(child):
            weight = getattr(child, "weight", None)
            if weight is None or not _profile_is_pinned(weight.data):
                return False
            bias = getattr(child, "bias", None)
            return bias is None or _profile_is_pinned(bias.data)

        if not sources:
            # No legacy per-Linear streamed layers on this module at all (e.g.
            # the immutable/arena backend, whose canonical block transfers go
            # through pin_manager + the device-side fetch ring and MANDATE a
            # pinned host source -- see ingraph_stream._fetch_start_impl,
            # which raises rather than reading a pageable tensor). There is
            # nothing this pool could ever bounce, so it would just burn WDDM
            # shared-budget pinned buffers for no transfers. Don't create one.
            print(
                "[MemoryManager] bounce pool disabled: no legacy-managed "
                "streamed layers on this module (arena/immutable block "
                "streaming requires pinned sources and never uses the pool)"
            )
            return
        if all(_layer_pinned(child) for _, child in sources):
            print(
                "[MemoryManager] bounce pool disabled: all "
                f"{registered} streamed layers are pinned (pinned-source "
                "bypass makes the pool redundant; its buffers would compete "
                "for the same WDDM shared pinned budget)"
            )
            return
        history = cls._historical_prefetch_defaults(registered)

        default_budget_gib, default_target_gib, default_mode = cls._training_bounce_pool_budget_defaults(
            module,
            block_stream_only=bool(getattr(getattr(module, "_memory_manager", None), "_training_block_stream_only", False)),
            sources=sources,
            history=history,
        )
        env_budget = _env("AI_TOOLKIT_BOUNCE_POOL_GIB", None)
        auto_budget = env_budget is None
        budget_gib = float(env_budget) if env_budget is not None else default_budget_gib
        max_budget_gib = float(_env("AI_TOOLKIT_BOUNCE_MAX_POOL_GIB", "6.0"))
        budget_gib = min(max_budget_gib, budget_gib)
        emergency_floor_gib = min(
            max_budget_gib,
            max(0.0, float(_env("AI_TOOLKIT_BOUNCE_EMERGENCY_FLOOR_GIB", "0.25"))),
        ) if auto_budget else 0.0
        pin_plan = getattr(getattr(module, "_memory_manager", None), "_pin_plan", None)
        if pin_plan is not None:
            planned_budget_gib = int(pin_plan.get("bounce_budget_bytes", 0) or 0) / gib
            if planned_budget_gib < budget_gib:
                print(
                    "[MemoryManager] bounce-pool budget from pin manager: "
                    f"want={budget_gib:.2f} GiB -> {planned_budget_gib:.2f} GiB "
                    f"strategy={pin_plan.get('strategy')}"
                )
            budget_gib = min(budget_gib, planned_budget_gib)
        else:
            ledger_headroom = bounce_pool.pinned_bytes_headroom(
                bounce_pool._cuda_device_index(device)
            )
            if ledger_headroom is not None:
                budget_gib = min(budget_gib, ledger_headroom / gib)
        if budget_gib <= 0:
            print(
                "[MemoryManager] bounce pool disabled by pin manager: "
                f"strategy={pin_plan.get('strategy') if pin_plan else 'zero_budget'}"
            )
            return
        env_target = _env("AI_TOOLKIT_BOUNCE_TARGET_READY_GIB", None)
        target_ready_gib = float(env_target) if env_target is not None else default_target_gib
        target_ready_gib = min(budget_gib, target_ready_gib)
        lookahead = int(_env(
            "AI_TOOLKIT_BOUNCE_LOOKAHEAD",
            str(history.get("lookahead", 32)),
        ))
        lookahead = max(1, min(
            int(_env("AI_TOOLKIT_BOUNCE_MAX_LOOKAHEAD", "64")),
            lookahead,
        ))
        budget = int(budget_gib * gib)
        target_ready = int(target_ready_gib * gib)
        workers = int(_env("AI_TOOLKIT_BOUNCE_WORKERS", "2"))
        ram_floor = int(
            float(_env("AI_TOOLKIT_BOUNCE_RAM_FLOOR_GIB", "2.0")) * gib
        )
        # Block streaming: batch each worker fill over a whole block's worth of
        # Linears so the per-Linear lock/CV/slot overhead is paid once per block
        # instead of once per Linear. Derive the group size from the largest
        # streamed block (in block_stream_only mode every streamed source is a
        # block layer); an explicit env override always wins.
        mm = getattr(module, "_memory_manager", None)
        fill_group_size = None
        if getattr(mm, "_training_block_stream_only", False):
            group_counts: dict = {}
            for key, _child in sources:
                gk = cls._offload_group_key(key)
                group_counts[gk] = group_counts.get(gk, 0) + 1
            block_parents = cls._streaming_block_parents(group_counts.keys())
            block_counts = [
                n for gk, n in group_counts.items()
                if cls._block_parent_of(gk) in block_parents
            ]
            if block_counts:
                fill_group_size = max(block_counts)
        env_group = _env("AI_TOOLKIT_BOUNCE_FILL_GROUP", "").strip()
        if env_group:
            fill_group_size = max(1, int(env_group))
        pool = bounce_pool.create_pool(
            device,
            budget_bytes=budget,
            lookahead=lookahead,
            target_ready_bytes=target_ready,
            num_workers=workers,
            ram_floor_bytes=ram_floor,
            fill_group_size=fill_group_size,
        )
        cold_start_schedule = []
        for key, child in sources:
            pool.register_source(key, child)
            cold_start_schedule.append(key)
        target_schedule_len = int(history.get("schedule_len", len(cold_start_schedule)))
        if cold_start_schedule and target_schedule_len > len(cold_start_schedule):
            repeats = (target_schedule_len + len(cold_start_schedule) - 1) // len(cold_start_schedule)
            cold_start_schedule = (cold_start_schedule * repeats)[:target_schedule_len]
        if cold_start_schedule:
            pool.set_schedule(cold_start_schedule)
        module._memory_manager._prefetch_pool = pool
        history_text = ""
        if history:
            history_text = (
                f" history_ratio={history.get('ratio', 0.0):.2f}"
                f" history_samples={history.get('samples', 0)}"
            )
        print(
            f"[MemoryManager] bounce pool attached: device={device} "
            f"budget={budget / gib:.2f} GiB lookahead={lookahead} "
            f"target_ready={target_ready / gib:.2f} GiB "
            f"workers={workers} fill_group={pool.fill_group_size} sources={registered} "
            f"default_mode={default_mode} "
            f"cold_start_schedule={len(cold_start_schedule)}{history_text}"
        )

    @classmethod
    def training_runtime_diagnostics(
        cls,
        module,
        device=None,
        peak_allocated_override=None,
        peak_reserved_override=None,
        observed_driver_free_min_bytes=None,
        observed_driver_total_bytes=None,
        observed_driver_free_samples=None,
    ):
        """Snapshot a smart training layout and its actual runtime memory.

        ``peak_allocated_override`` / ``peak_reserved_override`` (bytes) let the
        caller supply the true within-step allocator high-water aggregated across
        gradient accumulations. The live CUDA peak counter is reset per
        accumulation by the trainer's resolution sampler, so on multi-accumulation
        steps it under-reports; pass the step-aggregated peak so the controller
        governs on the real high-water.

        ``observed_driver_free_min_bytes`` is a sampled driver-level minimum from
        the training window. When present, it is the source of the reported
        device peak; the allocator-derived value is kept separately as *_est.
        """
        while module is not None and not hasattr(module, "_memory_manager"):
            wrapped = getattr(module, "module", None)
            if wrapped is None or wrapped is module:
                return None
            module = wrapped
        if module is None:
            return None
        mm = module._memory_manager
        plan = getattr(mm, "_smart_training_plan", None)
        if plan is None:
            return None
        device = torch.device(device or mm.process_device)
        memory = cls._cuda_memory(device)
        if memory is None:
            return None

        state = _DEVICE_STATE.get(device, {})
        # Allocator GC health: retries = OOM-retry reclaims (must stay ~0 in
        # steady state), device alloc/free counts = cudaMalloc/cudaFree churn
        # from cap- or gc_threshold-triggered sweeps.
        try:
            alloc_stats = torch.cuda.memory_stats(device)
        except Exception:
            alloc_stats = {}
        gc_now = {
            key: int(alloc_stats.get(key, 0) or 0)
            for key in ("num_alloc_retries", "num_device_alloc", "num_device_free")
        }
        gc_deltas = cls._gc_counter_deltas(state.get("gc_counters_prev"), gc_now)
        _DEVICE_STATE.setdefault(device, {})["gc_counters_prev"] = gc_now
        ring_live_bytes = int(state.get("ring_live_bytes", 0) or 0)
        if ring_live_bytes <= 0:
            seen = set()
            for key in ("w_buffers", "b_buffers", "w_grad_buffers", "b_grad_buffers"):
                for tensor in state.get(key, ()):
                    if tensor is None or id(tensor) in seen:
                        continue
                    seen.add(id(tensor))
                    ring_live_bytes += cls._tensor_storage_bytes(tensor)
        ring_peak_bytes = max(
            ring_live_bytes,
            int(state.get("ring_peak_bytes", 0) or 0),
        )

        allocated_bytes = int(memory[0] * 1024 ** 3)
        reserved_bytes = int(memory[1] * 1024 ** 3)
        # Working set has two very different readings:
        #   - residual: current allocation MINUS resident/ring at the diagnostics
        #     snapshot (step end). The backward graph is already freed here, so it
        #     reads the trough — misleadingly small (it is NOT "barely any of the
        #     budget was needed").
        #   - peak: the within-step high-water (max_memory_allocated, reset every
        #     step) MINUS resident/ring. This is the activation/dequant footprint
        #     that actually has to fit under the reserve.
        # Report both; the peak is the one that matters for sizing the budget.
        peak_allocated_bytes = int(
            peak_allocated_override
            if peak_allocated_override is not None
            else torch.cuda.max_memory_allocated(device)
        )
        working_residual_bytes = max(
            0, allocated_bytes - plan["resident_bytes"] - ring_live_bytes
        )
        working_peak_bytes = max(
            0, peak_allocated_bytes - plan["resident_bytes"] - ring_peak_bytes
        )
        # Kept under the original name for back-compat: the live auto-controller's
        # EMA reads ``working_reserve_used_gb``; changing its meaning would alter
        # tuning dynamics, so it stays the residual. New code/logs use the peak.
        working_bytes = working_residual_bytes
        pool = getattr(mm, "_prefetch_pool", None)
        pool_stats = None
        if pool is not None:
            try:
                pool_stats = pool.stats(reset=False)
            except Exception:
                pool_stats = None
        cpu_bounce_budget_gb = (
            pool_stats.get("budget_gib") if pool_stats is not None else 0.0
        )
        bounce_hard_miss = pool_stats.get("hard_misses", 0) if pool_stats else 0
        bounce_hit_rate = pool_stats.get("hit_rate", 0.0) if pool_stats else 0.0
        bounce_acquires = pool_stats.get("acquires", 0) if pool_stats else 0
        bounce_soft_miss = pool_stats.get("soft_misses", 0) if pool_stats else 0
        bounce_resyncs = pool_stats.get("resyncs", 0) if pool_stats else 0
        bounce_mismatches = pool_stats.get("mismatches", 0) if pool_stats else 0
        duplicate_key_blocked = pool_stats.get("duplicate_key_resync_blocked", 0) if pool_stats else 0
        hard_miss_rate = bounce_hard_miss / max(1, bounce_acquires)
        soft_miss_rate = bounce_soft_miss / max(1, bounce_acquires)
        resync_rate = bounce_resyncs / max(1, bounce_acquires)
        mismatch_rate = bounce_mismatches / max(1, bounce_acquires)
        duplicate_key_block_rate = duplicate_key_blocked / max(1, bounce_acquires)
        schedule_len = pool_stats.get("schedule_len", 0) if pool_stats else 0
        consume_pos = pool_stats.get("consume_pos", 0) if pool_stats else 0
        lookahead = pool_stats.get("lookahead", 0) if pool_stats else 0
        schedule_confidence = pool_stats.get("schedule_confidence", "cold") if pool_stats else "none"
        target_ready_gb = pool_stats.get("target_ready_gib", 0.0) if pool_stats else 0.0
        cpu_wait_s = pool_stats.get("cpu_wait_s", 0.0) if pool_stats else 0.0
        bounce_fills = pool_stats.get("fills", 0) if pool_stats else 0
        bounce_fill_batches = pool_stats.get("fill_batches", 0) if pool_stats else 0
        bounce_fill_group_size = pool_stats.get("fill_group_size", 1) if pool_stats else 1
        bounce_fills_per_batch = pool_stats.get("fills_per_batch", 0.0) if pool_stats else 0.0
        bounce_copy_s = pool_stats.get("copy_s", 0.0) if pool_stats else 0.0
        bounce_copy_gbps = pool_stats.get("copy_gbps", 0.0) if pool_stats else 0.0
        missing_schedule = bool(
            pool_stats is not None
            and schedule_len == 0
            and bounce_acquires > 0
            and hard_miss_rate > 0.25
        )
        invalid_hard_miss_rate = float(
            _env("AI_TOOLKIT_BOUNCE_INVALID_HARD_MISS_RATE", "0.25")
        )
        invalid_mismatch_rate = float(
            _env("AI_TOOLKIT_BOUNCE_INVALID_MISMATCH_RATE", "0.10")
        )
        invalid_trace = bool(
            pool_stats is not None
            and cls._prefetch_trace_invalid(
                schedule_len=schedule_len,
                consume_pos=consume_pos,
                lookahead=lookahead,
                hard_miss_rate=hard_miss_rate,
                mismatch_rate=mismatch_rate,
                duplicate_key_block_rate=duplicate_key_block_rate,
                hard_miss_threshold=invalid_hard_miss_rate,
                mismatch_threshold=invalid_mismatch_rate,
            )
        )
        capacity_limited = bool(
            pool_stats is not None
            and not invalid_trace
            and (soft_miss_rate > 0.01 or cpu_wait_s > 0.05)
        )
        budget_limited = bool(
            pool_stats is not None
            and not invalid_trace
            and (cpu_bounce_budget_gb <= 0.0 or target_ready_gb <= 0.0)
        )
        if pool_stats is None:
            prefetch_reason = "no_pool"
        elif missing_schedule:
            prefetch_reason = "missing_schedule"
        elif invalid_trace:
            prefetch_reason = "invalid_trace"
        elif budget_limited:
            prefetch_reason = "budget_limited"
        elif capacity_limited:
            prefetch_reason = "capacity_limited"
        elif hard_miss_rate > 0.05:
            prefetch_reason = "hard_miss"
        elif bounce_hit_rate < 0.90:
            prefetch_reason = "low_hit_rate"
        else:
            prefetch_reason = "healthy"
        healthy_hit_rate = float(
            _env("AI_TOOLKIT_BOUNCE_HEALTHY_HIT_RATE", "0.95")
        )
        healthy_hard_miss_rate = float(
            _env("AI_TOOLKIT_BOUNCE_HEALTHY_HARD_MISS_RATE", "0.01")
        )
        unhealthy_hit_rate = float(
            _env("AI_TOOLKIT_BOUNCE_UNHEALTHY_HIT_RATE", "0.90")
        )
        unhealthy_hard_miss_rate = float(
            _env("AI_TOOLKIT_BOUNCE_UNHEALTHY_HARD_MISS_RATE", "0.05")
        )
        prefetch_healthy = bool(
            pool_stats is not None
            and bounce_hit_rate >= healthy_hit_rate
            and hard_miss_rate <= healthy_hard_miss_rate
        )
        prefetch_unhealthy = bool(
            pool_stats is not None
            and (
                bounce_hit_rate < unhealthy_hit_rate
                or hard_miss_rate > unhealthy_hard_miss_rate
            )
        )
        autotune_state = getattr(mm, "_training_autotune_state", {}) or {}
        # --- Device peak reporting ------------------------------------------
        # memory[2]/[3] are step-end trough values. The allocator-derived peak is
        # only an estimate because WDDM/driver/other usage may also move during a
        # step. Prefer the trainer's sampled driver-free minimum when available.
        peak_reserved_gb = (
            peak_reserved_override
            if peak_reserved_override is not None
            else torch.cuda.max_memory_reserved(device)
        ) / 1024 ** 3
        device_other_gb = max(0.0, memory[2] - memory[1])
        device_used_peak_est_gb = peak_reserved_gb + device_other_gb
        device_free_peak_est_gb = max(0.0, memory[4] - device_used_peak_est_gb)
        device_used_peak_gb = device_used_peak_est_gb
        device_free_peak_gb = device_free_peak_est_gb
        device_peak_source = "estimate"
        driver_free_samples = observed_driver_free_samples
        if observed_driver_free_min_bytes is not None:
            try:
                observed_free_gb = max(0.0, int(observed_driver_free_min_bytes) / 1024 ** 3)
                observed_total_gb = (
                    int(observed_driver_total_bytes) / 1024 ** 3
                    if observed_driver_total_bytes is not None
                    else memory[4]
                )
                device_free_peak_gb = min(observed_free_gb, observed_total_gb)
                device_used_peak_gb = max(0.0, observed_total_gb - device_free_peak_gb)
                device_peak_source = "observed"
            except (TypeError, ValueError, OverflowError):
                driver_free_samples = None
        dxgi_fields = _dxgi_telemetry(
            bounce_pool._cuda_device_index(device),
            min_interval_s=0.5,
        )
        process_fields = _process_memory_telemetry()
        return {
            "strategy": "smart",
            "managed_layers": sum(
                1 for child in module.modules()
                if hasattr(child, "_layer_memory_manager")
            ),
            "candidate_layers": plan["candidate_layers"],
            "model_gb": plan["model_bytes"] / 1024 ** 3,
            "planned_resident_gb": plan["resident_bytes"] / 1024 ** 3,
            "must_resident_gb": plan.get("must_resident_bytes", 0) / 1024 ** 3,
            "pinned_resident_gb": plan.get("pinned_resident_bytes", 0) / 1024 ** 3,
            "pinned_resident_keys": sorted(plan.get("pinned_resident_keys", [])),
            "training_compile_ready_blocks": (
                (plan.get("training_compile_readiness") or {}).get("ready_blocks")
            ),
            "training_compile_blocked_blocks": (
                (plan.get("training_compile_readiness") or {}).get("blocked_blocks")
            ),
            "generic_resident_gb": plan.get("generic_resident_bytes", 0) / 1024 ** 3,
            "offloaded_cpu_gb": (
                plan["model_bytes"] - plan["resident_bytes"]
            ) / 1024 ** 3,
            "planned_ring_gb": plan["ring_bytes"] / 1024 ** 3,
            "gpu_stream_budget_gb": plan.get(
                "gpu_stream_budget_bytes", plan["ring_bytes"]
            ) / 1024 ** 3,
            "gpu_stream_need_gb": plan.get(
                "gpu_stream_need_bytes", plan["ring_bytes"]
            ) / 1024 ** 3,
            "cpu_bounce_budget_gb": cpu_bounce_budget_gb,
            "prefetch_pool_budget_gb": (pool_stats or {}).get("budget_gib", 0.0),
            "wddm_margin_gb": plan.get("wddm_margin_bytes", 0) / 1024 ** 3,
            "wddm_hard_gb": plan.get("wddm_hard_bytes", 0) / 1024 ** 3,
            "resident_growth_allowed": plan.get("resident_growth_allowed", False),
            "resident_growth_blocked_reason": plan.get(
                "resident_growth_blocked_reason"
            ),
            "bounce_fills": bounce_fills,
            "bounce_fill_batches": bounce_fill_batches,
            "bounce_fill_group_size": bounce_fill_group_size,
            "bounce_fills_per_batch": bounce_fills_per_batch,
            "bounce_copy_s": bounce_copy_s,
            "bounce_copy_gbps": bounce_copy_gbps,
            "bounce_hard_miss": bounce_hard_miss,
            "bounce_soft_miss": bounce_soft_miss,
            "bounce_resyncs": bounce_resyncs,
            "bounce_mismatches": bounce_mismatches,
            "bounce_duplicate_key_resync_blocked": duplicate_key_blocked,
            "bounce_hit_rate": bounce_hit_rate,
            "bounce_hard_miss_rate": hard_miss_rate,
            "bounce_soft_miss_rate": soft_miss_rate,
            "bounce_resync_rate": resync_rate,
            "bounce_mismatch_rate": mismatch_rate,
            "bounce_duplicate_key_resync_block_rate": duplicate_key_block_rate,
            "prefetch_reason": prefetch_reason,
            "prefetch_missing_schedule": missing_schedule,
            "prefetch_invalid_trace": invalid_trace,
            "prefetch_invalid_hard_miss_threshold": invalid_hard_miss_rate,
            "prefetch_invalid_mismatch_threshold": invalid_mismatch_rate,
            "prefetch_capacity_limited": capacity_limited,
            "prefetch_budget_limited": budget_limited,
            "prefetch_schedule_len": schedule_len,
            "prefetch_schedule_confidence": schedule_confidence,
            "prefetch_consume_pos": consume_pos,
            "prefetch_lookahead": lookahead,
            "prefetch_target_ready_gb": target_ready_gb,
            "prefetch_healthy": prefetch_healthy,
            "prefetch_unhealthy": prefetch_unhealthy,
            "live_ring_gb": ring_live_bytes / 1024 ** 3,
            "ring_peak_gb": ring_peak_bytes / 1024 ** 3,
            "pinned_cpu_gb": mm.pinned_weight_bytes / 1024 ** 3,
            "pinned_ledger_total_gb": bounce_pool._pinned_bytes_total / 1024 ** 3,
            **dxgi_fields,
            **process_fields,
            "training_working_reserve_gb": plan["working_reserve_bytes"] / 1024 ** 3,
            # Peak within-step working set — the truthful "how much of the reserve
            # did we actually need" number. Use this when reading logs.
            "working_reserve_peak_gb": working_peak_bytes / 1024 ** 3,
            # Step-end residual (trough); kept for back-compat / controller EMA.
            "working_reserve_used_gb": working_bytes / 1024 ** 3,
            "working_reserve_residual_gb": working_residual_bytes / 1024 ** 3,
            # Spare budget measured against the PEAK, not the trough, so it no
            # longer overstates how much reserve is sitting idle.
            "working_reserve_remaining_gb": (
                plan["working_reserve_bytes"] - working_peak_bytes
            ) / 1024 ** 3,
            "torch_allocated_gb": memory[0],
            "torch_reserved_gb": memory[1],
            "allocator_cached_gb": max(0.0, memory[1] - memory[0]),
            # TROUGH (step-end): looks generous because the activation peak has
            # already been freed. For "is the cliff close?" read *_peak below.
            "device_used_gb": memory[2],
            "device_free_gb": memory[3],
            "device_total_gb": memory[4],
            # Non-allocator residents (CUDA ctx, cuDNN, WDDM/desktop, other apps).
            "device_other_gb": device_other_gb,
            # PEAK (within-step): observed driver-level peak when sampled;
            # otherwise the allocator-derived estimate. The estimate is retained
            # separately so logs do not silently present it as fact.
            "device_used_peak_gb": device_used_peak_gb,
            "device_free_peak_gb": device_free_peak_gb,
            "device_peak_source": device_peak_source,
            "driver_free_samples": driver_free_samples,
            "device_used_peak_est_gb": device_used_peak_est_gb,
            "device_free_peak_est_gb": device_free_peak_est_gb,
            "peak_allocated_gb": (
                torch.cuda.max_memory_allocated(device) / 1024 ** 3
            ),
            "fp8_training_forward_layers": getattr(
                mm, "_fp8_training_layers", 0
            ),
            "autotune_enabled": bool(
                getattr(mm, "_training_autotune_enabled", False)
            ),
            "autotune_last_action": autotune_state.get("last_action"),
            "autotune_working_reserve_gb": autotune_state.get("current_working_reserve_gib"),
            "autotune_danger_working_reserve_gb": autotune_state.get(
                "danger_working_reserve_gib"
            ),
            "peak_reserved_gb": peak_reserved_gb,
            # Named *_count_* to avoid confusion with device_free_gb (VRAM).
            "alloc_retries_delta": gc_deltas["num_alloc_retries"],
            "alloc_retries_total": gc_now["num_alloc_retries"],
            "cuda_malloc_count_delta": gc_deltas["num_device_alloc"],
            "cuda_free_count_delta": gc_deltas["num_device_free"],
        }

    @staticmethod
    def _clear_cuda_pipeline_state():
        keys = [
            device for device in _DEVICE_STATE
            if isinstance(device, torch.device) and device.type == "cuda"
        ]
        for device in keys:
            try:
                torch.cuda.synchronize(device)
            except RuntimeError:
                # An asynchronous allocation failure may still be surfaced by
                # synchronize. The state must be discarded regardless.
                pass
            del _DEVICE_STATE[device]
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass

    @staticmethod
    def recover_cuda_pipeline_after_oom():
        """Drop persistent streaming slots and allocator cache after an OOM.

        The ring deliberately retains its largest BF16 buffers during normal
        execution. After an OOM that retained capacity and fragmented cache
        only make the retry less likely to recover, so rebuild it from empty.
        """
        MemoryManager._clear_cuda_pipeline_state()
        for pool in bounce_pool.all_pools():
            pool.abort_step()

    @staticmethod
    def reset_job_runtime():
        """Clear process-global offload state before configuring another job.

        ai-toolkit jobs normally run sequentially within a process. Explicitly
        tearing down pools, rings, traces, and feature flags prevents one job's
        experimental settings from leaking into the next one.
        """
        bounce_pool.destroy_all_pools()
        MemoryManager._clear_cuda_pipeline_state()
        set_offload_profile_enabled(False, reset=True)
        set_offload_trace_enabled(False)
        MemoryManager.set_offload_prefetch_enabled(False)
        set_fp8_grad_input_enabled(False)

    @staticmethod
    def offload_profile_report(reset: bool = False):
        """Return the slice-1 streamed-step timing report, or None if disabled."""
        return summarize_offload_profile(reset=reset)

    @staticmethod
    def ingraph_fetch_report(reset: bool = False, *, step_wall_ms=None):
        """Return in-graph streaming fetch stats, or None if inactive."""
        return ingraph_fetch_report(reset=reset, step_wall_ms=step_wall_ms)

    @staticmethod
    def offload_step_begin(shape_key=None):
        """Mark the start of one streamed training step for the trace recorder,
        and hand the frozen access order to each bounce pool so its workers can
        pre-pin the upcoming layers."""
        offload_step_begin(shape_key=shape_key)
        pools = bounce_pool.all_pools()
        if pools:
            schedule = offload_trace_schedule(shape_key=shape_key)
            confidence = offload_trace_schedule_confidence(shape_key=shape_key)
            version = offload_trace_version()
            warmup_bytes = int(
                float(_env("AI_TOOLKIT_BOUNCE_WARMUP_GIB", "0.375"))
                * 1024 ** 3
            )
            warmup_timeout_s = float(
                _env("AI_TOOLKIT_BOUNCE_WARMUP_TIMEOUT_S", "0.02")
            )
            for pool in pools:
                # Refresh when the trace re-records a different (steady) shape,
                # not just on the first freeze.
                if schedule is not None and (
                    pool.schedule_version != version
                    or getattr(pool, "schedule_shape_key", None) != shape_key
                ):
                    pool.set_schedule(
                        schedule,
                        confidence=confidence,
                        filter_to_sources=True,
                    )
                    pool.schedule_version = version
                    pool.schedule_shape_key = shape_key
                elif schedule is None and (
                    getattr(pool, "schedule_shape_key", None) != "observed"
                ):
                    # No frozen trace for this shape yet. Keep workers useful with
                    # registration order until the recorder freezes the real order.
                    # Once the pool has self-promoted an observed access order,
                    # leave it alone: re-seeding here would wipe _observed_step and
                    # reset schedule_shape_key every step, so step_begin's promotion
                    # could never take effect (the train path never engages prefetch).
                    if hasattr(pool, "seed_schedule_from_sources"):
                        pool.seed_schedule_from_sources()
                    else:
                        pool.set_schedule([])
                    pool.schedule_version = -1
                    pool.schedule_shape_key = shape_key
                pool.step_begin(
                    warmup_bytes=warmup_bytes,
                    warmup_timeout_s=warmup_timeout_s,
                )

    @staticmethod
    def offload_step_end():
        """Mark the end of one streamed training step (freeze/replay/validate)."""
        offload_step_end()

    @staticmethod
    def offload_step_abort():
        """Discard the in-flight streamed step's trace (e.g. on OOM)."""
        offload_step_abort()

    @staticmethod
    def reset_trace_due_to_execution_shape_change():
        """Refresh derived transfer plans after a layout/residency change."""
        mark_transfer_plan_dirty()
        for pool in bounce_pool.all_pools():
            pool.abort_step()

    @staticmethod
    def reset_offload_trace_for_tuning():
        """Invalidate trace/prefetch after selective-checkpoint policy changes."""
        invalidate_execution_trace()
        for pool in bounce_pool.all_pools():
            pool.set_schedule([])
            pool.schedule_version = -1
            pool.schedule_shape_key = None
            pool.schedule_confidence = "cold"
            pool.abort_step()

    @staticmethod
    def update_memory_budget_only(
        cpu_bounce_budget_bytes=None,
        cpu_bounce_budget_gib=None,
    ):
        """Resize memory budgets without clearing trace schedules or slots."""
        if cpu_bounce_budget_bytes is None and cpu_bounce_budget_gib is None:
            return
        if cpu_bounce_budget_bytes is None:
            cpu_bounce_budget_bytes = int(float(cpu_bounce_budget_gib) * 1024 ** 3)
        for pool in bounce_pool.all_pools():
            pool.set_budget(cpu_bounce_budget_bytes)

    @staticmethod
    def offload_trace_report():
        """Return a summary of the frozen execution trace, or None if disabled."""
        return offload_trace_report()

    @staticmethod
    def set_offload_profile_enabled(enabled: bool):
        set_offload_profile_enabled(enabled, reset=True)

    @staticmethod
    def set_offload_trace_enabled(enabled: bool):
        set_offload_trace_enabled(enabled)

    @staticmethod
    def set_offload_prefetch_enabled(enabled: bool):
        global _OFFLOAD_PREFETCH_ENABLED
        _OFFLOAD_PREFETCH_ENABLED = bool(enabled)

    @staticmethod
    def set_offload_prefetch_trace_capture(path=None, steps=None):
        bounce_pool.configure_trace_capture(path, steps)

    @staticmethod
    def set_fp8_grad_input_enabled(enabled: bool):
        set_fp8_grad_input_enabled(enabled)

    @staticmethod
    def offload_prefetch_report(reset: bool = False):
        """Concatenate per-device bounce-pool stats, or None if no pool exists."""
        reports = [pool.report(reset=reset) for pool in bounce_pool.all_pools()]
        return "\n".join(reports) if reports else None

    @staticmethod
    def offload_shape_key_from_batch(batch_list, **flags):
        """Build a stable trace key from batch tensor shapes and policy flags."""
        shapes = []

        def visit(value):
            if torch.is_tensor(value):
                shape = tuple(int(dim) for dim in value.shape)
                if len(shape) >= 2:
                    shapes.append((str(value.dtype), shape))
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
                return
            for name in ("tensor", "latents", "images", "control_tensor"):
                if hasattr(value, name):
                    visit(getattr(value, name))

        visit(batch_list)
        shape_key = tuple(sorted(set(shapes)))[:16]
        policy_key = tuple(sorted((key, value) for key, value in flags.items()))
        return (shape_key, policy_key)

    @staticmethod
    def _diagnostics_enabled():
        value = _env("AI_TOOLKIT_MEMORY_DIAGNOSTICS", "1").lower()
        return value not in ("0", "false", "no", "off")

    @staticmethod
    def _cuda_memory(device):
        if device is None:
            return None
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        gib = 1024 ** 3
        free_bytes, total_bytes = vram_budget.device_mem_info(device)
        try:
            driver_used = torch.cuda.device_memory_used(device)
        except Exception:
            # free_bytes is NVML-backed, so this includes every non-PyTorch
            # user on the card (other processes included).
            driver_used = total_bytes - free_bytes
        return (
            torch.cuda.memory_allocated(device) / gib,
            torch.cuda.memory_reserved(device) / gib,
            driver_used / gib,
            free_bytes / gib,
            total_bytes / gib,
        )

    @staticmethod
    def _format_cuda_memory(memory):
        if memory is None:
            return "non-CUDA"
        return (
            f"torch_allocated={memory[0]:.2f} GiB "
            f"torch_reserved={memory[1]:.2f} GiB "
            f"device_used={memory[2]:.2f}/{memory[4]:.2f} GiB "
            f"device_free={memory[3]:.2f} GiB"
        )

    @staticmethod
    def _resolve_sampling_working_reserve(
        working_reserve_gib,
        learned_bytes,
        *,
        cold_start_bytes,
        floor_bytes,
        pad_bytes,
    ):
        """Pick the sampling VRAM reserve and its source (pure, CPU-testable).

        Sampling working_reserve is independent of training:
          * ``working_reserve_gib`` None / < 0 / "auto" -> auto: learn the real working
            set and converge the reserve down to ``learned + pad`` (never below
            ``floor``); before any measurement use the cold-start reserve.
          * ``working_reserve_gib`` >= 0 -> a fixed pinned reserve in GiB. No learning
            is consulted; the post-move spill guard still validates it.

        Returns ``(working_reserve_bytes, source_label)``.
        """
        auto = working_reserve_gib is None
        if not auto:
            try:
                auto = float(working_reserve_gib) < 0
            except (TypeError, ValueError):
                auto = str(working_reserve_gib).lower() == "auto"
        if not auto:
            return int(float(working_reserve_gib) * 1024 ** 3), "fixed-config"
        if int(learned_bytes) > 0:
            return max(int(floor_bytes), int(learned_bytes) + int(pad_bytes)), "measured"
        return int(cold_start_bytes), "cold-start"

    # Pure sampling-cliff predicates/predictors live in vram_budget.
    _sampling_guard_predicted_peak_free = staticmethod(
        vram_budget.sampling_guard_predicted_peak_free
    )
    _sampling_step_should_trim = staticmethod(vram_budget.sampling_step_should_trim)
    _sampling_step_should_demote = staticmethod(vram_budget.sampling_step_should_demote)

    @classmethod
    def _log_demotion_argument(cls, trigger, device, **fields):
        """Print the FULL argument for a demotion decision (diagnostics only).

        Every path that gives residency back must state its complete case --
        trigger, the live device snapshot (allocated/reserved/used/free/
        non_torch), the current allocator cap, and the trigger's own
        thresholds -- so a bad demotion (stale cap, leaked reference, wrong
        threshold) is attributable from the log alone instead of needing a
        repro run. ``fields`` are trigger-specific, pre-formatted values.
        """
        if not cls._diagnostics_enabled():
            return
        parts = []
        snap = vram_budget.DeviceSnapshot.capture(device)
        if snap is not None:
            parts.append(snap.format())
            try:
                dev = torch.device(device)
                index = (
                    dev.index if dev.index is not None
                    else torch.cuda.current_device()
                )
                fraction = cls._wddm_hard_cap_applied.get(index)
                if fraction is not None:
                    parts.append(
                        f"allocator_cap={fraction * snap.total / 1024 ** 3:.2f} GiB"
                    )
            except Exception:
                pass
        parts.extend(f"{key}={value}" for key, value in fields.items())
        print(
            f"[MemoryManager] demotion argument ({trigger}): " + " ".join(parts)
        )

    @classmethod
    def _sampling_demote_largest_block(
        cls, module, plan, target, *, ignore_modules=None, fp8_restores=None
    ):
        """Stream the largest still-resident sampling block; refresh the plan.

        Returns bytes freed (0 if nothing demotable remains). Shared by the
        setup-time spill-guard and the live per-image cohabitation guard so both
        stay consistent: demote whole blocks largest-first, then re-derive the
        plan's resident/offload accounting from the new layout.

        ``fp8_restores`` is the live fp8-sampling restore list: resident FP8
        forwards pin the GPU weights via closure constants, so those closures
        must be dropped for the demoted layers or the demote frees nothing.
        """
        mm = getattr(module, "_memory_manager", None)
        if mm is None:
            return 0
        ignored_ids = {id(m) for m in (ignore_modules or [])}
        resident_blocks: dict = {}
        for name, child in module.named_modules():
            if (
                id(child) in ignored_ids
                or hasattr(child, "_layer_memory_manager")
                or (
                    child.__class__.__name__ not in LINEAR_MODULES
                    and child.__class__.__name__ not in CONV_MODULES
                )
            ):
                continue
            # Only layers whose weights actually occupy the DEVICE are demote
            # candidates. Pack-source linears carry no streaming hook, so they
            # look resident, but their weights live in CPU host packs --
            # "demoting" one frees zero device bytes (observed: futile demote
            # loop at 2000px picking pack blocks, 0.81 GiB plan / 0.00 gained).
            weight = getattr(child, "weight", None)
            if weight is None or weight.device.type == "cpu":
                continue
            key = cls._offload_group_key(name)
            entry = resident_blocks.setdefault(key, {"layers": [], "bytes": 0})
            entry["layers"].append((name, child))
            entry["bytes"] += cls._direct_module_bytes(child)
        if not resident_blocks:
            return 0
        key = max(resident_blocks, key=lambda k: resident_blocks[k]["bytes"])
        freed = resident_blocks[key]["bytes"]
        demoted_ids = {id(child) for _, child in resident_blocks[key]["layers"]}
        if fp8_restores:
            cls._release_fp8_sampling_for(demoted_ids, fp8_restores)
        for name, child in resident_blocks[key]["layers"]:
            cls.demote_layer(child, mm, layer_key=name)
            # The layer streams from now on; the streamed forward has its own
            # fp8 path, gated on this marker (mirrors _enable_fp8_sampling's
            # streamed-layer branch).
            weight = getattr(child, "weight", None)
            if (
                isinstance(weight, torch.nn.Parameter)
                and hasattr(weight.data, "qdata")
                and weight.data.qdata.dtype == torch.float8_e4m3fn
            ):
                child._memory_management_fp8_sampling = True
        if (
            target is not None
            and torch.device(target).type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(target)
            torch.cuda.empty_cache()
        offloaded_stream_layers = []
        offloaded_ids = set()
        offloaded_keys = set()
        resident_bytes = plan["model_bytes"]
        for name, child in module.named_modules():
            if not hasattr(child, "_layer_memory_manager"):
                continue
            offloaded_ids.add(id(child))
            offloaded_keys.add(cls._offload_group_key(name))
            resident_bytes -= cls._direct_module_bytes(child)
            offloaded_stream_layers.append(cls._stream_bytes(child))
        plan["offload_ids"] = offloaded_ids
        plan["offloaded_layers"] = len(offloaded_stream_layers)
        plan["offloaded_blocks"] = len(offloaded_keys)
        plan["resident_bytes"] = resident_bytes
        plan["ring_bytes"] = sum(
            sorted(offloaded_stream_layers, reverse=True)[:PIPELINE_DEPTH]
        )
        return freed

    @classmethod
    @contextlib.contextmanager
    def inference_resident(
        cls, module, device=None, fp8_sampling=False, working_reserve_gib=None,
        wddm_margin_gib=None, wddm_hard_gib=None,
        cold_start_hint_bytes=None,
    ):
        """Temporarily make an offloaded module fully GPU-resident for a forward-only run.

        Layer offloading re-streams (and, for quantized weights, re-dequantizes via fp32) every
        managed layer on *every* forward. That is ruinous for sampling, which runs the whole
        transformer once per denoise step — tens of times per image. For an inference run the
        VRAM footprint is small (no optimizer state, gradients, or backward activations), so the
        model usually fits resident.

        This detaches the streaming wrappers, moves the module to `device` so it runs its native
        (e.g. quantized) forward with no per-call streaming or fp32 dequant, then restores the
        original offload configuration afterward. If there is no manager it is a no-op; if the
        module does not fit resident (OOM) it restores offload and yields the streamed path.
        """
        cls._apply_wddm_hard_allocator_cap(device, wddm_hard_gib)
        diagnostics = cls._diagnostics_enabled()
        if module is None:
            yield
            return
        mm = getattr(module, "_memory_manager", None)
        had_manager = mm is not None
        args = dict(getattr(mm, "_attach_args", {}) or {}) if had_manager else {}
        original_smart_training_plan = (
            getattr(mm, "_smart_training_plan", None) if had_manager else None
        )
        original_fp8_training_requested = (
            getattr(mm, "_fp8_training_requested", False) if had_manager else False
        )
        target = device if device is not None else args.get("device")
        try:
            original_device = next(module.parameters()).device
        except StopIteration:
            original_device = torch.device("cpu")
        managed_layers = sum(
            1 for child in module.modules()
            if hasattr(child, "_layer_memory_manager")
        )
        original_offload_ids = {
            id(child) for child in module.modules()
            if hasattr(child, "_layer_memory_manager")
        }
        before = cls._cuda_memory(target)
        pre_sampling_dxgi = cls._dxgi_shared_budget_snapshot_bytes(target)
        if diagnostics:
            if not had_manager:
                training_layout = "none"
            elif args.get("training_strategy") == "smart":
                training_layout = f"smart ({managed_layers} streamed layers)"
            else:
                training_layout = (
                    f"{float(args.get('offload_percent', 0.0)):.0%}"
                )
            print(
                f"[MemoryManager] sampling start: {managed_layers} managed layers, "
                f"training_offload_to_restore={training_layout}, "
                f"{cls._format_cuda_memory(before)}"
            )

        fp8_restores = []
        # Mutable teardown state shared with the mid-denoise demote path, so the
        # finally block disables whatever fp8 forwards are CURRENTLY installed
        # (resident ones, or the streamed set after an emergency demote).
        sampling_state = {
            "fp8_restores": fp8_restores,
            "fully_streamed": False,
            "trim_count": 0,
            "trim_freed_bytes": 0,
            "trim_demotes": 0,
        }

        def _restore_offload():
            _FP8_STATS["enabled"] = False
            if _FP8_STATS.get("training_enabled", False):
                _FP8_STATS["kernel_calls"] = 0
                _FP8_STATS["fallback_calls"] = 0
            cls._disable_fp8_sampling(module, sampling_state["fp8_restores"])
            if hasattr(module, "_memory_manager"):
                cls.detach(module)
            try:
                cls._move_module_parameters(module, "cpu")
            except Exception:
                pass
            torch.cuda.empty_cache()
            if hasattr(module, "_mm_sampling_disable_resident_trace_hooks"):
                del module._mm_sampling_disable_resident_trace_hooks
            if args:
                cls.attach(
                    module,
                    **args,
                    _offload_module_ids=original_offload_ids,
                )
                if original_smart_training_plan is not None:
                    module._memory_manager._smart_training_plan = (
                        original_smart_training_plan
                    )
                    module._memory_manager._fp8_training_requested = (
                        original_fp8_training_requested
                    )
                    cls._refresh_training_fp8_flags(module, module._memory_manager)

                    if _OFFLOAD_PREFETCH_ENABLED and args.get("device") is not None:
                        cls._attach_prefetch_pool(module, args["device"])
                    # Sampling detach/restore replaces the streamed layout and
                    # destroys the old pool; any frozen positional trace from
                    # before sampling can now be stale against the restored set.
                    cls._invalidate_manual_training_shape_peaks(
                        module._memory_manager
                    )
                    cls.reset_trace_due_to_execution_shape_change()
                if args.get("device") is not None:
                    cls._move_unmanaged_parameters(module, args["device"])
            elif not had_manager:
                cls._move_module_parameters(module, original_device)

        cls.detach(module)
        module._mm_sampling_disable_resident_trace_hooks = True
        if not had_manager:
            cls._move_module_parameters(module, "cpu")
        # The streaming rings can hold several dequantized layer-sized CUDA buffers.
        # They are dead weight while the complete quantized model is resident.
        cls._clear_cuda_pipeline_state()
        after_clear = cls._cuda_memory(target)
        if diagnostics:
            freed = 0.0 if before is None or after_clear is None else before[0] - after_clear[0]
            print(
                f"[MemoryManager] released prior sampling layout: {freed:.2f} GiB; "
                f"{cls._format_cuda_memory(after_clear)}"
            )
        gib = 1024 ** 3
        dxgi_settle = cls._wait_for_sampling_dxgi_settle(target)
        if dxgi_settle.get("status") == "timeout":
            snap = dxgi_settle.get("snapshot") or {}
            _restore_offload()
            if diagnostics:
                print(
                    "[MemoryManager] sampling mode: streamed fallback "
                    "(DXGI NON_LOCAL did not settle after detach; "
                    f"raw_headroom={snap.get('raw_headroom_bytes', 0) / gib:.2f} GiB "
                    f"margin={snap.get('margin_bytes', 0) / gib:.2f} GiB "
                    f"attempts={dxgi_settle.get('attempts')})"
                )
            yield
            return
        elif diagnostics and dxgi_settle.get("status") == "settled":
            snap = dxgi_settle.get("snapshot") or {}
            print(
                "[MemoryManager] sampling DXGI settle: "
                f"raw_headroom={snap.get('raw_headroom_bytes', 0) / gib:.2f} GiB "
                f"margin={snap.get('margin_bytes', 0) / gib:.2f} GiB "
                f"attempts={dxgi_settle.get('attempts')}"
            )
        # Re-apply the allocator cap now that the previous layout's frees have
        # settled. The entry-time application can catch mem_get_info mid-flight
        # (WDDM returns freed memory lazily), inflating the measured non_torch
        # by GiBs and collapsing the cap -- observed as a second sample OOMing
        # at 8.5 GiB used with 3.5 GiB free, then demoting on that false
        # premise. This settled measurement replaces the stale one.
        cls._apply_wddm_hard_allocator_cap(target, wddm_hard_gib)
        # Sampling working_reserve is configured INDEPENDENTLY from training (the
        # ``working_reserve_gib`` argument, fed from layer_offloading_smart_sampling_
        # working_reserve_gb). Their VRAM profiles are very different: sampling is
        # forward-only, with no optimizer state, gradients, or backward
        # activations to reserve for, so it can run a much smaller reserve.
        # Floor/pad/cold-start stay as env overrides; the selection itself is a
        # pure helper (see _resolve_sampling_working_reserve).
        # Cold-start precedence: explicit env override (test harness) beats the
        # caller's shape-aware estimate (vram_budget.estimate_sampling_working_
        # reserve_bytes, passed by the trainer/smoke from the pending sample
        # configs), which beats the flat legacy default. A learned measured
        # reserve still replaces all of these (see _resolve_sampling_working_reserve).
        cold_start_env = os.environ.get("AI_TOOLKIT_SAMPLING_WORKING_RESERVE_GIB")
        if cold_start_env not in (None, ""):
            cold_start_working_reserve = int(float(cold_start_env) * gib)
        elif cold_start_hint_bytes is not None and int(cold_start_hint_bytes) > 0:
            cold_start_working_reserve = int(cold_start_hint_bytes)
        else:
            cold_start_working_reserve = int(3.0 * gib)
        working_reserve_floor = int(
            float(_env("AI_TOOLKIT_SAMPLING_WORKING_RESERVE_FLOOR_GIB", "1.5"))
            * gib
        )
        # Pad on top of the LEARNED (hot) reserve. 1.0 GiB by choice: streaming
        # one extra block is cheap, an underestimate costs a mid-denoise demote
        # (compiled-state invalidation, pack-set mutation).
        working_reserve_pad = int(
            float(_env("AI_TOOLKIT_SAMPLING_WORKING_RESERVE_PAD_GIB", "1.0")) * gib
        )
        learned_working_reserve = int(getattr(module, "_sampling_peak_working_reserve_bytes", 0))
        working_reserve_bytes, working_reserve_source = cls._resolve_sampling_working_reserve(
            working_reserve_gib,
            learned_working_reserve,
            cold_start_bytes=cold_start_working_reserve,
            floor_bytes=working_reserve_floor,
            pad_bytes=working_reserve_pad,
        )
        if (
            working_reserve_source == "cold-start"
            and cold_start_env in (None, "")
            and cold_start_hint_bytes is not None
            and int(cold_start_hint_bytes) > 0
        ):
            working_reserve_source = "cold-start-estimate"
        wddm_hard_bytes = int(
            float(
                _env("AI_TOOLKIT_SAMPLING_WDDM_HARD_GIB", "1.0")
                if wddm_hard_gib is None
                else wddm_hard_gib
            )
            * gib
        )
        # The sampling margin's only remaining job is the caching allocator's
        # reserved-over-allocated overshoot -- the WDDM cliff itself is now
        # guarded by the reclaim allocator cap (loud OOM/GC, never silent
        # paging). Measured on this class of run the overshoot is ~0.86 GiB and
        # rock-steady (std ~0.05 across 8 seeds), so the AUTO margin is that
        # measured overshoot + one safety block, NOT the old 0.10*card cushion
        # sized for a chaotic allocator that no longer jumps. Narrowing it hands
        # the difference to resident weights. An explicit config/env margin still
        # takes the legacy resolve + churn-pad path.
        def _margin_is_auto(value):
            if value is None:
                return True
            try:
                return float(value) < 0
            except (TypeError, ValueError):
                return str(value).strip().lower() == "auto"

        if os.environ.get("AI_TOOLKIT_SAMPLING_WDDM_MARGIN_GIB") in (None, "") and _margin_is_auto(
            wddm_margin_gib
        ):
            wddm_margin_bytes = vram_budget.sampling_overshoot_margin_bytes(
                hard_bytes=wddm_hard_bytes
            )
        else:
            wddm_margin_bytes = int(
                cls._resolve_wddm_margin_gib(
                    target,
                    wddm_margin_gib,
                    hard_gib=wddm_hard_bytes / gib,
                    env_name="AI_TOOLKIT_SAMPLING_WDDM_MARGIN_GIB",
                )
                * gib
            )
            # Reserved-churn pad on the explicit path: streaming FP8 forwards
            # leave the reserved pool ~0.5-1.3 GiB above allocated; budget for it
            # up front so a well-planned run never has to demote mid-denoise (a
            # mid-run demote grows the ingraph pack set, which strict ingraph
            # compilation rejects).
            wddm_margin_bytes += int(
                float(_env("AI_TOOLKIT_SAMPLING_RESERVED_CHURN_PAD_GIB", "0.5")) * gib
            )
        plan_snapshot = vram_budget.DeviceSnapshot.capture(target)
        free_bytes = plan_snapshot.free if plan_snapshot is not None else 0
        # Driver-free counts torch's own idle cache as used; the allocator cap
        # (just re-applied above, on a settled measurement) GCs that cache on
        # demand, so the allocated-side budget is the truer capacity. Take the
        # larger of the two so a stale/absent cap can only fall back to the
        # legacy driver-free behavior, never below it.
        budget_source = "driver-free"
        if plan_snapshot is not None:
            cap_index = (
                torch.device(target).index
                if torch.device(target).index is not None
                else torch.cuda.current_device()
            )
            alloc_side_free = vram_budget.sampling_allocator_budget_free_bytes(
                plan_snapshot.total,
                plan_snapshot.torch_allocated,
                cls._wddm_hard_cap_applied.get(cap_index),
                wddm_hard_bytes,
            )
            if alloc_side_free is not None and alloc_side_free > free_bytes:
                free_bytes = alloc_side_free
                budget_source = "allocator-cap"
        plan = cls._smart_sampling_plan(
            module,
            free_bytes,
            working_reserve_bytes,
            args.get("ignore_modules", []),
            wddm_margin_bytes=wddm_margin_bytes,
            wddm_hard_bytes=wddm_hard_bytes,
        )

        if diagnostics:
            print(
                f"[MemoryManager] smart budget: model={plan['model_bytes'] / gib:.2f} GiB "
                f"resident={plan['resident_bytes'] / gib:.2f} GiB "
                f"streamed_blocks={plan['offloaded_blocks']}/{plan['total_blocks']} "
                f"streamed_layers={plan['offloaded_layers']} "
                f"transfer_reserve={plan['ring_bytes'] / gib:.2f} GiB "
                f"sampling_working_reserve={plan['working_reserve_bytes'] / gib:.2f} GiB "
                f"({working_reserve_source}) "
                f"wddm_margin={plan['wddm_margin_bytes'] / gib:.2f} GiB "
                f"wddm_hard={plan.get('wddm_hard_bytes', 0) / gib:.2f} GiB "
                f"free={free_bytes / gib:.2f} GiB ({budget_source})"
            )

        # Bank the measured reclaimable. The entry cap was the cliff bound, but
        # the plan now tells us the true live footprint (resident + ring +
        # activation reserve). Tighten the cap to sit just above it, so the
        # allowance (0.95*cap - live) equals the reserved-churn pad and the
        # unused dedicated VRAM is handed back as the DXGI overflow valve and
        # tier-1 climb headroom -- exactly the floor the manual --cap-descent
        # probe finds. Clamped to the cliff inside the setter, so a tight
        # high-res plan is a no-op. The GC is lazy: this cap change is realized
        # on the resident move's first fresh malloc just below, not here.
        if (
            plan.get("fits")
            and plan_snapshot is not None
            and target is not None
            and torch.device(target).type == "cuda"
        ):
            planned_live_bytes = (
                int(plan.get("resident_bytes", 0))
                + int(plan.get("ring_bytes", 0))
                + int(plan.get("working_reserve_bytes", 0))
            )
            cache_budget_bytes = int(
                float(_env("AI_TOOLKIT_SAMPLING_RESERVED_CHURN_PAD_GIB", "0.5")) * gib
            )
            reclaim_cap_bytes = vram_budget.cap_bytes_for_live(
                planned_live_bytes,
                cache_budget_bytes,
                cliff_cap_bytes=plan_snapshot.total,
            )
            cls._apply_wddm_hard_allocator_cap(
                target, wddm_hard_gib, target_cap_bytes=reclaim_cap_bytes
            )

        move_started = time.perf_counter()
        try:
            if not plan["fits"]:
                raise torch.cuda.OutOfMemoryError(
                    "model, streaming buffers, and sampling working_reserve do not fit"
                )
            if plan["offload_ids"]:
                cls.attach(
                    module,
                    target,
                    offload_percent=1.0,
                    ignore_modules=args.get("ignore_modules", []),
                    _offload_module_ids=plan["offload_ids"],
                    pinned_weight_gib=args.get("pinned_weight_gib"),
                    # Ticket 534ea49: if training already folded these weights
                    # into a persistent pinned arena, reuse it here rather than
                    # falling back to pageable streaming -- _build_pinned_arena
                    # skips children already arena-current, so this costs
                    # nothing when the arena is already built.
                    use_pinned_arena=bool(args.get("use_pinned_arena", False)),
                )
                cls._move_unmanaged_parameters(module, target)
            elif target is not None:
                cls._move_module_parameters(module, target)
            if target is not None and torch.device(target).type == "cuda":
                torch.cuda.synchronize(target)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            _restore_offload()
            if diagnostics:
                print(
                    f"[MemoryManager] sampling mode: streamed fallback "
                    f"({type(error).__name__}: {error})"
                )
            yield  # fall back to the streamed path rather than crashing
            return

        # --- Driver-free hard-floor validation (anti-WDDM-spill retreat) ------
        # The residency plan is an ESTIMATE: the learned working working_reserve was
        # measured at a lower residency, and going resident costs more reserved
        # than the raw weight bytes (allocator cache + fragmentation), so it can
        # overshoot and land below the spill cliff (observed: planned 1.5 GiB
        # free, real 0.8 GiB -> WDDM spill -> steps ~1.5x slower). Measure the
        # REAL free now and demote whole resident blocks (largest first, to
        # clear fast) until denoising's working set will still leave the hard
        # buffer. Observed free validates; we never intentionally step over the
        # edge (AUTOTUNE_PLAN).
        retreat_cuda = (
            target is not None
            and torch.device(target).type == "cuda"
            and torch.cuda.is_available()
        )
        if retreat_cuda:
            # Room for the denoising working set (working_reserve) plus the spill
            # cushion. Its own name: this is the post-move validation floor, not
            # the planning margin (wddm_margin_bytes) it used to shadow.
            #
            # The reserve is deliberately OVERESTIMATED (+1 GiB cold-start
            # headroom / learned pad): that surplus is a BUFFER transients may
            # spend, not a floor to defend -- demoting to protect buffer space
            # would recreate the churn the buffer exists to prevent. So the
            # enforcement floor here subtracts the buffer; only the un-padded
            # working set plus the WDDM cushion is defended by demotion.
            reserve_buffer_bytes = int(
                float(_env("AI_TOOLKIT_SAMPLING_RESERVE_BUFFER_GIB", "1.0")) * gib
            )
            post_move_floor_bytes = (
                max(working_reserve_floor, working_reserve_bytes - reserve_buffer_bytes)
                + wddm_hard_bytes
            )
            free_now = vram_budget.device_free_bytes(target)
            if free_now < post_move_floor_bytes:
                if not hasattr(module, "_memory_manager"):
                    cls.attach(
                        module,
                        target,
                        offload_percent=1.0,
                        ignore_modules=args.get("ignore_modules", []),
                        _offload_module_ids=set(),
                    )
                cls._log_demotion_argument(
                    "post-move-retreat", target,
                    device_free=f"{free_now / gib:.2f} GiB",
                    post_move_floor=f"{post_move_floor_bytes / gib:.2f} GiB",
                    working_reserve=f"{working_reserve_bytes / gib:.2f} GiB",
                    reserve_buffer=f"{reserve_buffer_bytes / gib:.2f} GiB",
                    wddm_hard=f"{wddm_hard_bytes / gib:.2f} GiB",
                )
                demoted_blocks = 0
                while free_now < post_move_floor_bytes:
                    freed = cls._sampling_demote_largest_block(
                        module, plan, target,
                        ignore_modules=args.get("ignore_modules", []),
                    )
                    if not freed:
                        break
                    free_now = vram_budget.device_free_bytes(target)
                    demoted_blocks += 1
                if demoted_blocks and diagnostics:
                    floor_status = (
                        "cleared" if free_now >= post_move_floor_bytes else "still low"
                    )
                    print(
                        f"[MemoryManager] spill-guard retreat: demoted "
                        f"{demoted_blocks} resident block(s), floor={floor_status}; "
                        f"device_free={free_now / gib:.2f} GiB "
                        f"(target={post_move_floor_bytes / gib:.2f} GiB)"
                    )
        fp8_resident_layers = fp8_streamed_layers = 0
        fp8_supported = False
        if fp8_sampling and target is not None and torch.device(target).type == "cuda":
            major, minor = torch.cuda.get_device_capability(target)
            fp8_supported = hasattr(torch, "_scaled_mm") and (major, minor) >= (8, 9)
        if fp8_supported:
            (
                fp8_restores,
                fp8_resident_layers,
                fp8_streamed_layers,
            ) = cls._enable_fp8_sampling(module)
            sampling_state["fp8_restores"] = fp8_restores
            _FP8_STATS.update(
                enabled=diagnostics,
                kernel_calls=0,
                fallback_calls=0,
            )

        sampling_mode = (
            f"smart partial ({plan['offloaded_blocks']}/{plan['total_blocks']} "
            f"streamed blocks, {plan['offloaded_layers']} layers)"
            if plan["offload_ids"]
            else "fully resident"
        )
        if diagnostics:
            print(
                f"[MemoryManager] sampling mode: {sampling_mode}; "
                f"move={time.perf_counter() - move_started:.2f}s; "
                f"{cls._format_cuda_memory(cls._cuda_memory(target))}"
            )
            print(
                f"[MemoryManager] FP8 sampling: "
                f"{'enabled' if fp8_supported else 'disabled'} "
                f"(requested={bool(fp8_sampling)}); "
                f"resident_linear_layers={fp8_resident_layers} "
                f"streamed_linear_layers={fp8_streamed_layers}"
            )

        cuda_target = (
            target is not None
            and torch.device(target).type == "cuda"
            and torch.cuda.is_available()
        )
        resident_allocated = 0
        if cuda_target:
            # Start the peak window after the model move. The difference is the
            # activation/dequant/workspace working_reserve that sampling actually needed.
            resident_allocated = torch.cuda.memory_allocated(target)
            torch.cuda.reset_peak_memory_stats(target)

        def _demote_to_streamed(reason=None):
            """Mid-denoise OOM recovery (cap hit or another process grabbed VRAM).

            The residency plan is fixed for the whole run, so an OOM mid-step
            has no fallback. Relief is PROPORTIONAL: first stream a couple of
            the largest resident blocks (an OOM is usually a few hundred MiB
            short, not gigabytes -- demoting everything turned one 25-resident
            OOM into a fully-streamed run for the remaining passes). Only when
            no resident block is left to demote does it fall through to the
            full streamed transition. Returns True if it freed something
            (retry the step), False if already fully streamed (the caller
            should re-raise). Updates sampling_state so teardown disables
            whatever fp8 forwards are currently installed.
            """
            if sampling_state["fully_streamed"]:
                return False
            # First: was the OOM even real? The cap is derived from a measured
            # non_torch share, and a measurement taken while WDDM was still
            # returning freed memory (lazy frees) collapses the cap by GiBs --
            # observed: OOM at 8.5 GiB used with 3.5 GiB free. Re-measure now;
            # if the cap loosens meaningfully, that IS the relief -- retry the
            # step without paying any residency.
            if cuda_target:
                index = (
                    torch.device(target).index
                    if torch.device(target).index is not None
                    else torch.cuda.current_device()
                )
                prev_fraction = cls._wddm_hard_cap_applied.get(index)
                cls._apply_wddm_hard_allocator_cap(target, wddm_hard_bytes / gib)
                new_fraction = cls._wddm_hard_cap_applied.get(index)
                total_bytes = torch.cuda.get_device_properties(index).total_memory
                if (
                    prev_fraction is not None
                    and new_fraction is not None
                    and (new_fraction - prev_fraction) * total_bytes > 256 * 1024 ** 2
                ):
                    if diagnostics:
                        print(
                            "[MemoryManager] mid-denoise OOM: stale allocator cap "
                            f"({prev_fraction * total_bytes / gib:.2f} -> "
                            f"{new_fraction * total_bytes / gib:.2f} GiB after "
                            "re-measure); retrying without demotion."
                        )
                    return True
            # The OOM unwound a forward that may have had ingraph fetches in
            # flight; their tickets never reach fetch_free and would wedge the
            # next compiled forward at the depth limit ('fetch_start depth
            # exceeded'). Nothing consumes those buffers anymore -- drain.
            abandoned_fetches = ingraph_drain_fetch_runtime()
            if abandoned_fetches and diagnostics:
                print(
                    f"[MemoryManager] mid-denoise OOM: abandoned "
                    f"{abandoned_fetches} in-flight ingraph fetch(es)"
                )
            if cuda_target:
                # Full sync so side-stream frees (transfer-stream fetch
                # buffers) become releasable before the cache trim; without it
                # empty_cache leaves them as fragmented idle reserved blocks
                # that cannot serve a main-stream contiguous allocation.
                torch.cuda.synchronize(target)
            torch.cuda.empty_cache()
            cls._log_demotion_argument(
                "mid-denoise-oom", target,
                oom=repr(reason)[:220] if reason is not None else "unreported",
                abandoned_fetches=abandoned_fetches,
                resident_blocks=(
                    f"{plan.get('total_blocks', 0) - plan.get('offloaded_blocks', 0)}"
                    f"/{plan.get('total_blocks', 0)}"
                ),
                working_reserve=f"{working_reserve_bytes / gib:.2f} GiB",
            )
            if not hasattr(module, "_memory_manager"):
                cls.attach(
                    module,
                    target,
                    offload_percent=1.0,
                    ignore_modules=args.get("ignore_modules", []),
                    _offload_module_ids=set(),
                )
            demote_blocks = max(
                1, int(_env("AI_TOOLKIT_SAMPLING_OOM_DEMOTE_BLOCKS", "2"))
            )
            # Govern on MEASURED device-free, not the demoted blocks' weight
            # bytes: a leaked reference (e.g. the fp8 closure pin, now fixed)
            # or fragmentation can make demotion free nothing, and plan-bytes
            # would loop through the whole model claiming progress. If the
            # device gains almost nothing, per-block demotion is futile --
            # fall through to the full streamed transition, whose detach +
            # move + empty_cache releases and defragments everything at once.
            free_before = cls._torch_allocatable_bytes(target) if cuda_target else 0
            planned = 0
            demoted = 0
            for _ in range(demote_blocks):
                freed_now = cls._sampling_demote_largest_block(
                    module, plan, target,
                    ignore_modules=args.get("ignore_modules", []),
                    fp8_restores=sampling_state["fp8_restores"],
                )
                if not freed_now:
                    break
                planned += freed_now
                demoted += 1
            measured = (
                max(0, cls._torch_allocatable_bytes(target) - free_before)
                if cuda_target
                else planned
            )
            futile = planned > 0 and measured < min(planned // 4, 64 * 1024 ** 2)
            if demoted and not futile:
                if cuda_target:
                    torch.cuda.reset_peak_memory_stats(target)
                if diagnostics:
                    print(
                        f"[MemoryManager] mid-denoise OOM: streamed {demoted} "
                        f"resident block(s), freed {measured / gib:.2f} GiB "
                        f"(weights {planned / gib:.2f} GiB); retrying step."
                    )
                return True
            if diagnostics and futile:
                print(
                    f"[MemoryManager] mid-denoise OOM: demotion futile "
                    f"(weights {planned / gib:.2f} GiB, device gained "
                    f"{measured / gib:.2f} GiB) -> full streamed transition."
                )
            cls._disable_fp8_sampling(module, sampling_state["fp8_restores"])
            sampling_state["fp8_restores"] = []
            if hasattr(module, "_memory_manager"):
                cls.detach(module)
            try:
                cls._move_module_parameters(module, "cpu")
            except Exception:
                pass
            cls._clear_cuda_pipeline_state()
            torch.cuda.empty_cache()
            ignore = args.get("ignore_modules", [])
            all_ids = {
                id(child) for child, _, _ in cls._sampling_candidates(module, ignore)
            }
            cls.attach(
                module, target, offload_percent=1.0,
                ignore_modules=ignore, _offload_module_ids=all_ids,
                pinned_weight_gib=args.get("pinned_weight_gib"),
                use_pinned_arena=bool(args.get("use_pinned_arena", False)),
            )
            cls._move_unmanaged_parameters(module, target)
            if cuda_target:
                torch.cuda.synchronize(target)
            if fp8_supported:
                restores, _, _ = cls._enable_fp8_sampling(module)
                sampling_state["fp8_restores"] = restores
            sampling_state["fully_streamed"] = True
            if diagnostics:
                print(
                    "[MemoryManager] mid-denoise OOM: demoted to fully-streamed "
                    "(external VRAM pressure); retrying step."
                )
            return True

        # Exposed so the model's denoise loop can recover from an OOM caused by
        # external memory pressure instead of crashing the whole sampling run.
        module._mm_sampling_demote = _demote_to_streamed

        # Guard/trim thresholds are fixed for the phase: resolve them ONCE here
        # instead of re-reading env vars inside the per-step closures (they
        # cannot change mid-run, and re-reads hide which value governed).
        guard_margin = int(
            max(
                float(_env("AI_TOOLKIT_SAMPLING_GUARD_MARGIN_GIB", "0.5")) * gib,
                plan.get("wddm_hard_bytes", 0),
            )
        )
        trim_margin = int(
            max(
                float(_env("AI_TOOLKIT_SAMPLING_STEP_TRIM_GIB", "1.5")) * gib,
                plan.get("wddm_margin_bytes", 0),
            )
        )
        hard_floor = int(max(wddm_hard_bytes, plan.get("wddm_hard_bytes", 0)))

        def _sampling_guard():
            """Reactive cohabitation guard, called per image before compile.

            Forward-only sampling does NOT raise when it crosses the WDDM cliff —
            it silently pages to shared memory and runs ~5x slower, so the OOM
            recovery above never fires. This proactively gives VRAM back: if an
            external grab (Windows desktop, another app) would push the next
            forward's peak within the spill margin, stream one resident block.
            Reactive by design (per-image): a sudden spike may page one image
            before the next check catches it. Returns blocks demoted.
            """
            if not cuda_target or sampling_state["fully_streamed"]:
                return 0
            free_b, total_b = vram_budget.device_mem_info(target)
            reserved_b = torch.cuda.memory_reserved(target)
            peak_reserved_b = torch.cuda.max_memory_reserved(target)
            # Predicted device-free at the next forward's peak. (peak stats are
            # reset at sampling start, so this reflects only sampling forwards.)
            predicted_peak_free = cls._sampling_guard_predicted_peak_free(
                total_b, free_b, reserved_b, peak_reserved_b
            )
            if predicted_peak_free >= guard_margin:
                return 0
            cls._log_demotion_argument(
                "cohabitation-guard", target,
                predicted_peak_free=f"{predicted_peak_free / gib:.2f} GiB",
                guard_margin=f"{guard_margin / gib:.2f} GiB",
                peak_reserved=f"{peak_reserved_b / gib:.2f} GiB",
            )
            freed = cls._sampling_demote_largest_block(
                module, plan, target,
                ignore_modules=args.get("ignore_modules", []),
                fp8_restores=sampling_state["fp8_restores"],
            )
            if not freed:
                return 0
            # The high-water no longer reflects the smaller layout; let the next
            # forward re-establish it so the following check stays accurate.
            torch.cuda.reset_peak_memory_stats(target)
            if diagnostics:
                print(
                    f"[MemoryManager] sampling guard: external pressure "
                    f"(predicted peak free {predicted_peak_free / gib:.2f} < "
                    f"{guard_margin / gib:.2f} GiB) -> streamed 1 block, "
                    f"freed {freed / gib:.2f} GiB"
                )
            return 1

        # Exposed so the per-image generate loop can pre-empt WDDM paging when
        # external VRAM use grows mid-run (paging is silent, not an OOM).
        module._mm_sampling_guard = _sampling_guard

        def _sampling_step_trim():
            """Per-denoise-step cache trim, called before each forward.

            Streaming FP8 layers re-allocate transient buffers (input cast,
            _scaled_mm output, unpacked qdata) every forward; the caching
            allocator keeps those freed blocks at its reserved high-water rather
            than returning them to the driver. The more blocks stream, the larger
            that idle high-water grows -- so a *bigger* working_reserve (which
            forces more streaming) can push device-used UP, toward the WDDM cliff,
            even though live activations are small. WDDM pages on the committed
            (reserved) footprint and does so silently, so neither the OOM retry
            nor an allocated-side signal catches it.

            Remedy, escalating and gated so it is a no-op when there is slack:
              1. If realized device-free has dropped within the trim margin,
                 empty_cache() to return idle cached blocks to the driver (cheap,
                 non-destructive). The next forward re-allocates fresh, defragmented
                 blocks, so free typically recovers and later steps stop trimming.
              2. Only if free is STILL under the hard floor after trimming (no
                 idle cache left to reclaim -> genuine external pressure) demote
                 one resident block. Demotion adds streaming churn, so it is the
                 last resort, not the first.
            Triggers on realized free (not the peak high-water) so it never resets
            the peak stats the teardown uses to LEARN the working reserve.
            Returns bytes reclaimed by the trim (demotion counted separately).
            """
            if not cuda_target or sampling_state["fully_streamed"]:
                return 0
            # Non-torch device usage GROWS mid-run (Triton/compiled kernel code
            # loads outside the caching allocator: observed 1.15 -> 1.68 GiB
            # across compile warmup). Re-measure and re-tighten the allocator
            # cap so torch's share shrinks in step and the hard device-free
            # floor survives the growth. Cheap + idempotent when unchanged.
            cls._apply_wddm_hard_allocator_cap(target, hard_floor / gib)
            before = vram_budget.device_free_bytes(target)
            if not cls._sampling_step_should_trim(before, trim_margin):
                return 0
            torch.cuda.empty_cache()
            free_b = vram_budget.device_free_bytes(target)
            freed = max(0, free_b - before)
            sampling_state["trim_count"] += 1
            sampling_state["trim_freed_bytes"] += freed
            # Escalate to demotion only if trimming did not buy back the floor.
            demoted_blocks = 0
            if cls._sampling_step_should_demote(free_b, hard_floor):
                cls._log_demotion_argument(
                    "step-trim-escalation", target,
                    free_before_trim=f"{before / gib:.2f} GiB",
                    free_after_trim=f"{free_b / gib:.2f} GiB",
                    trim_margin=f"{trim_margin / gib:.2f} GiB",
                    hard_floor=f"{hard_floor / gib:.2f} GiB",
                )
                demoted = cls._sampling_demote_largest_block(
                    module, plan, target,
                    ignore_modules=args.get("ignore_modules", []),
                    fp8_restores=sampling_state["fp8_restores"],
                )
                if demoted:
                    demoted_blocks = 1
                    sampling_state["trim_demotes"] += 1
                    if diagnostics:
                        free_b = vram_budget.device_free_bytes(target)
                        print(
                            f"[MemoryManager] step trim: cache trim left "
                            f"{(before + freed) / gib:.2f} GiB free (< hard floor "
                            f"{hard_floor / gib:.2f}); demoted 1 block, freed "
                            f"{demoted / gib:.2f} GiB -> {free_b / gib:.2f} GiB free"
                        )
            # Returns blocks demoted THIS step: the caller must invalidate any
            # compiled-block set, because a demoted block just gained a streaming
            # hook and its stale compiled graph would replay resident weights.
            return demoted_blocks

        # Exposed so the denoise loop can collapse the streaming-churn reserved
        # high-water each step before it silently crosses the WDDM cliff.
        module._mm_sampling_step_trim = _sampling_step_trim

        try:
            yield
        finally:
            if cuda_target:
                torch.cuda.synchronize(target)
            sample_end = cls._cuda_memory(target)
            peak_allocated = (
                torch.cuda.max_memory_allocated(target) if cuda_target else 0
            )
            sampling_working_reserve = max(0, peak_allocated - resident_allocated)
            # Streaming buffers are budgeted separately. Learn only the residual
            # denoising activation/workspace requirement for the next sample.
            observed_working_reserve = max(
                0, sampling_working_reserve - plan["ring_bytes"]
            )
            previous_working_reserve = int(
                getattr(module, "_sampling_peak_working_reserve_bytes", 0)
            )
            module._sampling_peak_working_reserve_bytes = max(
                previous_working_reserve, observed_working_reserve
            )
            if diagnostics and cuda_target:
                # Everything below is already raw; the only new thing is the
                # synthesis: how many MORE same-batch forwards the free VRAM holds.
                # Denominator is the activation/workspace that scales with batch
                # (learned reserve — ring excluded, it is fixed streaming buffers).
                # Free is taken conservatively AT the peak: total - (peak_reserved +
                # non-torch other), so it does not overstate room using step-end
                # free. >= 1.0 spare => one extra concurrent sample (batched CFG)
                # fits at this batch.
                peak_reserved = torch.cuda.max_memory_reserved(target) / gib
                other_gib = max(0.0, sample_end[2] - sample_end[1]) if sample_end else 0.0
                free_at_peak = max(0.0, sample_end[4] - (peak_reserved + other_gib)) if sample_end else 0.0
                per_forward_gib = module._sampling_peak_working_reserve_bytes / gib
                batch_room = (
                    free_at_peak / per_forward_gib if per_forward_gib > 1e-6 else float("inf")
                )
                print(
                    f"[MemoryManager] sampling peak: "
                    f"torch_allocated={peak_allocated / gib:.2f} GiB "
                    f"sampling_extra={sampling_working_reserve / gib:.2f} GiB "
                    f"learned_working_reserve={per_forward_gib:.2f} GiB "
                    f"(free fits ~{batch_room:.1f} more forwards); "
                    f"{cls._format_cuda_memory(sample_end)}"
                )
            if diagnostics and fp8_supported:
                print(
                    f"[MemoryManager] FP8 execution: "
                    f"native_calls={_FP8_STATS['kernel_calls']} "
                    f"fallback_calls={_FP8_STATS['fallback_calls']}"
                )
            if diagnostics and sampling_state["trim_count"]:
                print(
                    f"[MemoryManager] step trim summary: trimmed cache on "
                    f"{sampling_state['trim_count']} step(s), reclaimed "
                    f"{sampling_state['trim_freed_bytes'] / gib:.2f} GiB total, "
                    f"{sampling_state['trim_demotes']} demote escalation(s). "
                    f"Frequent trims => working_reserve is too high (too much "
                    f"streaming); lower it for more resident/compiled blocks."
                )
            if hasattr(module, "_mm_sampling_demote"):
                del module._mm_sampling_demote
            if hasattr(module, "_mm_sampling_guard"):
                del module._mm_sampling_guard
            if hasattr(module, "_mm_sampling_step_trim"):
                del module._mm_sampling_step_trim
            restore_started = time.perf_counter()
            _restore_offload()
            post_restore_dxgi = cls._dxgi_shared_budget_snapshot_bytes(target)
            if diagnostics:
                dxgi_restore_text = "dxgi_non_local=unavailable"
                if pre_sampling_dxgi is not None and post_restore_dxgi is not None:
                    delta_gib = (
                        post_restore_dxgi["usage_bytes"] - pre_sampling_dxgi["usage_bytes"]
                    ) / gib
                    tolerance_gib = max(0.5, post_restore_dxgi.get("margin_bytes", 0) / gib * 0.25)
                    status = "ok" if abs(delta_gib) <= tolerance_gib else "changed"
                    dxgi_restore_text = (
                        f"dxgi_non_local_usage={pre_sampling_dxgi['usage_bytes'] / gib:.2f}"
                        f"->{post_restore_dxgi['usage_bytes'] / gib:.2f} GiB "
                        f"delta={delta_gib:+.2f} GiB status={status}"
                    )
                print(
                    f"[MemoryManager] sampling end: current before restore "
                    f"{cls._format_cuda_memory(sample_end)}; restore={time.perf_counter() - restore_started:.2f}s; "
                    f"{cls._format_cuda_memory(cls._cuda_memory(target))}; {dxgi_restore_text}"
                )
