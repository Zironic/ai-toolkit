import contextlib
import json
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
    _unpin_inner_tensors,
    fp8_linear_inference,
    fp8_sampling_qualifies,
    _fp8_linear_compiled,
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
)
from . import bounce_pool
import random


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
    "AI_TOOLKIT_TRAINING_MAX_WORKING_RESERVE_GIB": "AI_TOOLKIT_TRAINING_MAX_HEADROOM_GIB",
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


class MemoryManager:
    def __init__(
        self,
        module: torch.nn.Module,
        process_device: torch.device = torch.device("cpu"),
    ):
        self.module: torch.nn.Module = module
        self.process_device: torch.device = process_device
        self.unmanaged_modules: list[torch.nn.Module] = []
        self.pinned_weight_bytes = 0
        default_pin_gib = "0.0" if _OFFLOAD_PREFETCH_ENABLED else "1.0"
        self.pinned_weight_budget_bytes = int(
            float(_env("AI_TOOLKIT_PINNED_WEIGHT_GIB", default_pin_gib))
            * 1024 ** 3
        )
        self._prefetch_pool = None
        self._resident_trace_hooks = {}

    def memory_managed_to(self, *args, **kwargs):
        # check for a dtype argument
        dtype = None
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        elif len(args) > 0:
            for i, arg in enumerate(args):
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        target_device = kwargs.get("device")
        if target_device is None:
            for arg in args:
                if isinstance(arg, (torch.device, str)):
                    target_device = arg
                    break
        # Device-only moves need special handling for TorchAO Parameters.
        if target_device is not None and dtype is None:
            MemoryManager._move_unmanaged_parameters(
                self.module, target_device
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
        offload_percent: float = 1.0,
        ignore_modules: list[torch.nn.Module] = [],
        _offload_module_ids: set[int] | None = None,
        training_strategy: str = "percent",
    ):
        if hasattr(module, "_memory_manager"):
            # already attached
            return

        module._memory_manager = cls(module, device)
        # remember how we were attached so we can re-attach identically after a temporary
        # detach (see inference_resident).
        module._memory_manager._attach_args = {
            "device": device,
            "offload_percent": offload_percent,
            "ignore_modules": list(ignore_modules),
            "training_strategy": training_strategy,
        }

        # override the to method to handle memory management
        module._mm_to = module.to
        module.to = module._memory_manager.memory_managed_to

        # add ignore modules to unmanaged list
        for im in ignore_modules:
            module._memory_manager.unmanaged_modules.append(im)

        # count ignore modules as processed
        modules_processed = [x for x in ignore_modules]
        # attach to all modules
        for name, sub_module in module.named_modules():
            for child_name, child_module in sub_module.named_modules():
                if (
                    child_module.__class__.__name__ in LINEAR_MODULES
                    and child_module not in modules_processed
                ):
                    if _offload_module_ids is not None:
                        skip = id(child_module) not in _offload_module_ids
                    else:
                        skip = False
                    if _offload_module_ids is None and offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        # linear
                        LinearLayerMemoryManager.attach(
                            child_module, module._memory_manager
                        )
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara,
                                    device,
                                )
                    modules_processed.append(child_module)
                elif (
                    child_module.__class__.__name__ in CONV_MODULES
                    and child_module not in modules_processed
                ):
                    if _offload_module_ids is not None:
                        skip = id(child_module) not in _offload_module_ids
                    else:
                        skip = False
                    if _offload_module_ids is None and offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        # conv
                        ConvLayerMemoryManager.attach(
                            child_module, module._memory_manager
                        )
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara,
                                    device,
                                )
                            modules_processed.append(ara)
                    modules_processed.append(child_module)
                elif child_module.__class__.__name__ in UNMANAGED_MODULES or any(
                    inc in child_module.__class__.__name__
                    for inc in UNMANAGED_MODULES_INCLUDES
                ):
                    # unmanaged
                    module._memory_manager.unmanaged_modules.append(child_module)
                else:
                    continue
        # Assign each streamable candidate a stable identity from its module path.
        # The trace scheduler keys on this rather than id(weight), which would not
        # survive the Parameter replacement that sampling detach/restore does.
        for name, child in module.named_modules():
            if child.__class__.__name__ in LINEAR_MODULES or child.__class__.__name__ in CONV_MODULES:
                child._mm_layer_key = name or child.__class__.__name__
        cls._refresh_resident_trace_hooks(module, module._memory_manager)
        if cls._diagnostics_enabled():
            gib = 1024 ** 3
            managed = sum(
                1 for child in module.modules()
                if hasattr(child, "_layer_memory_manager")
            )
            print(
                f"[MemoryManager] training offload attached: "
                f"managed_layers={managed} "
                f"pinned_cpu={module._memory_manager.pinned_weight_bytes / gib:.2f} GiB "
                f"pin_budget={module._memory_manager.pinned_weight_budget_bytes / gib:.2f} GiB"
            )

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
                container = getattr(lmm, "_forward_container", None)
                attribute = getattr(lmm, "_forward_attribute", None)
                if container is not None and attribute is not None:
                    setattr(container, attribute, original_forward)
                elif hasattr(child, "ara_lora_ref"):
                    ara = child.ara_lora_ref()
                    if ara is not None:
                        ara.org_forward = original_forward
                else:
                    child.forward = original_forward

            for param_name in ("weight", "bias"):
                param = getattr(child, param_name, None)
                if param is None or not isinstance(param, torch.nn.Parameter):
                    continue
                try:
                    if _is_quantized_tensor(param.data):
                        _unpin_inner_tensors(param.data)
                    if param.data.is_pinned():
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
            for name, param in list(child._parameters.items()):
                if param is None:
                    continue
                if _is_quantized_tensor(param.data):
                    moved = MemoryManager._move_tensor_subclass(param.data, target)
                    replacement = torch.nn.Parameter(
                        moved, requires_grad=param.requires_grad
                    )
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
            restores.append((container, attribute, original_forward))
            resident_layers += 1
        return restores, resident_layers, streamed_layers

    @staticmethod
    def _disable_fp8_sampling(module, restores):
        for container, attribute, original_forward in reversed(restores):
            setattr(container, attribute, original_forward)
        for child in module.modules():
            if hasattr(child, "_memory_management_fp8_sampling"):
                del child._memory_management_fp8_sampling

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

    @classmethod
    def smart_training_plan(
        cls,
        module,
        device,
        working_reserve_gib=2.0,
        ignore_modules=None,
        must_resident_keys=("tproj.1",),
        resident_floor_gib=2.0,
        wddm_margin_gib=1.5,
        wddm_hard_gib=None,
        prefetch_healthy=False,
        pinned_resident_keys=None,
        cold_growth=False,
        block_stream_only=False,
    ):
        """Choose training-resident layers with stream buffers before growth."""
        ignore_modules = list(ignore_modules or [])
        device = torch.device(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
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
        wddm_margin_gib = max(float(wddm_margin_gib), wddm_hard_gib)
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
            item["pinned_resident"] = group_key in pinned_keys
            item["block_stream_resident"] = bool(
                block_stream_only and not _is_streaming_block(group_key)
            )
            if (
                item["pinned_resident"]
                or item["block_stream_resident"]
                or any(token and token in item["key"] for token in must_tokens)
            ):
                resident.append(item)
            else:
                offloaded.append(item)

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
            for item in sorted(offloaded, key=lambda row: row["resident_bytes"]):
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
                list(offloaded), key=lambda row: row["resident_bytes"], reverse=True
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
        """Decide the next training working_reserve reservation (pure, CPU-testable).

        The static reserve over-holds when it exceeds the real backward peak, so
        the realised ``device_free`` margin balloons past the stop-line. This
        walks the reserve down toward ``measured_peak + pad`` one conservative
        step at a time, but only while a full window has proven there is slack
        above the stop-line — and never below a level that previously breached.

        Asymmetric on purpose (per AUTOTUNE_PLAN): shrinking is additive and
        gated; a breach retreats hard and locks out the danger zone for good.

        Returns ``(new_working_reserve_gib, new_danger_gib, action)``.
        """
        # 1. Breach of the hard floor: give the memory straight back and remember
        #    this reserve level as unsafe — never shrink to/near it again.
        if min_device_free_gib < wddm_hard_gib:
            new_danger = max(danger_gib or 0.0, current_gib)
            return current_gib + retreat_gib, new_danger, "retreat"

        target_gib = measured_peak_gib + pad_gib
        # 2. RESPECT the working set: the activation peak is a physical given, not
        #    something to minimise. If the reserve sits BELOW what the backward
        #    actually peaked at, climb straight up to meet it — growing the reserve
        #    only sets aside more FREE VRAM, so it is always safe. This is the
        #    branch that was missing: the shrink path below only ever walked DOWN
        #    toward the target, so a reserve seeded under the real peak (auto seeds
        #    ~2-3 GiB) could never reach it. It then under-reserved the activations,
        #    which overflowed into the allocator's prefetch/cache every step →
        #    eviction → bounce misses → prefetch never healthy → resident growth
        #    locked out. Meeting the measured peak is the root fix.
        if current_gib < target_gib:
            return target_gib, danger_gib, "grow"

        # 3. Shrink only with proven slack: after a step we must still clear the
        #    stop-line, we must stay at/above the real backward need, and we must
        #    not approach a known danger level.
        if min_device_free_gib > wddm_stop_gib + step_gib and current_gib > target_gib:
            candidate = max(target_gib, current_gib - step_gib)
            if danger_gib is not None and candidate <= danger_gib + step_gib:
                return current_gib, danger_gib, "danger_locked"
            return candidate, danger_gib, "shrink"

        # 4. Settled — at the reserve the measurement and margin both endorse.
        return current_gib, danger_gib, "hold"

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

        lmm._install_base_forward(lmm._original_forward)
        if hasattr(child, "_memory_management_device"):
            del child._memory_management_device
        del child._layer_memory_manager
        cls._refresh_resident_trace_hooks(lmm.manager.module, lmm.manager)
        return True

    @classmethod
    def demote_layer(cls, child, manager, layer_key=None):
        """Stream one resident layer, in place (resident -> offloaded).

        Reuses the per-layer manager attach, which moves ``param.data`` to
        pinned CPU and installs the streaming forward without replacing the
        Parameter — again preserving identity for the optimizer. Returns True
        if a transition happened.
        """
        if child is None or hasattr(child, "_layer_memory_manager"):
            return False
        name = child.__class__.__name__
        if name in LINEAR_MODULES:
            LinearLayerMemoryManager.attach(child, manager)
        elif name in CONV_MODULES:
            ConvLayerMemoryManager.attach(child, manager)
        else:
            return False
        child._mm_layer_key = layer_key or getattr(child, "_mm_layer_key", None) or name
        cls._refresh_resident_trace_hooks(manager.module, manager)
        return True

    @classmethod
    def attach_smart_training(
        cls, module, device, working_reserve_gib=2.0, ignore_modules=None,
        wddm_margin_gib=None,
        wddm_hard_gib=None,
        fp8_training_forward=False,
        pinned_resident_keys=None,
        block_stream_only=False,
    ):
        ignore_modules = list(ignore_modules or [])
        pinned_resident_keys = set(pinned_resident_keys or ())
        auto_working_reserve = False
        try:
            auto_working_reserve = float(working_reserve_gib) < 0
        except (TypeError, ValueError):
            auto_working_reserve = str(working_reserve_gib).lower() == "auto"
        if auto_working_reserve:
            gib = 1024 ** 3
            model_gib = cls._module_bytes(module) / gib
            # Attention no longer materializes the old oversized workspace, so
            # seed auto mode closer to the measured working set and let the
            # live spill guard retreat if a shape proves larger.
            working_reserve_gib = float(
                _env(
                    "AI_TOOLKIT_TRAINING_AUTO_SEED_WORKING_RESERVE_GIB",
                    str(max(2.0, min(3.0, model_gib * 0.17))),
                )
            )
        plan = cls.smart_training_plan(
            module, device, working_reserve_gib, ignore_modules,
            wddm_margin_gib=(
                float(_env("AI_TOOLKIT_TRAINING_WDDM_MARGIN_GIB", "1.0"))
                if wddm_margin_gib is None
                else wddm_margin_gib
            ),
            wddm_hard_gib=(
                float(_env("AI_TOOLKIT_TRAINING_WDDM_HARD_GIB", "1.0"))
                if wddm_hard_gib is None
                else wddm_hard_gib
            ),
            pinned_resident_keys=pinned_resident_keys,
            # Manual working_reserve has no live loop to climb later, so fill the
            # surplus at attach. Auto working_reserve leaves the climb to the live loop.
            cold_growth=not auto_working_reserve,
            block_stream_only=block_stream_only,
        )
        cls.attach(
            module,
            device,
            offload_percent=0.0,
            ignore_modules=ignore_modules,
            _offload_module_ids=plan["offload_ids"],
            training_strategy="smart",
        )
        module._memory_manager._smart_training_plan = plan
        module._memory_manager._training_pinned_resident_keys = set(pinned_resident_keys)
        module._memory_manager._training_block_stream_only = bool(block_stream_only)
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
    def _refresh_training_plan_from_layout(cls, module, mm, working_reserve_gib=None):
        old_plan = getattr(mm, "_smart_training_plan", {}) or {}
        args = getattr(mm, "_attach_args", {}) or {}
        ignore = args.get("ignore_modules", [])
        pinned_keys = set(getattr(mm, "_training_pinned_resident_keys", set()))
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
    def _demote_training_layers(cls, module, mm, count, *, largest=True):
        args = getattr(mm, "_attach_args", {}) or {}
        pinned_keys = set(getattr(mm, "_training_pinned_resident_keys", set()))
        layout = list(cls._training_layout_candidates(
            module, args.get("ignore_modules", []), pinned_keys
        ))
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
            item for item in layout
            if not item["managed"]
            and not item.get("pinned_resident")
            and _streamable(item)
        ]
        candidates.sort(
            key=lambda item: item["resident_bytes"], reverse=bool(largest)
        )
        changed = 0
        for item in candidates[: max(0, int(count))]:
            if cls.demote_layer(item["module"], mm, layer_key=item["name"]):
                changed += 1
        if changed:
            cls._register_training_prefetch_sources(module, mm)
            cls._refresh_training_fp8_flags(module, mm)
            cls._refresh_training_plan_from_layout(module, mm)
            cls.reset_trace_due_to_execution_shape_change()
            cls._clear_cuda_pipeline_state()
        return changed

    @classmethod
    def _promote_training_layer(cls, module, mm, device, *, cache_pad_gib, wddm_stop_gib):
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
        driver_free_bytes = torch.cuda.mem_get_info(device)[0]
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
                free_after = torch.cuda.mem_get_info(device)[0]
            except Exception:
                free_after = stop_bytes
            if free_after < stop_bytes:
                cls.demote_layer(child, mm, layer_key=item["name"])
                cls._refresh_training_fp8_flags(module, mm)
                cls._refresh_training_plan_from_layout(module, mm)
                cls.reset_trace_due_to_execution_shape_change()
                cls._clear_cuda_pipeline_state()
                return 0, "validated_low_free"
            cls._refresh_training_fp8_flags(module, mm)
            cls._refresh_training_plan_from_layout(module, mm)
            cls.reset_trace_due_to_execution_shape_change()
            cls._clear_cuda_pipeline_state()
            return 1, "promote"
        return 0, "stop_line"

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
            driver_free_bytes = torch.cuda.mem_get_info(device)[0]
            allocator_cached_bytes = max(0, allocatable_bytes - driver_free_bytes)
            driver_bytes_needed = max(0, need - allocator_cached_bytes)
            if driver_free_bytes - driver_bytes_needed < stop_bytes:
                break
            child = item["module"]
            if not cls.promote_layer(child):
                continue
            try:
                torch.cuda.synchronize(device)
                free_after = torch.cuda.mem_get_info(device)[0]
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
        cls.reset_trace_due_to_execution_shape_change()
        cls._clear_cuda_pipeline_state()
        return promoted, "promote_batch"

    @staticmethod
    def _torch_allocatable_bytes(device):
        """Driver-free plus allocator cache that PyTorch can reuse directly."""
        free_bytes = torch.cuda.mem_get_info(device)[0]
        reserved_bytes = torch.cuda.memory_reserved(device)
        allocated_bytes = torch.cuda.memory_allocated(device)
        return int(free_bytes + max(0, reserved_bytes - allocated_bytes))

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
            return cls._training_cliff_safety_net(module, mm, device, did_oom=did_oom)
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
        retreat_layers = int(_env("AI_TOOLKIT_TRAINING_RETREAT_LAYERS", "3"))
        promote_interval = int(_env("AI_TOOLKIT_TRAINING_PROMOTE_INTERVAL", "4"))
        cache_pad_gib = float(_env("AI_TOOLKIT_TRAINING_CACHE_PAD_GIB", "0.25"))
        min_working_reserve_gib = float(_env("AI_TOOLKIT_TRAINING_MIN_WORKING_RESERVE_GIB", "1.5"))
        max_working_reserve_gib = float(_env("AI_TOOLKIT_TRAINING_MAX_WORKING_RESERVE_GIB", "3.0"))
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
        )
        if diagnostics is None:
            return None
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
        # Margin-to-spill is the "available VRAM" the deadband governs. Computed
        # by the shared pure helper so the offline simulator measures identically.
        min_device_free_gib = cls._available_vram_gib(
            diagnostics["device_total_gb"],
            diagnostics["device_used_gb"],
            diagnostics["torch_reserved_gb"],
            diagnostics["peak_reserved_gb"],
            safety_gib=safety_gib,
        )
        working_gib = max(0.0, diagnostics["working_reserve_used_gb"])
        allocation_peak_gib = max(
            diagnostics["peak_allocated_gb"]
            - diagnostics["planned_resident_gb"]
            - diagnostics["planned_ring_gb"],
            0.0,
        )
        measured_peak_gib = max(working_gib, allocation_peak_gib)

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
        if did_oom:
            action = "oom_retreat"
            new_working_reserve = current_gib + retreat_gib
            state["danger_working_reserve_gib"] = max(
                state.get("danger_working_reserve_gib") or 0.0, current_gib
            )
        else:
            # Worst-case resolution governs the reserve: a quiet low-res step must
            # not shrink the budget below what the highest-res bucket peaked at.
            governing_reserve_signal_gib = max(
                [working_reserve_signal_gib]
                + [b.get("peak_working_gib", 0.0) for b in state["buckets"].values()]
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
            state["danger_working_reserve_gib"] = danger

        new_working_reserve = min(max(new_working_reserve, min_working_reserve_gib), max_working_reserve_gib)

        # Govern layout moves off a free-VRAM deadband (user spec). Use the
        # worst-case *recent* free margin across resolution buckets so high-res
        # safety binds, but a transient dip does not latch the controller low.
        governing_free_gib = min(
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
        move = cls._training_layout_action(
            governing_free_gib,
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
        # WDDM safety demotion wins over prefetch repair. Otherwise repair a bad
        # prefetch schedule even while the memory deadband is holding steady.
        if move == "down":
            changed_layers = cls._demote_training_layers(
                module, mm, retreat_layers, largest=True
            )
            state["stopped"] = False
            layout_action = "demote" if changed_layers else "demote_unavailable"
        elif prefetch_recovery_action is not None:
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
            cadence_ready = last_step < 0 or step_index - last_step >= promote_interval
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
                )
                if changed_layers:
                    state["last_step"] = step_index
                    if bucket.get("no_improve", 0) >= 2:
                        state["stopped"] = True
                elif layout_action in ("no_offloaded_layers", "stop_line"):
                    state["stopped"] = True
            elif grew_prefetch:
                pass
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
                f"working_reserve={result['working_reserve_gb']:.2f} GiB "
                f"resident={result['resident_gb']:.2f} GiB "
                f"streamed_layers={result['streamed_layers']} "
                f"min_free={min_device_free_gib:.2f} GiB "
                f"peak_working={measured_peak_gib:.2f} GiB "
                f"learned_wddm_hard={state.get('learned_wddm_hard_gib') or 0.0:.2f} GiB"
            )
        return result

    @classmethod
    def _training_cliff_safety_net(cls, module, mm, device=None, *, did_oom=False):
        """Keep manual working_reserve runs off the WDDM spill cliff.

        Manual mode (a fixed ``layer_offloading_smart_working_reserve_gb``) does
        not auto-tune the activation budget, but it must still refuse to fall off
        the cliff. Allocator fragmentation under multi-resolution training can
        ratchet ``reserved`` up until driver-free hits 0 and every step then pays
        the ~5-30x WDDM paging tax for the rest of the run. So at each step
        boundary, if driver-free is below the hard floor we first hand the
        allocator's idle cache back to the driver (``empty_cache`` — usually
        enough on its own, since the idle reserve dwarfs the live tensors), and
        only if that does not restore the floor do we demote the largest resident
        layers until it does. The manual budget number is never changed; this is a
        guard rail, not a tuner.
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
        retreat_layers = max(1, int(_env("AI_TOOLKIT_TRAINING_RETREAT_LAYERS", "3")))
        max_demote = int(_env("AI_TOOLKIT_TRAINING_SAFETY_MAX_DEMOTE", "12"))

        free_gib = torch.cuda.mem_get_info(device)[0] / gib
        if cls._training_cliff_guard_action(
            free_gib, wddm_hard_gib=hard_gib, did_oom=did_oom
        ) == "ok":
            return None  # comfortably inside the budget — no work, no overhead

        before_gib = free_gib
        # 1. Cheapest reclaim first: return idle cached blocks to the driver. Only
        #    genuinely-unused blocks are freed, so live weights/ring/activations
        #    are untouched. This alone undoes a fragmentation blowup.
        try:
            torch.cuda.synchronize(device)
        except RuntimeError:
            pass
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
        free_gib = torch.cuda.mem_get_info(device)[0] / gib

        # 2. Still under the floor => genuinely over-committed for this card.
        #    Demote the largest resident layers until the floor clears or we run
        #    out of demotable layers. _demote_training_layers re-syncs and empties
        #    the cache itself, so the re-measure below is accurate.
        demoted = 0
        while (
            cls._training_cliff_guard_action(
                free_gib, wddm_hard_gib=hard_gib, did_oom=did_oom
            ) == "reclaim"
            and demoted < max_demote
        ):
            changed = cls._demote_training_layers(
                module, mm, retreat_layers, largest=True
            )
            if not changed:
                break
            demoted += changed
            free_gib = torch.cuda.mem_get_info(device)[0] / gib
            did_oom = False  # one reclaim round satisfies the OOM trigger

        action = "demote" if demoted else "empty_cache"
        if cls._diagnostics_enabled():
            print(
                f"[MemoryManager] manual cliff guard: action={action} "
                f"demoted_layers={demoted} "
                f"free={before_gib:.2f}->{free_gib:.2f} GiB "
                f"(hard_floor={hard_gib:.2f} GiB)"
            )
        return {
            "manual_safety": True,
            "action": action,
            "demoted_layers": demoted,
            "device_free_gib": free_gib,
            "device_free_before_gib": before_gib,
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
                free_bytes = torch.cuda.mem_get_info(device)[0]
                if free_bytes - item["resident_bytes"] < stop_bytes:
                    skipped += 1
                    continue
                if cls.promote_layer(item["module"]):
                    promoted += 1
        cls._refresh_training_fp8_flags(root, mm)
        plan = cls._refresh_training_plan_from_layout(root, mm)
        if promoted:
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
        history = cls._historical_prefetch_defaults(registered)

        budget_gib = float(_env(
            "AI_TOOLKIT_BOUNCE_POOL_GIB",
            str(history.get("budget_gib", 5.0)),
        ))
        max_budget_gib = float(_env("AI_TOOLKIT_BOUNCE_MAX_POOL_GIB", "6.0"))
        budget_gib = min(max_budget_gib, budget_gib)
        target_ready_gib = float(_env(
            "AI_TOOLKIT_BOUNCE_TARGET_READY_GIB",
            str(history.get("target_ready_gib", 3.5)),
        ))
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
            f"cold_start_schedule={len(cold_start_schedule)}{history_text}"
        )

    @classmethod
    def training_runtime_diagnostics(
        cls,
        module,
        device=None,
        peak_allocated_override=None,
        peak_reserved_override=None,
    ):
        """Snapshot a smart training layout and its actual runtime memory.

        ``peak_allocated_override`` / ``peak_reserved_override`` (bytes) let the
        caller supply the true within-step peak high-water aggregated across
        gradient accumulations. The live CUDA peak counter is reset per
        accumulation by the trainer's resolution sampler, so on multi-accumulation
        steps it under-reports; pass the step-aggregated peak so the controller
        governs on the real high-water.
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
        ring_bytes = 0
        seen = set()
        for key in ("w_buffers", "b_buffers", "w_grad_buffers", "b_grad_buffers"):
            for tensor in state.get(key, ()):
                if tensor is None or id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                ring_bytes += cls._tensor_storage_bytes(tensor)

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
            0, allocated_bytes - plan["resident_bytes"] - ring_bytes
        )
        working_peak_bytes = max(
            0, peak_allocated_bytes - plan["resident_bytes"] - ring_bytes
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
        # --- Peak-based device footprint (what actually matters for the cliff) --
        # device_used_gb / device_free_gb (memory[2]/[3]) are read at step END --
        # the TROUGH, after the backward graph frees. They overstate free because
        # the activation peak is already gone. The number that governs spill is the
        # WITHIN-STEP peak: our allocator high-water (peak_reserved, which already
        # includes resident weights + ring + activations) plus the non-allocator
        # overhead (CUDA ctx, cuDNN, WDDM/desktop, other apps). That overhead is
        # measured at the trough but is ~constant across the step since only our
        # allocator grows during the forward/backward. resident + ring are inside
        # peak_reserved, so they are never added again.
        peak_reserved_gb = (
            peak_reserved_override
            if peak_reserved_override is not None
            else torch.cuda.max_memory_reserved(device)
        ) / 1024 ** 3
        device_other_gb = max(0.0, memory[2] - memory[1])
        device_used_peak_gb = peak_reserved_gb + device_other_gb
        device_free_peak_gb = max(0.0, memory[4] - device_used_peak_gb)
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
            "live_ring_gb": ring_bytes / 1024 ** 3,
            "pinned_cpu_gb": mm.pinned_weight_bytes / 1024 ** 3,
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
            # PEAK (within-step): the footprint and free margin the spill cliff
            # actually sees. This is what the auto controller governs on.
            "device_used_peak_gb": device_used_peak_gb,
            "device_free_peak_gb": device_free_peak_gb,
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
                    pool.set_schedule(schedule, confidence=confidence)
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
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        try:
            driver_used = torch.cuda.device_memory_used(device)
        except Exception:
            # mem_get_info is still driver-level and includes non-PyTorch users.
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

    @staticmethod
    def _sampling_guard_predicted_peak_free(total_b, free_b, reserved_b, peak_reserved_b):
        """Predicted free VRAM at the next forward's peak (pure, CPU-testable).

        ``other = (total - free) - reserved`` is everything not in our allocator
        (Windows desktop, other apps, CUDA context). ``peak_reserved`` is our
        worst forward's reserved high-water. Their sum is what the next peak will
        occupy; the prediction is ``total`` minus that. It shrinks one-for-one as
        external use grows — which is the cohabitation guard's trigger. Forward-
        only sampling never OOMs at the cliff (it pages silently), so the guard
        watches this instead of waiting for an exception.
        """
        other_b = max(0, (total_b - free_b) - reserved_b)
        return total_b - (peak_reserved_b + other_b)

    @staticmethod
    def _sampling_step_should_trim(free_before_b, trim_margin_b):
        """Whether realized device-free warrants a per-step cache trim (pure).

        WDDM pages on the committed footprint silently, so the trigger is realized
        free, not an allocated-side or peak signal. Trim (empty_cache) is cheap and
        non-destructive, so the bar is just "free has dropped into the margin."
        """
        return free_before_b < trim_margin_b

    @staticmethod
    def _sampling_step_should_demote(free_after_b, hard_floor_b):
        """Whether to escalate to a block demote after a trim (pure).

        Only when trimming left free still under the hard floor — i.e. there was
        no idle cache to reclaim, so the pressure is real (external) and the only
        way down is to stream a resident block. Demotion adds streaming churn, so
        it is the last resort.
        """
        return free_after_b < hard_floor_b

    @classmethod
    def _sampling_demote_largest_block(cls, module, plan, target, *, ignore_modules=None):
        """Stream the largest still-resident sampling block; refresh the plan.

        Returns bytes freed (0 if nothing demotable remains). Shared by the
        setup-time spill-guard and the live per-image cohabitation guard so both
        stay consistent: demote whole blocks largest-first, then re-derive the
        plan's resident/offload accounting from the new layout.
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
            key = cls._offload_group_key(name)
            entry = resident_blocks.setdefault(key, {"layers": [], "bytes": 0})
            entry["layers"].append((name, child))
            entry["bytes"] += cls._direct_module_bytes(child)
        if not resident_blocks:
            return 0
        key = max(resident_blocks, key=lambda k: resident_blocks[k]["bytes"])
        freed = resident_blocks[key]["bytes"]
        for name, child in resident_blocks[key]["layers"]:
            cls.demote_layer(child, mm, layer_key=name)
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
        original_fp8_training_layers = (
            getattr(mm, "_fp8_training_layers", 0) if had_manager else 0
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
                    module._memory_manager._fp8_training_layers = (
                        original_fp8_training_layers
                    )
                    if _OFFLOAD_PREFETCH_ENABLED and args.get("device") is not None:
                        cls._attach_prefetch_pool(module, args["device"])
                    # Sampling detach/restore replaces the streamed layout and
                    # destroys the old pool; any frozen positional trace from
                    # before sampling can now be stale against the restored set.
                    cls.reset_trace_due_to_execution_shape_change()
                if args.get("device") is not None:
                    cls._move_unmanaged_parameters(module, args["device"])
            elif not had_manager:
                cls._move_module_parameters(module, original_device)

        cls.detach(module)
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
        # Sampling working_reserve is configured INDEPENDENTLY from training (the
        # ``working_reserve_gib`` argument, fed from layer_offloading_smart_sampling_
        # working_reserve_gb). Their VRAM profiles are very different: sampling is
        # forward-only, with no optimizer state, gradients, or backward
        # activations to reserve for, so it can run a much smaller reserve.
        # Floor/pad/cold-start stay as env overrides; the selection itself is a
        # pure helper (see _resolve_sampling_working_reserve).
        cold_start_working_reserve = int(
            float(_env("AI_TOOLKIT_SAMPLING_WORKING_RESERVE_GIB", "3.0")) * gib
        )
        working_reserve_floor = int(
            float(_env("AI_TOOLKIT_SAMPLING_WORKING_RESERVE_FLOOR_GIB", "1.5"))
            * gib
        )
        working_reserve_pad = int(
            float(_env("AI_TOOLKIT_SAMPLING_WORKING_RESERVE_PAD_GIB", "0.5")) * gib
        )
        learned_working_reserve = int(getattr(module, "_sampling_peak_working_reserve_bytes", 0))
        working_reserve_bytes, working_reserve_source = cls._resolve_sampling_working_reserve(
            working_reserve_gib,
            learned_working_reserve,
            cold_start_bytes=cold_start_working_reserve,
            floor_bytes=working_reserve_floor,
            pad_bytes=working_reserve_pad,
        )
        wddm_hard_bytes = int(
            float(
                _env("AI_TOOLKIT_SAMPLING_WDDM_HARD_GIB", "1.0")
                if wddm_hard_gib is None
                else wddm_hard_gib
            )
            * gib
        )
        wddm_margin_bytes = int(
            max(
                float(
                    _env("AI_TOOLKIT_SAMPLING_WDDM_MARGIN_GIB", "1.0")
                    if wddm_margin_gib is None
                    else wddm_margin_gib
                ),
                wddm_hard_bytes / gib,
            )
            * gib
        )
        free_bytes = int(after_clear[3] * gib) if after_clear is not None else 0
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
                f"free={free_bytes / gib:.2f} GiB"
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
            wddm_hard_bytes = max(
                int(float(_env("AI_TOOLKIT_SAMPLING_WDDM_HARD_GIB", "1.0")) * gib),
                int(plan.get("wddm_hard_bytes", 0)),
            )
            # Room for the denoising working set (working_reserve) plus the spill cushion.
            wddm_margin_bytes = working_reserve_bytes + wddm_hard_bytes
            free_now = torch.cuda.mem_get_info(target)[0]
            if free_now < wddm_margin_bytes:
                if not hasattr(module, "_memory_manager"):
                    cls.attach(
                        module,
                        target,
                        offload_percent=1.0,
                        ignore_modules=args.get("ignore_modules", []),
                        _offload_module_ids=set(),
                    )
                demoted_blocks = 0
                while free_now < wddm_margin_bytes:
                    freed = cls._sampling_demote_largest_block(
                        module, plan, target,
                        ignore_modules=args.get("ignore_modules", []),
                    )
                    if not freed:
                        break
                    free_now = torch.cuda.mem_get_info(target)[0]
                    demoted_blocks += 1
                if demoted_blocks and diagnostics:
                    floor_status = (
                        "cleared" if free_now >= wddm_margin_bytes else "still low"
                    )
                    print(
                        f"[MemoryManager] spill-guard retreat: demoted "
                        f"{demoted_blocks} resident block(s), floor={floor_status}; "
                        f"device_free={free_now / gib:.2f} GiB "
                        f"(target={wddm_margin_bytes / gib:.2f} GiB)"
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

        def _demote_to_streamed():
            """Mid-denoise OOM recovery (e.g. another process grabbed VRAM).

            The sampling plan is fixed for the whole run, so an external VRAM
            grab can push a denoise step into OOM with no fallback. This moves
            every resident sampling weight to CPU and streams it instead, freeing
            GPU so the caller can retry the failed step. Returns True if it
            transitioned (retry), False if already fully streamed (nothing left
            to free this way -> the caller should re-raise). Updates
            sampling_state so teardown disables the new fp8 forwards.
            """
            if sampling_state["fully_streamed"]:
                return False
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
            guard_margin = int(
                max(
                    float(_env("AI_TOOLKIT_SAMPLING_GUARD_MARGIN_GIB", "0.5")),
                    plan.get("wddm_hard_bytes", 0) / gib,
                )
                * gib
            )
            free_b, total_b = torch.cuda.mem_get_info(target)
            reserved_b = torch.cuda.memory_reserved(target)
            peak_reserved_b = torch.cuda.max_memory_reserved(target)
            # Predicted device-free at the next forward's peak. (peak stats are
            # reset at sampling start, so this reflects only sampling forwards.)
            predicted_peak_free = cls._sampling_guard_predicted_peak_free(
                total_b, free_b, reserved_b, peak_reserved_b
            )
            if predicted_peak_free >= guard_margin:
                return 0
            freed = cls._sampling_demote_largest_block(
                module, plan, target, ignore_modules=args.get("ignore_modules", [])
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
            trim_margin = int(
                max(
                    float(_env("AI_TOOLKIT_SAMPLING_STEP_TRIM_GIB", "1.5")),
                    plan.get("wddm_margin_bytes", 0) / gib,
                )
                * gib
            )
            hard_floor = int(
                max(
                    float(_env("AI_TOOLKIT_SAMPLING_WDDM_HARD_GIB", "1.0")),
                    plan.get("wddm_hard_bytes", 0) / gib,
                )
                * gib
            )
            before = torch.cuda.mem_get_info(target)[0]
            if not cls._sampling_step_should_trim(before, trim_margin):
                return 0
            torch.cuda.empty_cache()
            free_b = torch.cuda.mem_get_info(target)[0]
            freed = max(0, free_b - before)
            sampling_state["trim_count"] += 1
            sampling_state["trim_freed_bytes"] += freed
            # Escalate to demotion only if trimming did not buy back the floor.
            demoted_blocks = 0
            if cls._sampling_step_should_demote(free_b, hard_floor):
                demoted = cls._sampling_demote_largest_block(
                    module, plan, target,
                    ignore_modules=args.get("ignore_modules", []),
                )
                if demoted:
                    demoted_blocks = 1
                    sampling_state["trim_demotes"] += 1
                    if diagnostics:
                        free_b = torch.cuda.mem_get_info(target)[0]
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
            if diagnostics:
                print(
                    f"[MemoryManager] sampling end: current before restore "
                    f"{cls._format_cuda_memory(sample_end)}; restore={time.perf_counter() - restore_started:.2f}s; "
                    f"{cls._format_cuda_memory(cls._cuda_memory(target))}"
                )
