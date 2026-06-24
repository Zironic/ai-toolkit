import contextlib
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
    _FP8_STATS,
    PIPELINE_DEPTH,
    summarize_offload_profile,
    set_offload_profile_enabled,
    offload_step_begin,
    offload_step_end,
    offload_step_abort,
    offload_trace_report,
    offload_trace_schedule,
    offload_trace_version,
    reset_offload_trace_for_current_step,
    set_offload_trace_enabled,
    set_fp8_grad_input_enabled,
)
from . import bounce_pool
import random

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
_OFFLOAD_PREFETCH_ENABLED = os.environ.get(
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
            float(os.environ.get("AI_TOOLKIT_PINNED_WEIGHT_GIB", default_pin_gib))
            * 1024 ** 3
        )
        self._prefetch_pool = None

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
        # Assign each managed layer a stable identity from its module path. The
        # trace scheduler keys on this rather than id(weight), which would not
        # survive the Parameter replacement that sampling detach/restore does.
        for name, child in module.named_modules():
            if hasattr(child, "_layer_memory_manager"):
                child._mm_layer_key = name or child.__class__.__name__
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

            original_forward = getattr(container, attribute)

            def _fp8_forward(x, *args, _child=child, _original=original_forward, **kwargs):
                if not args and not kwargs:
                    result = fp8_linear_inference(
                        x, _child.weight, getattr(_child, "bias", None)
                    )
                    if result is not None:
                        return result
                return _original(x, *args, **kwargs)

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

    @classmethod
    def _smart_sampling_plan(cls, module, free_bytes, headroom_bytes, ignore_modules):
        """Comfy-style byte budget: resident weights + transfer ring + headroom."""
        candidates = cls._sampling_candidates(module, ignore_modules)
        total_model_bytes = cls._module_bytes(module)
        usable_bytes = max(0, free_bytes - headroom_bytes)
        resident_bytes = total_model_bytes
        offloaded = []

        # Like Comfy's static loader, consider the most expensive modules first.
        # Reserve the largest PIPELINE_DEPTH materialized weights for our async ring.
        for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
            ring_bytes = sum(
                sorted((item[2] for item in offloaded), reverse=True)[:PIPELINE_DEPTH]
            )
            if resident_bytes + ring_bytes <= usable_bytes:
                break
            offloaded.append(candidate)
            resident_bytes -= candidate[1]

        ring_bytes = sum(
            sorted((item[2] for item in offloaded), reverse=True)[:PIPELINE_DEPTH]
        )
        fits = resident_bytes + ring_bytes <= usable_bytes
        return {
            "offload_ids": {id(item[0]) for item in offloaded},
            "offloaded_layers": len(offloaded),
            "resident_bytes": resident_bytes,
            "ring_bytes": ring_bytes,
            "model_bytes": total_model_bytes,
            "headroom_bytes": headroom_bytes,
            "usable_bytes": usable_bytes,
            "fits": fits,
        }

    @classmethod
    def smart_training_plan(
        cls, module, device, headroom_gib=4.0, ignore_modules=None
    ):
        """Choose training-resident layers from a Comfy-style byte budget."""
        ignore_modules = list(ignore_modules or [])
        device = torch.device(device)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        headroom_bytes = int(float(headroom_gib) * 1024 ** 3)
        usable_bytes = max(0, free_bytes - headroom_bytes)
        total_model_bytes = cls._module_bytes(module)
        candidates = [
            (child, resident, cls._training_stream_bytes(child))
            for child, resident, _ in cls._sampling_candidates(
                module, ignore_modules
            )
        ]
        resident_bytes = total_model_bytes
        offloaded = []
        for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
            ring_bytes = sum(
                sorted(
                    (item[2] for item in offloaded), reverse=True
                )[:PIPELINE_DEPTH]
            )
            if resident_bytes + ring_bytes <= usable_bytes:
                break
            offloaded.append(candidate)
            resident_bytes -= candidate[1]
        ring_bytes = sum(
            sorted((item[2] for item in offloaded), reverse=True)[:PIPELINE_DEPTH]
        )
        return {
            "offload_ids": {id(item[0]) for item in offloaded},
            "offloaded_layers": len(offloaded),
            "candidate_layers": len(candidates),
            "model_bytes": total_model_bytes,
            "resident_bytes": resident_bytes,
            "ring_bytes": ring_bytes,
            "headroom_bytes": headroom_bytes,
            "free_bytes": free_bytes,
            "fits": resident_bytes + ring_bytes <= usable_bytes,
        }

    @classmethod
    def attach_smart_training(
        cls, module, device, headroom_gib=4.0, ignore_modules=None,
        fp8_training_forward=False,
    ):
        ignore_modules = list(ignore_modules or [])
        plan = cls.smart_training_plan(
            module, device, headroom_gib, ignore_modules
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
            f"streamed_layers={plan['offloaded_layers']}/{plan['candidate_layers']} "
            f"ring_reserve={plan['ring_bytes'] / gib:.2f} GiB "
            f"training_headroom={plan['headroom_bytes'] / gib:.2f} GiB "
            f"free={plan['free_bytes'] / gib:.2f} GiB "
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
    def _attach_prefetch_pool(cls, module, device):
        """Create a bounce pool for this device and register the streamed
        layers as sources the background workers can pre-pin from."""
        gib = 1024 ** 3
        budget = int(
            float(os.environ.get("AI_TOOLKIT_BOUNCE_POOL_GIB", "4.0")) * gib
        )
        lookahead = int(os.environ.get("AI_TOOLKIT_BOUNCE_LOOKAHEAD", "8"))
        workers = int(os.environ.get("AI_TOOLKIT_BOUNCE_WORKERS", "2"))
        ram_floor = int(
            float(os.environ.get("AI_TOOLKIT_BOUNCE_RAM_FLOOR_GIB", "2.0")) * gib
        )
        pool = bounce_pool.create_pool(
            device,
            budget_bytes=budget,
            lookahead=lookahead,
            num_workers=workers,
            ram_floor_bytes=ram_floor,
        )
        registered = 0
        cold_start_schedule = []
        for child in module.modules():
            key = getattr(child, "_mm_layer_key", None)
            if key is not None and hasattr(child, "_layer_memory_manager"):
                pool.register_source(key, child)
                cold_start_schedule.append(key)
                registered += 1
        # Before the first completed step there is no measured trace. Module
        # registration order is a useful, correctness-safe approximation of
        # the initial forward and avoids making that entire pass pageable. A
        # mismatch merely falls back to demand loading; the frozen real trace
        # replaces this seed on the next step because schedule_version is -1.
        if cold_start_schedule:
            pool.set_schedule(cold_start_schedule)
        module._memory_manager._prefetch_pool = pool
        print(
            f"[MemoryManager] bounce pool attached: device={device} "
            f"budget={budget / gib:.2f} GiB lookahead={lookahead} "
            f"workers={workers} sources={registered} "
            f"cold_start_schedule={len(cold_start_schedule)}"
        )

    @classmethod
    def training_runtime_diagnostics(cls, module, device=None):
        """Snapshot a smart training layout and its actual runtime memory."""
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
        working_bytes = max(
            0, allocated_bytes - plan["resident_bytes"] - ring_bytes
        )
        return {
            "strategy": "smart",
            "managed_layers": sum(
                1 for child in module.modules()
                if hasattr(child, "_layer_memory_manager")
            ),
            "candidate_layers": plan["candidate_layers"],
            "model_gb": plan["model_bytes"] / 1024 ** 3,
            "planned_resident_gb": plan["resident_bytes"] / 1024 ** 3,
            "offloaded_cpu_gb": (
                plan["model_bytes"] - plan["resident_bytes"]
            ) / 1024 ** 3,
            "planned_ring_gb": plan["ring_bytes"] / 1024 ** 3,
            "live_ring_gb": ring_bytes / 1024 ** 3,
            "pinned_cpu_gb": mm.pinned_weight_bytes / 1024 ** 3,
            "training_headroom_gb": plan["headroom_bytes"] / 1024 ** 3,
            "working_headroom_used_gb": working_bytes / 1024 ** 3,
            "headroom_remaining_gb": (
                plan["headroom_bytes"] - working_bytes
            ) / 1024 ** 3,
            "torch_allocated_gb": memory[0],
            "torch_reserved_gb": memory[1],
            "allocator_cached_gb": max(0.0, memory[1] - memory[0]),
            "device_used_gb": memory[2],
            "device_free_gb": memory[3],
            "device_total_gb": memory[4],
            "peak_allocated_gb": (
                torch.cuda.max_memory_allocated(device) / 1024 ** 3
            ),
            "fp8_training_forward_layers": getattr(
                mm, "_fp8_training_layers", 0
            ),
            "peak_reserved_gb": (
                torch.cuda.max_memory_reserved(device) / 1024 ** 3
            ),
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
    def offload_profile_report(reset: bool = False):
        """Return the slice-1 streamed-step timing report, or None if disabled."""
        return summarize_offload_profile(reset=reset)

    @staticmethod
    def offload_step_begin():
        """Mark the start of one streamed training step for the trace recorder,
        and hand the frozen access order to each bounce pool so its workers can
        pre-pin the upcoming layers."""
        offload_step_begin()
        pools = bounce_pool.all_pools()
        if pools:
            schedule = offload_trace_schedule()
            version = offload_trace_version()
            for pool in pools:
                # Refresh when the trace re-records a different (steady) shape,
                # not just on the first freeze.
                if schedule is not None and pool.schedule_version != version:
                    pool.set_schedule(schedule)
                    pool.schedule_version = version
                pool.step_begin()

    @staticmethod
    def offload_step_end():
        """Mark the end of one streamed training step (freeze/replay/validate)."""
        offload_step_end()

    @staticmethod
    def offload_step_abort():
        """Discard the in-flight streamed step's trace (e.g. on OOM)."""
        offload_step_abort()

    @staticmethod
    def reset_offload_trace_for_tuning():
        """Invalidate trace/prefetch after selective-checkpoint policy changes."""
        reset_offload_trace_for_current_step()
        for pool in bounce_pool.all_pools():
            pool.set_schedule([])
            pool.schedule_version = -1
            pool.step_begin()

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
    def set_fp8_grad_input_enabled(enabled: bool):
        set_fp8_grad_input_enabled(enabled)

    @staticmethod
    def offload_prefetch_report(reset: bool = False):
        """Concatenate per-device bounce-pool stats, or None if no pool exists."""
        reports = [pool.report(reset=reset) for pool in bounce_pool.all_pools()]
        return "\n".join(reports) if reports else None

    @staticmethod
    def _diagnostics_enabled():
        value = os.environ.get("AI_TOOLKIT_MEMORY_DIAGNOSTICS", "1").lower()
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

    @classmethod
    @contextlib.contextmanager
    def inference_resident(cls, module, device=None):
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

        def _restore_offload():
            _FP8_STATS["enabled"] = False
            if _FP8_STATS.get("training_enabled", False):
                _FP8_STATS["kernel_calls"] = 0
                _FP8_STATS["fallback_calls"] = 0
            cls._disable_fp8_sampling(module, fp8_restores)
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
        default_headroom = int(
            float(os.environ.get("AI_TOOLKIT_SAMPLING_HEADROOM_GIB", "2.0")) * gib
        )
        learned_headroom = int(
            getattr(module, "_sampling_peak_headroom_bytes", 0)
        )
        headroom_bytes = max(default_headroom, learned_headroom)
        free_bytes = int(after_clear[3] * gib) if after_clear is not None else 0
        plan = cls._smart_sampling_plan(
            module,
            free_bytes,
            headroom_bytes,
            args.get("ignore_modules", []),
        )

        if diagnostics:
            print(
                f"[MemoryManager] smart budget: model={plan['model_bytes'] / gib:.2f} GiB "
                f"resident={plan['resident_bytes'] / gib:.2f} GiB "
                f"streamed_layers={plan['offloaded_layers']} "
                f"transfer_reserve={plan['ring_bytes'] / gib:.2f} GiB "
                f"sampling_headroom={plan['headroom_bytes'] / gib:.2f} GiB "
                f"free={free_bytes / gib:.2f} GiB"
            )

        move_started = time.perf_counter()
        try:
            if not plan["fits"]:
                raise torch.cuda.OutOfMemoryError(
                    "model, streaming buffers, and sampling headroom do not fit"
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

        fp8_resident_layers = fp8_streamed_layers = 0
        fp8_requested = os.environ.get(
            "AI_TOOLKIT_FP8_SAMPLING", "1"
        ).lower() not in ("0", "false", "no", "off")
        fp8_supported = False
        if fp8_requested and target is not None and torch.device(target).type == "cuda":
            major, minor = torch.cuda.get_device_capability(target)
            fp8_supported = (major, minor) >= (8, 9)
        if fp8_supported:
            (
                fp8_restores,
                fp8_resident_layers,
                fp8_streamed_layers,
            ) = cls._enable_fp8_sampling(module)
            _FP8_STATS.update(
                enabled=diagnostics,
                kernel_calls=0,
                fallback_calls=0,
            )

        sampling_mode = (
            f"smart partial ({plan['offloaded_layers']} streamed layers)"
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
                f"{'enabled' if fp8_supported else 'disabled'}; "
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
            # activation/dequant/workspace headroom that sampling actually needed.
            resident_allocated = torch.cuda.memory_allocated(target)
            torch.cuda.reset_peak_memory_stats(target)

        try:
            yield
        finally:
            if cuda_target:
                torch.cuda.synchronize(target)
            sample_end = cls._cuda_memory(target)
            peak_allocated = (
                torch.cuda.max_memory_allocated(target) if cuda_target else 0
            )
            sampling_headroom = max(0, peak_allocated - resident_allocated)
            # Streaming buffers are budgeted separately. Learn only the residual
            # denoising activation/workspace requirement for the next sample.
            observed_working_headroom = max(
                0, sampling_headroom - plan["ring_bytes"]
            )
            previous_headroom = int(
                getattr(module, "_sampling_peak_headroom_bytes", 0)
            )
            module._sampling_peak_headroom_bytes = max(
                previous_headroom, observed_working_headroom
            )
            if diagnostics and cuda_target:
                print(
                    f"[MemoryManager] sampling peak: "
                    f"torch_allocated={peak_allocated / gib:.2f} GiB "
                    f"sampling_extra={sampling_headroom / gib:.2f} GiB "
                    f"learned_headroom={module._sampling_peak_headroom_bytes / gib:.2f} GiB; "
                    f"{cls._format_cuda_memory(sample_end)}"
                )
            if diagnostics and fp8_supported:
                print(
                    f"[MemoryManager] FP8 execution: "
                    f"native_calls={_FP8_STATS['kernel_calls']} "
                    f"fallback_calls={_FP8_STATS['fallback_calls']}"
                )
            restore_started = time.perf_counter()
            _restore_offload()
            if diagnostics:
                print(
                    f"[MemoryManager] sampling end: current before restore "
                    f"{cls._format_cuda_memory(sample_end)}; restore={time.perf_counter() - restore_started:.2f}s; "
                    f"{cls._format_cuda_memory(cls._cuda_memory(target))}"
                )
