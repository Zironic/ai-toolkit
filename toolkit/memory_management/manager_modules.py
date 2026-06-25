"""
This code was heavily inspired by the work of Lodestone-Rock, pretty much all credit goes
to them. The original code can be found here:
https://github.com/lodestone-rock/RamTorch/blob/main/ramtorch/modules/linear.py

I simply modified it to work with a memory management model and with AI Toolkit's models
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional, Tuple
from torch.overrides import has_torch_function_unary  # (ADD) torchao detection

if TYPE_CHECKING:
    from .manager import MemoryManager

# --- Per-device global state registry ---
_DEVICE_STATE = {}

# How many layers deep to prefetch weights. The old ping-pong used 2 slots, which
# only lets one transfer overlap one compute (1-deep). A deeper ring lets Python
# enqueue several layers ahead so the H2D stream stays saturated instead of
# stalling on a per-layer sync. Override with AI_TOOLKIT_OFFLOAD_DEPTH.
PIPELINE_DEPTH = int(os.environ.get("AI_TOOLKIT_OFFLOAD_DEPTH", "4"))

# Native FP8 sampling counters. Opt-in via MemoryManager.inference_resident(
# fp8_sampling=True); the diagnostics print is gated on "enabled".
_FP8_STATS = {
    "enabled": False,
    "kernel_calls": 0,
    "fallback_calls": 0,
}


def _fp8_stats_enabled():
    return _FP8_STATS["enabled"]


def _get_device_state(device: torch.device):
    """Get or initialize per-device state."""
    if isinstance(device, str):
        device = torch.device(device)

    # CPU path needs no CUDA state
    if device.type != "cuda":
        if device not in _DEVICE_STATE:
            _DEVICE_STATE[device] = {}
        return _DEVICE_STATE[device]

    if device not in _DEVICE_STATE:
        d = max(2, PIPELINE_DEPTH)
        with torch.cuda.device(device):
            _DEVICE_STATE[device] = {
                "depth": d,
                # streams
                "transfer_stream": torch.cuda.Stream(device=device),
                "transfer_grad_stream": torch.cuda.Stream(device=device),
                # forward weight ring: slot_ready = H2D done, slot_free = compute
                # that consumed the slot done (so it can be overwritten).
                "w_buffers": [None] * d,
                "b_buffers": [None] * d,
                "fwd_slot_ready": [torch.cuda.Event() for _ in range(d)],
                "fwd_slot_free": [torch.cuda.Event() for _ in range(d)],
                "forward_clk": 0,
                # backward weight ring (re-fetch for grad-input).
                "w_bwd_buffers": [None] * d,
                "bwd_slot_ready": [torch.cuda.Event() for _ in range(d)],
                "bwd_slot_free": [torch.cuda.Event() for _ in range(d)],
                "backward_clk": 0,
                # backward grad-staging ring (device-side grads -> CPU).
                "w_grad_buffers": [None] * d,
                "b_grad_buffers": [None] * d,
                "grad_compute_done": [torch.cuda.Event() for _ in range(d)],
                "grad_xfer_done": [torch.cuda.Event() for _ in range(d)],
            }
    return _DEVICE_STATE[device]


# ---- ring-buffer staging helpers -----------------------------------------
#
# Each transfer waits only on the event for the *specific slot* it is about to
# overwrite (the compute that used that slot D layers ago), not on a single
# global "compute started" event. With D slots that prior compute is long done,
# so the transfer stream never actually stalls and stays D layers ahead of
# compute. This is the deeper-pipeline + relaxed-dependency change in one.


def _stage_forward_weight(state, device, materialize, weight_cpu, bias_cpu):
    """H2D the next forward weight (+bias) into its ring slot; return (idx, w, b).
    Caller runs compute, then calls _release_forward_slot(state, idx)."""
    d = state["depth"]
    idx = state["forward_clk"]
    state["forward_clk"] = (idx + 1) % d
    ts = state["transfer_stream"]
    with torch.cuda.stream(ts):
        ts.wait_event(state["fwd_slot_free"][idx])
        state["w_buffers"][idx] = materialize(weight_cpu, device)
        state["b_buffers"][idx] = (
            bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
        )
        state["fwd_slot_ready"][idx].record()
    torch.cuda.current_stream().wait_event(state["fwd_slot_ready"][idx])
    return idx, state["w_buffers"][idx], state["b_buffers"][idx]


def _release_forward_slot(state, idx):
    # Slot is reusable once the compute stream finishes the op that read it.
    state["fwd_slot_free"][idx].record()


def _stage_backward_weight(state, device, materialize, weight_cpu):
    """H2D the next backward weight into its ring slot; return (idx, w).
    Caller runs grad-input compute, then _release_backward_weight_slot."""
    d = state["depth"]
    idx = state["backward_clk"]
    state["backward_clk"] = (idx + 1) % d
    ts = state["transfer_stream"]
    with torch.cuda.stream(ts):
        ts.wait_event(state["bwd_slot_free"][idx])
        state["w_bwd_buffers"][idx] = materialize(weight_cpu)
        state["bwd_slot_ready"][idx].record()
    torch.cuda.current_stream().wait_event(state["bwd_slot_ready"][idx])
    return idx, state["w_bwd_buffers"][idx]


def _release_backward_weight_slot(state, idx):
    state["bwd_slot_free"][idx].record()


def _stage_grads_to_cpu(state, idx, grad_w_gpu, grad_b_gpu):
    """Copy freshly-computed device grads (in staging slot idx) to CPU on the
    grad stream, overlapping the next H2D. Returns (grad_w_cpu, grad_b_cpu)."""
    gs = state["transfer_grad_stream"]
    state["grad_compute_done"][idx].record()  # on the compute stream
    grad_w_cpu = grad_b_cpu = None
    with torch.cuda.stream(gs):
        gs.wait_event(state["grad_compute_done"][idx])
        if grad_w_gpu is not None:
            grad_w_cpu = grad_w_gpu.to("cpu", non_blocking=True)
        if grad_b_gpu is not None:
            grad_b_cpu = grad_b_gpu.to("cpu", non_blocking=True)
        state["grad_xfer_done"][idx].record()
    return grad_w_cpu, grad_b_cpu


# (ADD) detect torchao wrapper tensors
def _is_ao_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    if t is None:
        return False
    try:
        if has_torch_function_unary(t):
            return t.__class__.__module__.startswith("torchao.")
    except Exception:
        pass
    for attr in (
        "_scale",
        "_scales",
        "_zero_point",
        "_zp",
        "_block_size",
        "_group_size",
        "_pack_dim",
    ):
        if hasattr(t, attr):
            return True
    return False


def _is_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    if t is None:
        return False
    # torch quantized tensors
    try:
        if torch.is_quantized(t):  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    # (ADD) torchao quantized wrappers
    if _is_ao_quantized_tensor(t):
        return True
    # packed/int formats (weight-only)
    return not t.dtype.is_floating_point


def fp8_rowwise_qdata_scale(weight):
    """Extract (qdata, scale) from a TorchAO rowwise float8 weight, or (None, None).

    TorchAO has moved the fp8 storage around between releases: newer versions
    expose a ``Float8Tensor`` with ``.qdata``/``.scale`` directly, while older
    ones (e.g. the 0.10 line) wrap an ``AffineQuantizedTensor`` whose
    ``tensor_impl`` holds ``float8_data`` + ``scale``. Rather than special-case
    each class, discover the fp8 bytes and per-row scale through the stable
    tensor-subclass ``__tensor_flatten__`` protocol: the bytes are the 2D
    float8_e4m3fn leaf and the scale is the floating leaf sized to the output
    rows (or a scalar). Non-float8 weights return (None, None) so the caller
    falls back to the ordinary forward.
    """
    # Fast path: explicit attributes on newer torchao Float8Tensor.
    qd = getattr(weight, "qdata", None)
    sc = getattr(weight, "scale", None)
    if (
        isinstance(qd, torch.Tensor)
        and qd.dtype == torch.float8_e4m3fn
        and isinstance(sc, torch.Tensor)
    ):
        return qd, sc

    qdata = None
    scales = []

    def _walk(t):
        nonlocal qdata
        flatten = getattr(t, "__tensor_flatten__", None)
        if flatten is None:
            return
        try:
            names, _ = flatten()
        except Exception:
            return
        for name in names:
            inner = getattr(t, name, None)
            if inner is None:
                continue
            if hasattr(inner, "__tensor_flatten__"):
                _walk(inner)
            elif isinstance(inner, torch.Tensor):
                if inner.dtype == torch.float8_e4m3fn and inner.ndim == 2:
                    qdata = inner
                elif inner.dtype in (torch.float32, torch.bfloat16, torch.float16):
                    scales.append(inner)

    _walk(weight)
    if qdata is None:
        return None, None
    out_rows = qdata.shape[0]
    for scale in scales:
        if scale.numel() == out_rows or scale.numel() == 1:
            return qdata, scale
    return None, None


def fp8_linear_inference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Native FP8 GEMM for TorchAO rowwise float8 weights on SM89+.

    Ada cannot consume rowwise scales directly in torch._scaled_mm. Compute
    with the raw FP8 bytes, then apply the existing per-output-row scales to
    the bf16/fp16 result. Return None when the op is unsupported so the caller
    falls back to the ordinary (dequantized) forward.
    """
    qdata, scale = fp8_rowwise_qdata_scale(weight)
    if (
        x.device.type != "cuda"
        or x.dtype not in (torch.bfloat16, torch.float16)
        or qdata is None
        or scale is None
    ):
        if _fp8_stats_enabled():
            _FP8_STATS["fallback_calls"] += 1
        return None
    if (
        not hasattr(torch, "_scaled_mm")
        or torch.cuda.get_device_capability(x.device) < (8, 9)
        or qdata.device != x.device
        or scale.device != x.device
        or (bias is not None and bias.device != x.device)
        or qdata.dtype != torch.float8_e4m3fn
        or qdata.ndim != 2
        or scale.numel() != qdata.shape[0]
        or x.shape[-1] != qdata.shape[1]
        or qdata.shape[0] % 16
        or qdata.shape[1] % 16
        or x.numel() == 0
    ):
        if _fp8_stats_enabled():
            _FP8_STATS["fallback_calls"] += 1
        return None

    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1])
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    scale_x = torch.clamp(
        x_2d.abs().amax().float() / fp8_info.max,
        min=torch.finfo(torch.float32).tiny,
    )
    x_fp8 = torch.clamp(
        x_2d / scale_x.to(x_2d.dtype),
        min=fp8_info.min,
        max=fp8_info.max,
    ).to(torch.float8_e4m3fn)
    one = torch.ones((), device=x.device, dtype=torch.float32)
    try:
        out = torch._scaled_mm(
            x_fp8,
            qdata.t(),
            scale_a=scale_x,
            scale_b=one,
            out_dtype=x.dtype,
            use_fast_accum=True,
        )
    except RuntimeError:
        if _fp8_stats_enabled():
            _FP8_STATS["fallback_calls"] += 1
        return None

    if _fp8_stats_enabled():
        _FP8_STATS["kernel_calls"] += 1
    out = out * scale.reshape(1, -1).to(out.dtype)
    if bias is not None:
        out = out + bias.to(device=x.device, dtype=out.dtype)
    return out.reshape(*original_shape[:-1], qdata.shape[0])


def _pin_inner_tensors(t: torch.Tensor) -> None:
    """Pin the leaf storage of a tensor-subclass (e.g. torchao float8) in place.

    Quantized wrappers can't be pin_memory()'d directly, but they expose their
    real data as inner tensors via __tensor_flatten__. Pinning those lets the
    per-layer H2D bounce run async and overlap with compute instead of blocking.
    """
    try:
        names, _ = t.__tensor_flatten__()
    except Exception:
        return
    for name in names:
        inner = getattr(t, name, None)
        if inner is None:
            continue
        if hasattr(inner, "__tensor_flatten__"):
            _pin_inner_tensors(inner)  # recurse: AQT -> tensor_impl -> data/scale
        elif (
            isinstance(inner, torch.Tensor)
            and inner.device.type == "cpu"
            and not inner.is_pinned()
        ):
            try:
                setattr(t, name, inner.pin_memory())
            except Exception:
                pass


def _ensure_cpu_pinned(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if t.device.type != "cpu":
        try:
            t = t.to("cpu", copy=True)
        except Exception:
            t = t.to("cpu")
    # Quantized wrappers can't be pin_memory()'d directly, but pinning their
    # inner storage gives the same async-transfer benefit.
    if _is_quantized_tensor(t):
        if torch.cuda.is_available():
            _pin_inner_tensors(t)
        return t
    if torch.cuda.is_available():
        try:
            t = t.pin_memory()
        except RuntimeError:
            pass
    return t


def _move_params_to_cpu_and_pin(module: nn.Module):
    """Force parameters to CPU (+pinned) so we can 'bounce' them per forward/backward."""
    with torch.no_grad():
        for name in ("weight", "bias"):
            param = getattr(module, name, None)
            if not isinstance(param, nn.Parameter):
                continue
            cpu_data = _ensure_cpu_pinned(param.data).detach()
            if _is_quantized_tensor(param.data):
                # Tensor-subclass weights (e.g. torchao float8 AffineQuantizedTensor)
                # ignore `param.data = ...`: the wrapper reports CPU but its inner
                # storage stays on the GPU, so the weight never actually offloads.
                # Replace the whole Parameter so the device move sticks.
                setattr(
                    module,
                    name,
                    nn.Parameter(cpu_data, requires_grad=param.requires_grad),
                )
            else:
                param.data = cpu_data


# ==========================
# Autograd functions (CUDA)
# ==========================


class _BouncingLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device: torch.device):
        # choose compute dtype to match activations
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_linear_weight(cpu_w, dev):
            if _is_quantized_tensor(cpu_w):
                # move quantized wrapper to GPU -> dequantize on GPU -> cast on GPU
                w_q_gpu = cpu_w.to(dev, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w_gpu = cpu_w.to(dev, non_blocking=True)
            return w_gpu

        if device.type != "cuda":
            out = F.linear(
                x.to("cpu"),
                _materialize_linear_weight(weight_cpu, torch.device("cpu")),
                bias_cpu,
            )
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.device = torch.device("cpu")
            return out.to(x.device)

        state = _get_device_state(device)
        idx, w_gpu, b_gpu = _stage_forward_weight(
            state, device, _materialize_linear_weight, weight_cpu, bias_cpu
        )
        out = F.linear(x, w_gpu, b_gpu)
        _release_forward_slot(state, idx)

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        ctx.target_dtype = target_dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        target_dtype = getattr(ctx, "target_dtype", grad_out.dtype)

        if device.type != "cuda":
            go_cpu = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_mat = (
                weight_cpu.dequantize()
                if _is_quantized_tensor(weight_cpu)
                else weight_cpu
            )
            if w_mat.dtype != target_dtype and target_dtype in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
            ):
                w_mat = w_mat.to(target_dtype)
            grad_input = go_cpu @ w_mat
            grad_weight = (
                go_cpu.flatten(0, -2).T @ x_cpu.flatten(0, -2)
                if getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
                else None
            )
            grad_bias = (
                go_cpu.sum(dim=tuple(range(go_cpu.ndim - 1)))
                if (bias_cpu is not None and getattr(bias_cpu, "requires_grad", False))
                else None
            )
            return grad_input.to(grad_out.device), grad_weight, grad_bias, None

        state = _get_device_state(device)

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_for_bwd(cpu_w):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = cpu_w.to(device, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w = cpu_w.to(device, non_blocking=True)
            return w

        idx, w_bwd = _stage_backward_weight(
            state, device, _materialize_for_bwd, weight_cpu
        )

        # grad wrt input (GPU)
        grad_input = grad_out.to(dtype=target_dtype) @ w_bwd
        _release_backward_weight_slot(state, idx)

        # compute grads if float masters exist (frozen/quantized bases skip this)
        grad_weight = None
        grad_bias = None
        need_w = (
            getattr(weight_cpu, "requires_grad", False)
            and weight_cpu.dtype.is_floating_point
        )
        need_b = bias_cpu is not None and getattr(bias_cpu, "requires_grad", False)
        if need_w or need_b:
            # ensure the prior grad D2H using this staging slot finished
            torch.cuda.current_stream().wait_event(state["grad_xfer_done"][idx])
            w_grad_gpu = b_grad_gpu = None
            if need_w:
                w_grad_gpu = grad_out.flatten(0, -2).T @ x.flatten(0, -2)
                state["w_grad_buffers"][idx] = w_grad_gpu
            if need_b:
                b_grad_gpu = grad_out.sum(dim=tuple(range(grad_out.ndim - 1)))
                state["b_grad_buffers"][idx] = b_grad_gpu
            grad_weight, grad_bias = _stage_grads_to_cpu(
                state, idx, w_grad_gpu, b_grad_gpu
            )

        return grad_input.to(dtype=grad_out.dtype), grad_weight, grad_bias, None


class _BouncingConv2dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight_cpu,
        bias_cpu,
        device: torch.device,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
    ):
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_conv_weight(cpu_w, dev):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = cpu_w.to(dev, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w_gpu = cpu_w.to(dev, non_blocking=True)
            return w_gpu

        if device.type != "cuda":
            out = F.conv2d(
                x.to("cpu"),
                _materialize_conv_weight(weight_cpu, torch.device("cpu")),
                bias_cpu,
                stride,
                padding,
                dilation,
                groups,
            )
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.meta = ("cpu", stride, padding, dilation, groups, target_dtype)
            return out.to(x.device)

        state = _get_device_state(device)
        idx, w_gpu, b_gpu = _stage_forward_weight(
            state, device, _materialize_conv_weight, weight_cpu, bias_cpu
        )
        out = F.conv2d(x, w_gpu, b_gpu, stride, padding, dilation, groups)
        _release_forward_slot(state, idx)

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.meta = (device, stride, padding, dilation, groups, target_dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device, stride, padding, dilation, groups, target_dtype = ctx.meta

        if (
            isinstance(device, torch.device) and device.type != "cuda"
        ) or device == "cpu":
            go = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_cpu = (
                weight_cpu.dequantize()
                if _is_quantized_tensor(weight_cpu)
                else weight_cpu
            )
            if w_cpu.dtype != target_dtype and target_dtype in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
            ):
                w_cpu = w_cpu.to(target_dtype)
            from torch.nn.grad import conv2d_input, conv2d_weight  # type: ignore

            grad_input = conv2d_input(
                x_cpu.shape,
                w_cpu,
                go,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            grad_weight = (
                conv2d_weight(
                    x_cpu,
                    w_cpu.shape,
                    go,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                if getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
                else None
            )
            grad_bias = (
                go.sum(dim=(0, 2, 3))
                if (bias_cpu is not None and getattr(bias_cpu, "requires_grad", False))
                else None
            )
            return (
                grad_input.to(grad_out.device),
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
            )

        state = _get_device_state(device)

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_for_bwd(cpu_w):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = cpu_w.to(device, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w = cpu_w.to(device, non_blocking=True)
            return w

        idx, w_bwd = _stage_backward_weight(
            state, device, _materialize_for_bwd, weight_cpu
        )

        from torch.nn.grad import conv2d_input, conv2d_weight  # type: ignore

        grad_input = conv2d_input(
            x.shape,
            w_bwd,
            grad_out.to(dtype=target_dtype),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        _release_backward_weight_slot(state, idx)

        # Compute heavy grads on GPU into staging buffers (frozen bases skip this)
        grad_weight = None
        grad_bias = None
        need_w = (
            getattr(weight_cpu, "requires_grad", False)
            and weight_cpu.dtype.is_floating_point
        )
        need_b = bias_cpu is not None and getattr(bias_cpu, "requires_grad", False)
        if need_w or need_b:
            torch.cuda.current_stream().wait_event(state["grad_xfer_done"][idx])
            w_grad_gpu = b_grad_gpu = None
            if need_w:
                w_grad_gpu = conv2d_weight(
                    x,
                    weight_cpu.shape,
                    grad_out,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                state["w_grad_buffers"][idx] = w_grad_gpu
            if need_b:
                b_grad_gpu = grad_out.sum(dim=(0, 2, 3))
                state["b_grad_buffers"][idx] = b_grad_gpu
            grad_weight, grad_bias = _stage_grads_to_cpu(
                state, idx, w_grad_gpu, b_grad_gpu
            )

        return (
            grad_input.to(dtype=grad_out.dtype),
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )


class BaseLayerMemoryManager:
    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        self.module: nn.Module = module
        self.manager: "MemoryManager" = manager

    @classmethod
    def attach(cls, module: nn.Module, manager: "MemoryManager"):
        if hasattr(module, "_layer_memory_manager"):
            return
        module._layer_memory_manager = cls(module, manager)

        # mark parameters as memory managed
        for param in module.parameters(recurse=False):
            param._is_memory_managed = True


class LinearLayerMemoryManager(BaseLayerMemoryManager):
    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # 1) Move params to CPU + pin memory for fast H2D
        _move_params_to_cpu_and_pin(self.module)

        # 2) Hijack forward
        if hasattr(self.module, "ara_lora_ref"):
            # ARA, we need to replace the lora forward
            self._original_forward = getattr(self.module.ara_lora_ref(), "org_forward")
        else:
            self._original_forward = getattr(self.module, "forward")

        def _mm_forward(x, *args, **kwargs):
            # ensure we only use expected signature (Linear: x)
            if args or kwargs:
                # fall back to original if a custom signature is used
                return self._original_forward(x, *args, **kwargs)

            weight_cpu = self.module.weight
            bias_cpu = getattr(self.module, "bias", None)
            device = self.manager.process_device

            # NOTE: do NOT move params to device here; autograd fn streams & bounces them
            return _BouncingLinearFn.apply(x, weight_cpu, bias_cpu, device)

        if hasattr(self.module, "ara_lora_ref"):
            self.module.ara_lora_ref().org_forward = _mm_forward
        else:
            self.module.forward = _mm_forward
        
        self.module._memory_management_device = self.manager.process_device


class ConvLayerMemoryManager(BaseLayerMemoryManager):
    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # 1) Move params to CPU + pin memory for fast H2D
        _move_params_to_cpu_and_pin(self.module)

        # Cache static conv attributes from the module
        stride = (
            self.module.stride
            if isinstance(self.module.stride, tuple)
            else (self.module.stride, self.module.stride)
        )
        padding = (
            self.module.padding
            if isinstance(self.module.padding, tuple)
            else (self.module.padding, self.module.padding)
        )
        dilation = (
            self.module.dilation
            if isinstance(self.module.dilation, tuple)
            else (self.module.dilation, self.module.dilation)
        )
        groups = self.module.groups

        # 2) Hijack forward
        if hasattr(self.module, "ara_lora_ref"):
            # ARA, we need to replace the lora forward
            self._original_forward = getattr(self.module.ara_lora_ref(), "org_forward")
        else:
            self._original_forward = getattr(self.module, "forward")

        def _mm_forward(x, *args, **kwargs):
            # Support the typical Conv2d(x) call; if user passes uncommon extras, fallback.
            if args or kwargs:
                return self._original_forward(x, *args, **kwargs)

            weight_cpu = self.module.weight
            bias_cpu = getattr(self.module, "bias", None)
            device = self.manager.process_device

            return _BouncingConv2dFn.apply(
                x, weight_cpu, bias_cpu, device, stride, padding, dilation, groups
            )

        if hasattr(self.module, "ara_lora_ref"):
            self.module.ara_lora_ref().org_forward = _mm_forward
        else:
            self.module.forward = _mm_forward
        
        self.module._memory_management_device = self.manager.process_device
