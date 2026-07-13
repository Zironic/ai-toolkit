"""Arena-owned temporary native-FP8 forward transforms."""

from __future__ import annotations

import torch
from ..fp8_transpose import column_major

LINEAR_MODULES = {"Linear", "LoRACompatibleLinear", "QLinear"}
_FP8_GRAD_INPUT = False


def set_fp8_grad_input_enabled(enabled: bool) -> None:
    global _FP8_GRAD_INPUT
    _FP8_GRAD_INPUT = bool(enabled)


def _qualifies(weight) -> bool:
    qdata = getattr(weight, "qdata", None)
    scale = getattr(weight, "scale", None)
    if qdata is None or scale is None or not hasattr(torch, "_scaled_mm"):
        return False
    if qdata.device.type != "cuda":
        return False
    return not (
        torch.cuda.get_device_capability(qdata.device) < (8, 9)
        or qdata.dtype != torch.float8_e4m3fn
        or qdata.ndim != 2
        or scale.device != qdata.device
        or scale.numel() != qdata.shape[0]
        or qdata.shape[0] % 16
        or qdata.shape[1] % 16
    )


def _fp8_linear(x, qdata_t, scale_row, bias):
    shape = x.shape
    x_2d = x.reshape(-1, shape[-1])
    info = torch.finfo(torch.float8_e4m3fn)
    scale_x = torch.clamp(
        x_2d.abs().amax().float() / info.max,
        min=torch.finfo(torch.float32).tiny,
    )
    x_fp8 = torch.clamp(
        x_2d / scale_x.to(x_2d.dtype), min=info.min, max=info.max
    ).to(torch.float8_e4m3fn)
    one = torch.ones((), device=x.device, dtype=torch.float32)
    out = torch._scaled_mm(
        x_fp8,
        qdata_t,
        scale_a=scale_x,
        scale_b=one,
        out_dtype=x.dtype,
        use_fast_accum=True,
    )
    out = out * scale_row.reshape(1, -1).to(out.dtype)
    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out.reshape(*shape[:-1], scale_row.shape[0])


class _TrainingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qdata_t, scale_row, bias):
        ctx.save_for_backward(qdata_t, scale_row)
        ctx.input_dtype = x.dtype
        ctx.fp8_grad_input = bool(_FP8_GRAD_INPUT)
        return _fp8_linear(x, qdata_t, scale_row, bias)

    @staticmethod
    def backward(ctx, grad_out):
        qdata_t, scale_row = ctx.saved_tensors
        dtype = ctx.input_dtype
        grad_input = None
        qdata = qdata_t.t()
        if ctx.fp8_grad_input:
            try:
                shape = grad_out.shape
                grad = grad_out.reshape(-1, shape[-1]).to(torch.float32)
                grad = grad * scale_row.reshape(1, -1).to(torch.float32)
                info = torch.finfo(torch.float8_e4m3fn)
                scale_grad = torch.clamp(
                    grad.abs().amax() / info.max,
                    min=torch.finfo(torch.float32).tiny,
                )
                grad_fp8 = torch.clamp(
                    grad / scale_grad, min=info.min, max=info.max
                ).to(torch.float8_e4m3fn)
                one = torch.ones(
                    (), device=grad_out.device, dtype=torch.float32
                )
                grad_input = torch._scaled_mm(
                    grad_fp8,
                    column_major(qdata),
                    scale_a=scale_grad,
                    scale_b=one,
                    out_dtype=dtype,
                    use_fast_accum=True,
                ).reshape(*shape[:-1], qdata.shape[1])
            except Exception:
                grad_input = None
        if grad_input is None:
            weight = qdata.to(dtype) * scale_row.reshape(-1, 1).to(dtype)
            grad_input = grad_out.to(dtype) @ weight
        return grad_input.to(grad_out.dtype), None, None, None


def _container(child):
    container, attribute = child, "forward"
    owner_ref = getattr(child, "ara_lora_ref", None)
    owner = owner_ref() if callable(owner_ref) else None
    if owner is None:
        candidate = getattr(getattr(child, "forward", None), "__self__", None)
        if candidate is not None and candidate is not child:
            owner = candidate
    if owner is not None and hasattr(owner, "org_forward"):
        container, attribute = owner, "org_forward"
    return container, attribute


def enable(model, *, include_ids=None, training: bool):
    restores = []
    include_ids = None if include_ids is None else set(include_ids)
    for child in model.modules():
        if child.__class__.__name__ not in LINEAR_MODULES:
            continue
        if include_ids is not None and id(child) not in include_ids:
            continue
        weight = getattr(child, "weight", None)
        if (
            not isinstance(weight, torch.nn.Parameter)
            or weight.requires_grad
            or not _qualifies(weight.data)
        ):
            continue
        container, attribute = _container(child)
        original = getattr(container, attribute)
        qdata_t = weight.data.qdata.t()
        scale_row = weight.data.scale
        bias = getattr(child, "bias", None)

        def installed(x, *args, _qt=qdata_t, _scale=scale_row, _bias=bias, _original=original, **kwargs):
            if args or kwargs:
                return _original(x, *args, **kwargs)
            if training:
                return _TrainingFn.apply(x, _qt, _scale, _bias)
            return _fp8_linear(x, _qt, _scale, _bias)

        setattr(container, attribute, installed)
        restores.append((container, attribute, original, installed))
    return restores


def disable(restores) -> None:
    for container, attribute, original, installed in reversed(restores):
        if getattr(container, attribute, None) is installed:
            setattr(container, attribute, original)
