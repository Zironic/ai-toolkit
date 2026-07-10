"""Krea 2 (K2) single-stream MMDiT backbone.

Vendored from the reference ``mmdit.py`` for ai-toolkit. This is a single-stream
MMDiT: Qwen3-VL text features are fused by a small ``TextFusionTransformer`` and
then concatenated with the patchified image latent tokens into one sequence that
flows through ``SingleStreamBlock`` layers. The model predicts the flow-matching
velocity on the image tokens.

Differences from the reference (all training-driven, numerically equivalent):
  - ``torch.compile`` decorators are dropped (they fight gradient checkpointing,
    LoRA module swapping and variable shapes during training).
  - Attention uses a plain ``F.scaled_dot_product_attention`` instead of forcing
    the cuDNN SDPA backend, so it works across dtypes / masks / backward.
  - ``enable_gradient_checkpointing`` / ``disable_gradient_checkpointing`` and a
    per-block ``torch.utils.checkpoint`` wrapper are added (gated on
    ``torch.is_grad_enabled()`` so eval/sampling never pays for it).
"""

import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from toolkit.memory_management.ingraph_stream import (
    CompileRegionError,
    IngraphPackError,
    LoraEntry,
    TrainLeaf,
    assemble_leaf_args,
    assert_compile_region_clean,
    block_linear_views,
    block_tensor_views,
    build_block_leaf_plans,
    checkpoint_recompute_context,
    compiled_checkpoint_context,
    configure_fetch_runtime,
    free_on_backward,
    in_recompute,
    is_streamed_module,
    release_pack,
    streamed_linear,
    streamed_linear_tensors,
)


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def _split_heads(x: Tensor, heads: int) -> Tensor:
    b, length, width = x.shape
    return x.reshape(b, length, heads, width // heads).permute(0, 2, 1, 3)


def _merge_heads(x: Tensor) -> Tensor:
    b, heads, length, dim = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, length, heads * dim)


# Debug printout of which SDPA backend the dispatcher will run for each
# distinct attention signature (shape/dtype/mask/gqa/grad). Off by default;
# enable with AI_TOOLKIT_SDPA_DEBUG=1 (diagnostic only -- never wire runtime
# behavior to it). SDPA picks its backend inside the C++ dispatcher, so this
# replays the same decision: walk the priority order, take the first backend
# that is both enabled and eligible for these exact tensors.
_SDPA_DEBUG = os.environ.get("AI_TOOLKIT_SDPA_DEBUG", "0") not in ("", "0")
_SDPA_DEBUG_SEEN: set = set()

# Build-level fact, safe to freeze at import: Windows torch wheels ship
# without the Flash SDPA kernel. attention() uses this to decide whether
# unmasked GQA can stay on Flash's native GQA path or must expand KV heads
# for the memory-efficient backend (a trace-time constant under compile).
try:
    _FLASH_SDP_AVAILABLE = bool(torch.backends.cuda.is_flash_attention_available())
except Exception:
    _FLASH_SDP_AVAILABLE = False


def _sdpa_debug_report(q, k, v, mask, gqa) -> None:
    if q.device.type != "cuda":
        return
    key = (
        tuple(q.shape), tuple(k.shape), str(q.dtype),
        mask is not None, bool(gqa), torch.is_grad_enabled(),
    )
    if key in _SDPA_DEBUG_SEEN:
        return
    _SDPA_DEBUG_SEEN.add(key)
    bc = torch.backends.cuda
    try:
        params = bc.SDPAParams(q, k, v, mask, 0.0, False, bool(gqa))
        eligible = {
            "FLASH_ATTENTION": bc.flash_sdp_enabled()
            and bc.can_use_flash_attention(params, False),
            "EFFICIENT_ATTENTION": bc.mem_efficient_sdp_enabled()
            and bc.can_use_efficient_attention(params, False),
            "CUDNN_ATTENTION": bc.cudnn_sdp_enabled()
            and bc.can_use_cudnn_attention(params, False),
            "MATH": bc.math_sdp_enabled(),
        }
        value_to_name = {
            int(member.value): name
            for name, member in torch.nn.attention.SDPBackend.__members__.items()
        }
        order = [
            value_to_name.get(int(value), str(value))
            for value in torch._C._get_sdp_priority_order()
        ]
        selected = next((name for name in order if eligible.get(name)), "NONE")
    except Exception as error:  # never let diagnostics break a forward
        print(f"[SDPA] backend probe failed: {error}")
        return
    print(
        f"[SDPA] q={tuple(q.shape)} kv={tuple(k.shape)} dtype={q.dtype} "
        f"mask={mask is not None} gqa={bool(gqa)} grad={torch.is_grad_enabled()} "
        f"-> {selected} (eligible: "
        f"{', '.join(n for n in order if eligible.get(n)) or 'none'}; "
        f"priority: {' > '.join(order)})"
    )


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    scale: float | None = None,
    gqa: bool = False,
) -> Tensor:
    # Do not force cuDNN here. For some training sequence shapes its selected
    # plan reserves a multi-GiB workspace (observed as a 9.9 -> 24.4 GiB live
    # allocation spike on a 12 GiB Ada card). Automatic SDPA dispatch can use
    # Flash or another memory-efficient backend and retains the math fallback.
    #
    # But enable_gqa=True disqualifies every fast backend this box has: the
    # memory-efficient (cutlass) backend rejects enable_gqa outright, Flash
    # (which would accept unmasked GQA) is not compiled into Windows torch
    # builds, and torch 2.12's default priority order puts MATH above CUDNN —
    # so dispatch silently falls to the math backend, which materializes the
    # full (B, heads, L, L) score tensor (multiple GiB at sampling
    # resolutions). Expand the KV heads to match Q and drop enable_gqa, so
    # the memory-efficient backend becomes eligible. This is numerically
    # identical to enable_gqa=True: it repeats each KV head across its
    # query-head group. Two cases need it:
    #   * an explicit mask (Flash rejects arbitrary masks everywhere), and
    #   * no Flash in the build (the unmasked GQA path would fall to MATH —
    #     this regressed once when the expansion was gated on mask-only).
    # On builds WITH Flash, unmasked GQA stays on Flash's native GQA path.
    if gqa and k.shape[1] != q.shape[1] and (
        mask is not None or not _FLASH_SDP_AVAILABLE
    ):
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        gqa = False
    # Constant-folds away under dynamo (both operands are trace-time
    # constants), so compiled graphs stay print-free and break-free.
    if _SDPA_DEBUG and not torch.compiler.is_compiling():
        _sdpa_debug_report(q, k, v, mask, gqa)
    x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
    )
    return _merge_heads(x)


def _mask(mask: Tensor) -> Tensor | None:
    """Expand a (B, L) key-padding mask into a broadcast (B, 1, 1, L) attention
    mask, or None when nothing is masked.

    Key-only masking is equivalent to the dense (B, 1, L, L) outer product for
    every output that is actually consumed: pad tokens are excluded as *keys*,
    so they never contribute to real tokens, and the pad tokens' own outputs
    are either masked again downstream or sliced off before use. Dropping the
    query side keeps SDPA's additive-bias materialization at O(L) per sample
    instead of O(L^2) (~85 MB/sample at 1024px), and returning None for an
    all-True mask keeps Flash + enable_gqa dispatch eligible in attention().
    """
    if bool(mask.all()):
        return None
    return mask[:, None, None, :]


def _streamed_arg_linear_sample(
    x: Tensor,
    arg,
    fp8_qualifies: bool,
    lora=None,
) -> Tensor:
    weight, bias, scale = arg
    if lora is None:
        return streamed_linear_tensors(
            x,
            weight,
            bias,
            scale,
            fp8_qualifies=fp8_qualifies,
            training=False,
        )
    lora_a, lora_b, lora_scale = lora
    return streamed_linear_tensors(
        x,
        weight,
        bias,
        scale,
        fp8_qualifies=fp8_qualifies,
        training=False,
        lora_a=lora_a,
        lora_b=lora_b,
        lora_scale=lora_scale,
    )


def _streamed_arg_linear_train(
    x: Tensor,
    arg,
    fp8_qualifies: bool,
    lora=None,
) -> Tensor:
    weight, bias, scale = arg
    if lora is None:
        return streamed_linear_tensors(
            x,
            weight,
            bias,
            scale,
            fp8_qualifies=fp8_qualifies,
            training=True,
        )
    lora_a, lora_b, lora_scale = lora
    return streamed_linear_tensors(
        x,
        weight,
        bias,
        scale,
        fp8_qualifies=fp8_qualifies,
        training=True,
        lora_a=lora_a,
        lora_b=lora_b,
        lora_scale=lora_scale,
    )


def temb(
    t: Tensor,
    dim: int,
    period: float = 1e4,
    tfactor: float = 1e3,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(period)
        * torch.arange(half, dtype=torch.float32, device=device)
        / half
    )
    # t: (B,) -> args: (B, 1, half), so the embedding broadcasts as a per-sample vec.
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int
    tdim: int
    txtdim: int
    heads: int
    multiplier: int
    layers: int
    patch: int
    channels: int
    bias: bool = False
    theta: float = 1e3
    kvheads: int | None = None
    txtlayers: int = 1
    txtheads: int = 20
    txtkvheads: int = 20


class SimpleModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    # vec (b d)
    def forward(self, vec: Tensor):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(6 * dim))

    # vec (b (6 d))
    def forward(self, vec: Tensor):
        out = vec + self.lin
        prescale, preshift, pregate, postscale, postshift, postgate = out.chunk(
            6, dim=-1
        )
        return prescale, preshift, pregate, postscale, postshift, postgate


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims  # how to split the head dimension across the position axes
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [
                rope(pos[..., i], d, self.theta, self.ntk)
                for i, d in enumerate(self.axdims)
            ],
            dim=-3,
        )


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k), v


class RMSNorm(torch.nn.Module):
    def __init__(self, features: int, eps: float = 1e-05, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.zeros(features, device=device, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(
            t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0)
        )
        return t.to(dtype)


class SwiGLU(torch.nn.Module):
    def __init__(
        self, features: int, multiplier: int, bias: bool = False, multiple: int = 128
    ):
        super().__init__()

        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)

        self.gate = torch.nn.Linear(features, mlpdim, bias=bias)
        self.up = torch.nn.Linear(features, mlpdim, bias=bias)
        self.down = torch.nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor, leaves: dict | None = None) -> Tensor:
        if leaves is None:
            return self.down(F.silu(self.gate(x)) * self.up(x))
        gate = streamed_linear(x, leaves["gate"])
        up = streamed_linear(x, leaves["up"])
        return streamed_linear(F.silu(gate) * up, leaves["down"])

    def forward_streamed(
        self,
        x: Tensor,
        gate_arg,
        up_arg,
        down_arg,
        fp8_flags,
        *,
        training: bool = False,
        loras=None,
    ) -> Tensor:
        loras = (None, None, None) if loras is None else loras
        if training:
            gate = _streamed_arg_linear_train(x, gate_arg, fp8_flags[0], loras[0])
            up = _streamed_arg_linear_train(x, up_arg, fp8_flags[1], loras[1])
            return _streamed_arg_linear_train(
                F.silu(gate) * up,
                down_arg,
                fp8_flags[2],
                loras[2],
            )
        gate = _streamed_arg_linear_sample(x, gate_arg, fp8_flags[0], loras[0])
        up = _streamed_arg_linear_sample(x, up_arg, fp8_flags[1], loras[1])
        return _streamed_arg_linear_sample(
            F.silu(gate) * up,
            down_arg,
            fp8_flags[2],
            loras[2],
        )


class Attention(torch.nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads

        self.wq = torch.nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = torch.nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.gqa = self.heads != self.kvheads
        self.wo = torch.nn.Linear(dim, dim, bias=bias)

    def forward(
        self,
        qkv: Tensor,
        freqs: Tensor | None = None,
        mask: Tensor | None = None,
        leaves: dict | None = None,
        ref_span: tuple[int, int] | None = None,
        kv_capture: list | None = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if leaves is None:
            q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)
        else:
            q = streamed_linear(qkv, leaves["wq"])
            k = streamed_linear(qkv, leaves["wk"])
            v = streamed_linear(qkv, leaves["wv"])
            gate = streamed_linear(qkv, leaves["gate"])

        q, k, v = (
            _split_heads(q, self.heads),
            _split_heads(k, self.kvheads),
            _split_heads(v, self.kvheads),
        )

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        if kv_capture is not None and ref_span is not None:
            kv_capture.append(
                (
                    k[:, :, ref_span[0] : ref_span[1]].clone(),
                    v[:, :, ref_span[0] : ref_span[1]].clone(),
                )
            )
        if kv_cache is not None:
            k = torch.cat((k, kv_cache[0]), dim=2)
            v = torch.cat((v, kv_cache[1]), dim=2)
        out = attention(q, k, v, mask=mask, gqa=self.gqa) * F.sigmoid(gate)
        if leaves is None:
            out = self.wo(out)
        else:
            out = streamed_linear(out, leaves["wo"])

        return out

    def forward_streamed(
        self,
        qkv: Tensor,
        freqs: Tensor | None,
        mask: Tensor | None,
        wq_arg,
        wk_arg,
        wv_arg,
        gate_arg,
        wo_arg,
        fp8_flags,
        *,
        training: bool = False,
        loras=None,
    ) -> Tensor:
        loras = (None, None, None, None, None) if loras is None else loras
        if training:
            q = _streamed_arg_linear_train(qkv, wq_arg, fp8_flags[0], loras[0])
            k = _streamed_arg_linear_train(qkv, wk_arg, fp8_flags[1], loras[1])
            v = _streamed_arg_linear_train(qkv, wv_arg, fp8_flags[2], loras[2])
            gate = _streamed_arg_linear_train(qkv, gate_arg, fp8_flags[3], loras[3])
        else:
            q = _streamed_arg_linear_sample(qkv, wq_arg, fp8_flags[0], loras[0])
            k = _streamed_arg_linear_sample(qkv, wk_arg, fp8_flags[1], loras[1])
            v = _streamed_arg_linear_sample(qkv, wv_arg, fp8_flags[2], loras[2])
            gate = _streamed_arg_linear_sample(qkv, gate_arg, fp8_flags[3], loras[3])

        q, k, v = (
            _split_heads(q, self.heads),
            _split_heads(k, self.kvheads),
            _split_heads(v, self.kvheads),
        )

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        out = attention(q, k, v, mask=mask, gqa=self.gqa) * F.sigmoid(gate)
        if training:
            return _streamed_arg_linear_train(out, wo_arg, fp8_flags[4], loras[4])
        return _streamed_arg_linear_sample(out, wo_arg, fp8_flags[4], loras[4])


class LastLayer(torch.nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = torch.nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        x = self.linear(x)
        return x


class TextFusionBlock(torch.nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))

        return x


class TextFusionTransformer(torch.nn.Module):
    # num_txt_layers is the number of selected encoder hidden-state layers fed in
    # (projected down to 1), NOT the transformer depth — that's fixed at 2 + 2 blocks.
    def __init__(
        self,
        num_txt_layers: int,
        txt_dim: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.layerwise_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )
        self.projector = torch.nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        # Collapse to 3D for the projector: a quantized (quanto) Linear's matmul
        # kernel only accepts 2D/3D activations, and this layer-axis projection
        # (n -> 1) otherwise feeds it a 4D (b, l, d, n) tensor.
        x = self.projector(x.reshape(b * l, d, n))
        x = x.reshape(b, l, d)

        for block in self.refiner_blocks:
            x = block(x, mask=mask)

        return x


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        freqs: Tensor,
        mask: Tensor | None = None,
        leaves: dict | None = None,
        ref_span: tuple[int, int] | None = None,
        kv_capture: list | None = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        attn_kwargs = dict(ref_span=ref_span, kv_capture=kv_capture, kv_cache=kv_cache)
        # ``vec`` is the (B, 1, 6*features) modulation input, or a tuple
        # ``(vec, refvec, split)`` for reference-image conditioning: tokens
        # ``[:split]`` (text + noisy image) are modulated with ``vec`` while
        # tokens ``[split:]`` (clean reference tokens) use ``refvec`` built from
        # t=0 (ComfyUI Kontext "index_timestep_zero"). Applied per span rather
        # than materializing a per-token (B, L, 6*features) tensor.
        if isinstance(vec, tuple):
            vec, refvec, split = vec
            m = self.mod(vec)
            r = self.mod(refvec)

            def mod(h, scale, shift):
                return torch.cat(
                    (
                        (1 + m[scale]) * h[:, :split] + m[shift],
                        (1 + r[scale]) * h[:, split:] + r[shift],
                    ),
                    dim=1,
                )

            def gate(h, g):
                return torch.cat((m[g] * h[:, :split], r[g] * h[:, split:]), dim=1)

            x = x + gate(
                self.attn(mod(self.prenorm(x), 0, 1), freqs, mask, **attn_kwargs), 2
            )
            x = x + gate(self.mlp(mod(self.postnorm(x), 3, 4)), 5)
            return x

        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        attn_leaves = None if leaves is None else leaves["attn"]
        mlp_leaves = None if leaves is None else leaves["mlp"]
        x = x + pregate * self.attn(
            (1 + prescale) * self.prenorm(x) + preshift,
            freqs,
            mask,
            leaves=attn_leaves,
            **attn_kwargs,
        )
        x = x + postgate * self.mlp(
            (1 + postscale) * self.postnorm(x) + postshift,
            leaves=mlp_leaves,
        )

        return x

    def forward_streamed(
        self,
        x: Tensor,
        vec: Tensor,
        freqs: Tensor,
        mask: Tensor | None,
        leaf_args,
        fp8_flags,
        *,
        training: bool = False,
        loras=None,
    ) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        loras = (None,) * 8 if loras is None else loras
        x = x + pregate * self.attn.forward_streamed(
            (1 + prescale) * self.prenorm(x) + preshift,
            freqs,
            mask,
            leaf_args[0],
            leaf_args[1],
            leaf_args[2],
            leaf_args[3],
            leaf_args[4],
            fp8_flags[:5],
            training=training,
            loras=loras[:5],
        )
        x = x + postgate * self.mlp.forward_streamed(
            (1 + postscale) * self.postnorm(x) + postshift,
            leaf_args[5],
            leaf_args[6],
            leaf_args[7],
            fp8_flags[5:],
            training=training,
            loras=loras[5:],
        )

        return x


class SingleStreamDiT(nn.Module):
    def __init__(self, config: SingleMMDiTConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        # Number of trailing blocks to leave uncheckpointed (activations kept
        # resident, no backward recompute). Trading a little VRAM for fewer
        # recompute-forward kernel launches once the step is launch-bound.
        self._checkpoint_keep_last = 0
        # Populated by enable_compiled_sampling(): a per-block list where each
        # entry is either a compiled block or None (run that block eager).
        # A block is left eager when it still carries MemoryManager offload
        # hooks (_BouncingLinearFn), which must not be traced by torch.compile.
        self._compiled_blocks: list | None = None
        # Fingerprint of which block indices were hook-free at compile time, so
        # we can detect when inference_resident changed the residency layout and
        # rebuild instead of running stale (guard-churning) compiled blocks.
        self._compiled_fingerprint: tuple | None = None
        self._compiled_training_blocks: list | None = None
        self._compiled_training_fingerprint: tuple | None = None
        self._compiled_training_fp8_restores: list = []
        self._compiled_training_lora_restores: list = []
        self._ingraph_sampling_packs: dict[int, object] = {}
        self._ingraph_sampling_plans: dict[int, object] = {}
        self._ingraph_sampling_loras: dict[int, dict] = {}
        self._ingraph_sampling_lora_leaf_count = None
        # One live scalar per model, shared by the sampling and training trunks
        # (they never coexist). Identity is stable so torch.compile does not
        # recompile when generate_images reassigns network.multiplier per image.
        self._ingraph_lora_multiplier = None
        self._ingraph_lora_multiplier_value = None
        self._ingraph_lora_network = None
        self._ingraph_sampling_restores: list = []
        self._ingraph_unavailable_reasons: tuple[str, ...] = ()
        self._ingraph_sampling_depth = 2
        self._compiled_ingraph_sampling = None
        self._compiled_ingraph_sampling_blocks: dict[int, object] = {}
        self._compiled_ingraph_fingerprint: tuple | None = None
        self._ingraph_sampling_measure = False
        self._ingraph_training_plans: dict[int, object] = {}
        self._ingraph_training_loras: dict[int, dict] = {}
        self._ingraph_training_restores: list = []
        self._ingraph_training_block_fns: list = []
        self._compiled_ingraph_training = None
        self._ingraph_sampling_timing = None
        self._last_ingraph_sampling_timing = None

        headdim = config.features // config.heads
        axes = [
            headdim - 12 * (headdim // 16),
            6 * (headdim // 16),
            6 * (headdim // 16),
        ]
        assert sum(axes) == headdim, f"sum(axes) = {sum(axes)}, headdim = {headdim}"
        assert all(a % 2 == 0 for a in axes), f"axes = {axes}"

        self.posemb = PositionalEncoding(
            config.features, axes, theta=config.theta, ntk=1.0
        )
        self.first = nn.Linear(
            config.channels * config.patch**2, config.features, bias=True
        )

        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    config.features,
                    config.heads,
                    config.multiplier,
                    config.bias,
                    config.kvheads,
                )
                for _ in range(config.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers,
            config.txtdim,
            config.txtheads,
            config.multiplier,
            config.bias,
            config.txtkvheads,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch, config.channels)

        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, keep_last: int = 0):
        self.gradient_checkpointing = True
        self._checkpoint_keep_last = max(0, int(keep_last))

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self._checkpoint_keep_last = 0

    @staticmethod
    def _block_compile_safe(block) -> bool:
        """True if no submodule streams weights.

        Only the ``_layer_memory_manager`` streaming hook (_BouncingLinearFn, a
        device-mutating custom autograd function) forces a block to stay eager.
        Resident layers are compile-safe — including native FP8 sampling layers,
        whose forward (_fp8_linear_compiled) is a pure torch._scaled_mm path with
        all validation hoisted to install time. Checked per block after
        inference_resident() has set residency.
        """
        return SingleStreamDiT._block_compile_safe_for(block, training=False)

    @staticmethod
    def _block_compile_reject_reasons(block, *, training: bool = False) -> list[str]:
        """Return why a block is not a graph-clean compile candidate.

        The sampler can lower resident FP8 Linear layers to a pure forward-only
        ``_scaled_mm`` closure. Training cannot use that path blindly because it
        also needs a correct grad-input path, so resident FP8 tensor subclasses
        remain a blocker until a grad-safe lowering is installed.
        """
        reasons: list[str] = []
        for sub in block.modules():
            if hasattr(sub, "_layer_memory_manager"):
                reasons.append("streaming_hook")
            if (
                bool(getattr(sub, "_forward_pre_hooks", None))
                or bool(getattr(sub, "_forward_hooks", None))
                or bool(getattr(sub, "_forward_hooks_with_kwargs", None))
            ):
                reasons.append("hook_present")
            if hasattr(sub, "_memory_management_device"):
                reasons.append("memory_management_marker")
        if training:
            for sub in block.modules():
                if isinstance(sub, torch.nn.Conv2d):
                    reasons.append("conv_layer")
                    break
            for sub in block.modules():
                weight = getattr(sub, "weight", None)
                if (
                    isinstance(weight, torch.nn.Parameter)
                    and weight.requires_grad
                    and getattr(weight, "dtype", None) is not None
                    and weight.dtype.is_floating_point
                ):
                    reasons.append("trainable_base_weight")
                    break
            for sub in block.modules():
                ready = getattr(sub, "_memory_management_compile_fast_lora_ready", None)
                if ready is None:
                    continue
                try:
                    if not ready():
                        reasons.append("lora_untraceable")
                        break
                except Exception:
                    reasons.append("lora_untraceable")
                    break
            for sub in block.modules():
                weight = getattr(sub, "weight", None)
                if (
                    isinstance(weight, torch.nn.Parameter)
                    and hasattr(weight.data, "qdata")
                    and getattr(weight.data.qdata, "dtype", None) == torch.float8_e4m3fn
                    and not getattr(sub, "_memory_management_training_compile_fp8", False)
                ):
                    reasons.append("resident_fp8_tensor_subclass")
                    break
        return sorted(set(reasons))

    @classmethod
    def _block_compile_safe_for(cls, block, *, training: bool = False) -> bool:
        return len(cls._block_compile_reject_reasons(block, training=training)) == 0

    def training_compile_readiness(self, pinned_keys: set[str] | None = None):
        """Classify permanent-resident blocks before enabling training compile.

        This is intentionally only a diagnostic gate. A block is training-ready
        only if it is permanent, uncheckpointed by construction, has no streaming
        hooks, and does not call an unlowered FP8 tensor-subclass forward.
        """
        if pinned_keys is None:
            mm = getattr(self, "_memory_manager", None)
            pinned_keys = set(getattr(mm, "_training_pinned_resident_keys", set()))
        else:
            pinned_keys = set(pinned_keys)
        statuses = []
        ready = 0
        pinned = 0
        for i, block in enumerate(self.blocks):
            key = f"blocks.{i}"
            is_pinned = key in pinned_keys
            reasons = [] if not is_pinned else self._block_compile_reject_reasons(
                block, training=True
            )
            if is_pinned:
                pinned += 1
                if not reasons:
                    ready += 1
            statuses.append(
                {
                    "index": i,
                    "key": key,
                    "pinned": is_pinned,
                    "ready": bool(is_pinned and not reasons),
                    "reasons": reasons,
                }
            )
        return {
            "pinned_blocks": pinned,
            "ready_blocks": ready,
            "blocked_blocks": pinned - ready,
            "statuses": statuses,
        }

    def enable_compiled_sampling(self):
        """Compile the hook-free SingleStreamBlocks for sampler-only use.

        Call this from inside the sampling (inference_resident) context, after
        residency is set. Blocks that are fully GPU-resident (no streaming hook)
        are compiled; blocks that still stream weights are left eager. The
        result is keyed by a fingerprint of which blocks were clean, so a later
        sampling session with a different residency layout rebuilds instead of
        replaying stale compiled blocks (which would guard-churn or, worse,
        trace a hook that got re-attached).

        `dynamic=False` matches every other compile family here (in-graph
        sampling blocks, training blocks/trunk). Dynamo pins each cache entry
        to its backend via a BACKEND_MATCH guard that compares backends by
        (mode, options, dynamic) equality, so a mismatched `dynamic` makes
        shared code objects traced under both families (RMSNorm.forward,
        _split_heads) duplicate their compiled entries instead of reusing
        them. Static shapes are cheap because the trunk pads sequences to a
        multiple of 256; a genuinely new sample resolution recompiles, same
        as the in-graph blocks. The caller runs the invocation under
        `torch.compiler.set_stance("eager_then_compile")` so the very first
        call is not a wasted compile.

        Returns (compiled_count, eager_count).
        """
        streamed = set(getattr(self, "_ingraph_sampling_plans", {}) or {})
        clean = tuple(
            i for i, block in enumerate(self.blocks)
            if i not in streamed and self._block_compile_safe(block)
        )
        if (
            self._compiled_blocks is not None
            and self._compiled_fingerprint == clean
        ):
            return len(clean), len(self.blocks) - len(clean)

        compiled: list = [None] * len(self.blocks)
        for i in clean:
            compiled[i] = torch.compile(
                self.blocks[i],
                fullgraph=False,
                dynamic=False,
                mode="default",
            )
        self._compiled_blocks = compiled
        self._compiled_fingerprint = clean
        return len(clean), len(self.blocks) - len(clean)

    def disable_compiled_sampling(self):
        self._compiled_blocks = None
        self._compiled_fingerprint = None

    @staticmethod
    def _block_linear_entries(block):
        return (
            ("attn.wq", block.attn.wq),
            ("attn.wk", block.attn.wk),
            ("attn.wv", block.attn.wv),
            ("attn.gate", block.attn.gate),
            ("attn.wo", block.attn.wo),
            ("mlp.gate", block.mlp.gate),
            ("mlp.up", block.mlp.up),
            ("mlp.down", block.mlp.down),
        )

    @staticmethod
    def _nest_block_leaves(flat, pack):
        views = block_linear_views(flat, pack)
        return {
            "attn": {
                "wq": views["attn.wq"],
                "wk": views["attn.wk"],
                "wv": views["attn.wv"],
                "gate": views["attn.gate"],
                "wo": views["attn.wo"],
            },
            "mlp": {
                "gate": views["mlp.gate"],
                "up": views["mlp.up"],
                "down": views["mlp.down"],
            },
        }

    @staticmethod
    def _clear_module_forward_hooks(module):
        saved = []
        for attr in ("_forward_pre_hooks", "_forward_hooks", "_forward_hooks_with_kwargs"):
            hooks = getattr(module, attr, None)
            if hooks:
                saved.append((module, attr, hooks.copy()))
                hooks.clear()
        return saved

    @staticmethod
    def _restore_forward_hooks(saved):
        for module, attr, hooks in reversed(saved):
            current = getattr(module, attr, None)
            if current is not None:
                current.clear()
                current.update(hooks)

    def _strip_ingraph_compile_contaminants(self, streamed_blocks):
        restores = []
        for index in streamed_blocks:
            block = self.blocks[index]
            hook_state = []
            for child in block.modules():
                hook_state.extend(self._clear_module_forward_hooks(child))
                lmm = getattr(child, "_layer_memory_manager", None)
                if lmm is None:
                    continue
                original_forward = getattr(lmm, "_original_forward", None)
                container = getattr(lmm, "_forward_container", None)
                attribute = getattr(lmm, "_forward_attribute", None)
                managed_forward = None
                if container is not None and attribute is not None:
                    managed_forward = getattr(container, attribute, None)
                if original_forward is not None:
                    if container is child and attribute == "forward" and "forward" in child.__dict__:
                        del child.__dict__["forward"]
                    elif container is not None and attribute is not None:
                        setattr(container, attribute, original_forward)
                    else:
                        managed_forward = getattr(child, "forward", None)
                        child.forward = original_forward
                # When a LoRA is attached, the manager installs its streaming
                # forward into the LoRA's org_forward slot (_capture_base_forward),
                # so child.forward is still the LoRA hijack and managed_forward
                # above captured org_forward instead. The blanket delete below
                # would drop that hijack with nothing recording it: a
                # disable -> enable cycle (i.e. a sampling boundary) would strip
                # every LoRA off the model, sample the bare base, and then build
                # a trunk of pure frozen math whose loss has no grad_fn. Save it.
                orphaned_forward = None
                if container is not None and container is not child:
                    orphaned_forward = child.__dict__.get("forward")
                if "forward" in child.__dict__:
                    del child.__dict__["forward"]
                saved_attrs = {}
                for attr in (
                    "_layer_memory_manager",
                    "_memory_management_device",
                    "_memory_management_fp8_sampling",
                    "_memory_management_fp8_training",
                ):
                    if hasattr(child, attr):
                        saved_attrs[attr] = getattr(child, attr)
                        delattr(child, attr)
                restores.append(
                    (
                        child,
                        lmm,
                        container,
                        attribute,
                        managed_forward,
                        saved_attrs,
                        orphaned_forward,
                    )
                )
            if hook_state:
                restores.append(("hooks", hook_state))
        return restores

    @staticmethod
    def _restore_ingraph_compile_contaminants(restores):
        for item in reversed(restores):
            if item and item[0] == "hooks":
                SingleStreamDiT._restore_forward_hooks(item[1])
                continue
            (
                child,
                lmm,
                container,
                attribute,
                managed_forward,
                saved_attrs,
                orphaned_forward,
            ) = item
            for attr, value in saved_attrs.items():
                setattr(child, attr, value)
            if managed_forward is not None:
                if container is not None and attribute is not None:
                    setattr(container, attribute, managed_forward)
                else:
                    child.forward = managed_forward
            # Reinstall the LoRA hijack the strip deleted, after org_forward is
            # back, so the eager path routes through the LoRA again.
            if orphaned_forward is not None:
                child.forward = orphaned_forward

    def reset_ingraph_sampling_timing(self):
        self._ingraph_sampling_timing = {
            "calls": 0,
            "cold_wall_s": None,
            "steady_calls": 0,
            "steady_wall_total_s": 0.0,
            "steady_wall_min_s": None,
            "steady_wall_max_s": None,
        }
        self._last_ingraph_sampling_timing = None

    def _record_ingraph_sampling_timing(self, elapsed_s: float):
        stats = self._ingraph_sampling_timing
        if stats is None:
            self.reset_ingraph_sampling_timing()
            stats = self._ingraph_sampling_timing
        stats["calls"] += 1
        if stats["cold_wall_s"] is None:
            stats["cold_wall_s"] = float(elapsed_s)
            return
        stats["steady_calls"] += 1
        stats["steady_wall_total_s"] += float(elapsed_s)
        if stats["steady_wall_min_s"] is None or elapsed_s < stats["steady_wall_min_s"]:
            stats["steady_wall_min_s"] = float(elapsed_s)
        if stats["steady_wall_max_s"] is None or elapsed_s > stats["steady_wall_max_s"]:
            stats["steady_wall_max_s"] = float(elapsed_s)

    def ingraph_sampling_timing_snapshot(self):
        stats = self._ingraph_sampling_timing
        if not stats:
            return None
        steady_calls = int(stats["steady_calls"])
        steady_avg = (
            stats["steady_wall_total_s"] / steady_calls
            if steady_calls
            else None
        )
        estimate_compile = None
        if stats["cold_wall_s"] is not None and steady_avg is not None:
            estimate_compile = max(0.0, stats["cold_wall_s"] - steady_avg)
        return {
            "calls": int(stats["calls"]),
            "cold_compile_plus_first_forward_s": stats["cold_wall_s"],
            "steady_forward_calls": steady_calls,
            "steady_forward_avg_s": steady_avg,
            "steady_forward_min_s": stats["steady_wall_min_s"],
            "steady_forward_max_s": stats["steady_wall_max_s"],
            "compile_time_estimate_s": estimate_compile,
        }

    def ingraph_streamed_block_indices(self):
        indices = []
        for index, block in enumerate(self.blocks):
            if any(hasattr(child, "_layer_memory_manager") for child in block.modules()):
                indices.append(index)
        return tuple(indices)

    def enable_ingraph_sampling(self, streamed_blocks=None, depth: int = 2, compile: bool = False):
        """Enable phase-3 in-graph sampling for selected Krea2 blocks."""
        self.disable_ingraph_sampling()
        for handle in getattr(self, "_mm_block_stream_handles", []) or []:
            try:
                handle.remove()
            except Exception:
                pass
        self._mm_block_stream_handles = []
        if streamed_blocks is None:
            streamed_blocks = range(len(self.blocks))
        streamed_blocks = tuple(int(index) for index in streamed_blocks)
        # Snapshot which leaves the manager streams, BEFORE the strip removes
        # `_layer_memory_manager` and makes every leaf look resident.
        streamed_ids = {
            id(child)
            for index in streamed_blocks
            for _, child in self._block_linear_entries(self.blocks[index])
            if is_streamed_module(child)
        }
        # Collect LoRA entries BEFORE stripping compile contaminants: the strip
        # deletes the instance forwards the entries are read from. Sampling
        # needs this as much as training does -- `can_merge_in` is forced False
        # whenever we quantize or offload, so `generate_images` never takes its
        # merge-in shortcut and the adapter stays a live forward hijack that
        # `forward_streamed` would otherwise walk straight past.
        try:
            loras, network = self._collect_block_loras(streamed_blocks)
        except CompileRegionError as error:
            self._ingraph_unavailable_reasons = error.reasons
            raise RuntimeError(
                "in-graph sampling unavailable: " + ",".join(error.reasons)
            ) from error
        collected_loras = sum(len(entries) for entries in loras.values())
        expected_loras = getattr(self, "_ingraph_sampling_lora_leaf_count", None)
        if expected_loras is not None and collected_loras < expected_loras:
            self._ingraph_unavailable_reasons = ("lora_hijack_missing",)
            raise RuntimeError(
                "in-graph sampling unavailable: lora_hijack_missing "
                f"(collected {collected_loras} LoRA leaves, expected {expected_loras})"
            )
        if collected_loras:
            self._ingraph_sampling_lora_leaf_count = collected_loras
        self._ingraph_sampling_loras = loras
        self._ensure_ingraph_lora_multiplier(network, loras)
        self._ingraph_sampling_restores = self._strip_ingraph_compile_contaminants(streamed_blocks)
        reasons = []
        details = []
        for index in streamed_blocks:
            packed_linears = {id(module) for _, module in self._block_linear_entries(self.blocks[index])}
            for name, child in self.blocks[index].named_modules():
                prefix = f"blocks.{index}" + (f".{name}" if name else "")
                if hasattr(child, "_layer_memory_manager"):
                    reasons.append("legacy_layer_manager_present")
                    details.append(f"legacy_layer_manager_present:{prefix}")
                if bool(getattr(child, "_forward_pre_hooks", None)) or bool(getattr(child, "_forward_hooks", None)) or bool(getattr(child, "_forward_hooks_with_kwargs", None)):
                    reasons.append("hook_present")
                    details.append(f"hook_present:{prefix}")
                if "forward" in getattr(child, "__dict__", {}) and id(child) not in packed_linears:
                    reasons.append("forward_hijack_present")
                    details.append(f"forward_hijack_present:{prefix}")
        if reasons:
            self._ingraph_unavailable_reasons = tuple(dict.fromkeys(reasons))
            suffix = ""
            if details:
                suffix = " [" + ";".join(details[:16]) + ("]" if len(details) <= 16 else ";...")
            raise RuntimeError(
                "in-graph sampling unavailable: " + ",".join(self._ingraph_unavailable_reasons) + suffix
            )
        # Model-side glue: enumerate blocks into stable keys; the shared helper
        # owns the borrow-or-own policy, the fail-closed reasons, and cleanup.
        #
        # Only the STREAMED leaves are packed (same fix 07563ad made on the
        # training side). The memory planner splits residency per-Linear, so a
        # block is routinely part streamed / part resident, and the arena only
        # ever holds the offloaded leaves -- demanding all 8 is what produced
        # `borrow refused: stale_modules=3/8`, an owned-pack fallback, and a
        # failed pin budget at every sampling boundary.
        entries_by_block = {
            f"blocks.{index}": list(self._block_linear_entries(self.blocks[index]))
            for index in streamed_blocks
        }
        try:
            result = build_block_leaf_plans(
                getattr(self, "_mm_weight_arena", None),
                entries_by_block,
                is_streamed=lambda module: id(module) in streamed_ids,
                repoint=False,
                pin_mechanism="register",
            )
        except IngraphPackError as error:
            self._ingraph_unavailable_reasons = error.reasons
            raise RuntimeError(f"in-graph sampling unavailable: {error}") from error
        plans = {index: result.plans[f"blocks.{index}"] for index in streamed_blocks}
        # Ticket 534ea49 Phase 2 Slice E: diagnostics only, not a gate -- every
        # STREAMED pack must be pinned (the helper enforces that); it need not be
        # arena-borrowed (a block outside the arena, or one that fell back to an
        # owned pack, is equally valid). Exposed for the smoke harness / tests.
        self._ingraph_sampling_borrowed_count = result.borrowed
        self._ingraph_sampling_owned_count = result.owned
        self._ingraph_sampling_resident_blocks = result.fully_resident
        self._ingraph_sampling_streamed_leaves = result.streamed_leaves
        self._ingraph_sampling_resident_leaves = result.resident_leaves
        self._ingraph_sampling_plans = plans
        # Compat/diagnostic view: only the blocks that actually stream carry a
        # pack. Ownership stays with the plans; disable releases through them.
        self._ingraph_sampling_packs = {
            index: plan.pack for index, plan in plans.items() if plan.streams
        }
        self._ingraph_unavailable_reasons = ()
        self._ingraph_sampling_depth = max(1, int(depth))
        if getattr(self, "_ingraph_sampling_measure", False):
            self.reset_ingraph_sampling_timing()
        configure_fetch_runtime(depth=self._ingraph_sampling_depth)
        fingerprint = tuple(sorted(plans))
        self._compiled_ingraph_fingerprint = fingerprint
        if compile and plans:
            if len(plans) != len(self.blocks):
                self._compiled_ingraph_sampling_blocks = {
                    index: torch.compile(
                        self._make_ingraph_sample_block_fn(index),
                        fullgraph=True,
                        dynamic=False,
                        mode="default",
                    )
                    for index in plans
                }
                return len(plans)
            self._compiled_ingraph_sampling = torch.compile(
                lambda combined, tvec, freqs, mask: self._blocks_trunk(
                    combined,
                    tvec,
                    freqs,
                    mask,
                    force_ingraph=True,
                ),
                fullgraph=True,
                dynamic=False,
                mode="default",
            )
        return len(plans)

    def disable_ingraph_sampling(self):
        snapshot = self.ingraph_sampling_timing_snapshot()
        if snapshot is not None:
            self._last_ingraph_sampling_timing = snapshot
        restores = getattr(self, "_ingraph_sampling_restores", [])
        if restores:
            self._restore_ingraph_compile_contaminants(restores)
        self._ingraph_sampling_restores = []
        for plan in getattr(self, "_ingraph_sampling_plans", {}).values():
            # No-ops on a fully-resident block (pack is None) and on a borrowed
            # arena flat (owns_flat=False).
            release_pack(plan.pack)
        self._ingraph_sampling_plans = {}
        self._ingraph_sampling_packs = {}
        self._ingraph_sampling_loras = {}
        self._ingraph_sampling_borrowed_count = 0
        self._ingraph_sampling_owned_count = 0
        self._ingraph_sampling_resident_blocks = 0
        self._ingraph_sampling_streamed_leaves = 0
        self._ingraph_sampling_resident_leaves = 0
        self._ingraph_unavailable_reasons = ()
        self._ingraph_sampling_depth = 2
        self._compiled_ingraph_sampling = None
        self._compiled_ingraph_sampling_blocks = {}
        self._compiled_ingraph_fingerprint = None

    @staticmethod
    def _lora_owners_on(child):
        """Every LoRA module in this Linear's forward chain.

        A second network applied to the same Linear (`assistant_lora`) chains
        its hijack onto the first, and only the outermost is reachable from
        `child.forward`. The memory manager complicates the walk: with a LoRA
        present, `_capture_base_forward` takes over the LoRA's `org_forward`
        slot and parks whatever was there on `_layer_memory_manager
        ._original_forward` -- so the rest of the chain hangs off the manager,
        not off the Linear."""
        owners = []
        seen = set()
        pending = [getattr(child, "__dict__", {}).get("forward")]
        lmm = getattr(child, "_layer_memory_manager", None)
        if lmm is not None:
            pending.append(getattr(lmm, "_original_forward", None))
        while pending:
            owner = getattr(pending.pop(0), "__self__", None)
            if owner is None or id(owner) in seen:
                continue
            seen.add(id(owner))
            if hasattr(owner, "lora_down"):
                owners.append(owner)
            pending.append(getattr(owner, "org_forward", None))
        return owners

    @staticmethod
    def _describe_lora_chain(child):
        """One-line forensic description of a Linear's forward chain, for the
        lora_hijack_missing error: where (if anywhere) the LoRA hijack went."""

        def _name(fn):
            if fn is None:
                return "None"
            owner = getattr(fn, "__self__", None)
            label = getattr(fn, "__qualname__", None) or getattr(
                fn, "__name__", type(fn).__name__
            )
            if owner is not None:
                return f"{label}@{type(owner).__name__}"
            return label

        parts = []
        parts.append(
            "inst_fwd=" + _name(getattr(child, "__dict__", {}).get("forward"))
        )
        lmm = getattr(child, "_layer_memory_manager", None)
        if lmm is None:
            parts.append("lmm=None")
        else:
            container = getattr(lmm, "_forward_container", None)
            parts.append(
                "lmm(container="
                + ("child" if container is child else type(container).__name__)
                + f",attr={getattr(lmm, '_forward_attribute', None)}"
                + f",orig={_name(getattr(lmm, '_original_forward', None))})"
            )
        ara = getattr(child, "ara_lora_ref", None)
        if ara is None:
            parts.append("ara=None")
        else:
            owner = ara()
            if owner is None:
                parts.append("ara=dead")
            else:
                parts.append(
                    f"ara={type(owner).__name__}"
                    f"(org_forward={_name(getattr(owner, 'org_forward', None))})"
                )
        return " ".join(parts)

    @staticmethod
    def _has_foreign_forward_hijack(child):
        """True when child.forward is wrapped by something that is not the
        memory manager's own streaming forward -- i.e. something the trunk
        would bypass without folding."""
        if "forward" not in getattr(child, "__dict__", {}):
            return False
        lmm = getattr(child, "_layer_memory_manager", None)
        if lmm is None:
            return True
        return not (
            getattr(lmm, "_forward_container", None) is child
            and getattr(lmm, "_forward_attribute", None) == "forward"
        )

    @staticmethod
    def _collect_lora_entry(child):
        """LoraEntry from a Linear's LoRA hijack, None if no LoRA, raise if
        a LoRA is present but not expressible as pure traced math.

        `scale` carries only alpha/rank -- the network multiplier is a live
        tensor applied in the block fn (see `_block_lora_tuple`), because
        `generate_images` reassigns it per image and krea2 reuses one trunk
        across them."""
        owners = SingleStreamDiT._lora_owners_on(child)
        if not owners:
            return None
        if len(owners) > 1:
            # Two networks over one Linear cannot share a single multiplier.
            raise CompileRegionError(["lora_chained"])
        owner = owners[0]
        network_ref = getattr(owner, "network_ref", None)
        network = network_ref() if network_ref is not None else None
        multiplier = getattr(network, "torch_multiplier", None)
        dropout = getattr(owner, "dropout", None)
        if (
            network is None
            or multiplier is None
            or getattr(multiplier, "numel", lambda: 0)() != 1
            or getattr(network, "is_lorm", False)
            or getattr(network, "vector_gates", None) is not None
            or owner.__class__.__name__ in ("DoRAModule", "LokrModule")
            or getattr(owner, "module_dropout", None) is not None
            or getattr(owner, "rank_dropout", None) not in (None, 0)
            or (dropout is not None and not isinstance(dropout, torch.nn.Identity))
        ):
            raise CompileRegionError(["lora_untraceable"])
        return LoraEntry(
            a=owner.lora_down.weight,
            b=owner.lora_up.weight,
            scale=float(owner.scale),
        )

    def _collect_block_loras(self, block_indices):
        """Per-block LoRA entries plus the one network they all belong to.

        MUST run before `_strip_ingraph_compile_contaminants`, which deletes the
        instance forwards the entries are read from. Fails closed rather than
        quietly returning fewer entries: a trunk that drops a LoRA renders the
        base model (sampling) or produces a loss with no grad_fn (training)."""
        loras = {}
        networks = []
        for index in block_indices:
            block_loras = {}
            for name, child in self._block_linear_entries(self.blocks[index]):
                entry = self._collect_lora_entry(child)
                if entry is None:
                    if self._has_foreign_forward_hijack(child):
                        # Some wrapper we cannot express; the trunk calls the
                        # leaf tensors directly and would silently skip it.
                        raise CompileRegionError(["unknown_forward_hijack"])
                    continue
                block_loras[name] = entry
                network = self._lora_owners_on(child)[0].network_ref()
                if all(network is not seen for seen in networks):
                    networks.append(network)
            if block_loras:
                loras[index] = block_loras
        if len(networks) > 1:
            raise CompileRegionError(["lora_multiple_networks"])
        return loras, (networks[0] if networks else None)

    @staticmethod
    def _effective_lora_multiplier(network):
        """What the eager LoRA forward would multiply by, right now.

        Zero covers the three cases where `LoRAModule.forward` skips the adapter
        entirely (inactive, merged into the base, multiplier 0); the trunk has no
        branch, so it renders `base + lora * 0` instead."""
        if network is None:
            return 1.0
        if not getattr(network, "is_active", True) or getattr(network, "is_merged_in", False):
            return 0.0
        if getattr(network, "_multiplier", None) == 0:
            return 0.0
        multiplier = getattr(network, "torch_multiplier", None)
        if multiplier is None or multiplier.numel() != 1:
            raise RuntimeError(
                "in-graph trunk: network multiplier is no longer a scalar "
                f"(numel={0 if multiplier is None else multiplier.numel()}); "
                "the trunk folds one scalar per model and cannot express it"
            )
        return float(multiplier.reshape(()))

    def _ensure_ingraph_lora_multiplier(self, network, loras):
        """One live scalar shared by every folded LoRA leaf.

        Its identity is stable across forwards so torch.compile never
        recompiles; only its value changes, in place. Rebuilt only when the
        adapters move device/dtype."""
        if not loras:
            return None
        entry = next(iter(next(iter(loras.values())).values()))
        multiplier = getattr(self, "_ingraph_lora_multiplier", None)
        if (
            multiplier is None
            or multiplier.device != entry.a.device
            or multiplier.dtype != entry.a.dtype
        ):
            multiplier = torch.ones((), device=entry.a.device, dtype=entry.a.dtype)
            self._ingraph_lora_multiplier = multiplier
        self._ingraph_lora_network = network
        self._ingraph_lora_multiplier_value = None
        self._refresh_ingraph_lora_multiplier()
        return multiplier

    def _refresh_ingraph_lora_multiplier(self):
        """Pull the network's current multiplier into the trunk's live scalar.

        Called from `_forward_impl` before either trunk runs, outside every
        compiled region."""
        multiplier = getattr(self, "_ingraph_lora_multiplier", None)
        if multiplier is None:
            return
        value = self._effective_lora_multiplier(getattr(self, "_ingraph_lora_network", None))
        if value != self._ingraph_lora_multiplier_value:
            with torch.no_grad():
                multiplier.fill_(value)
            self._ingraph_lora_multiplier_value = value

    @staticmethod
    def _nest_block_train_leaves(views, loras):
        def leaf(name):
            return TrainLeaf(view=views[name], lora=loras.get(name))

        return {
            "attn": {
                "wq": leaf("attn.wq"),
                "wk": leaf("attn.wk"),
                "wv": leaf("attn.wv"),
                "gate": leaf("attn.gate"),
                "wo": leaf("attn.wo"),
            },
            "mlp": {
                "gate": leaf("mlp.gate"),
                "up": leaf("mlp.up"),
                "down": leaf("mlp.down"),
            },
        }

    def _make_ingraph_sample_block_fn(self, index):
        block = self.blocks[index]
        plan = self._ingraph_sampling_plans[index]
        fp8_flags = plan.fp8_flags
        # The pack carries the FROZEN base weights. A LoRA is a live forward
        # hijack that forward_streamed bypasses, so it has to be folded back in
        # here or the preview renders the base model.
        loras = self._ingraph_sampling_loras.get(index, {})
        multiplier = self._ingraph_lora_multiplier

        if not plan.streams:
            # Every leaf is already on the device: no flat, no fetch, no token.
            # The planner keeps leaves resident when VRAM allows, and a resident
            # leaf's Parameter is an ordinary graph input.
            leaf_args = assemble_leaf_args(plan)

            def fn(x, tvec, freqs, mask):
                lora_args = (
                    SingleStreamDiT._block_lora_tuple(loras, multiplier)
                    if loras
                    else None
                )
                return block.forward_streamed(
                    x,
                    tvec,
                    freqs,
                    mask,
                    leaf_args,
                    fp8_flags,
                    loras=lora_args,
                )

            return fn

        pack = plan.pack
        host = pack.host_flat
        nbytes = int(pack.required_pin_bytes)

        def fn(x, tvec, freqs, mask):
            token = torch.ops.mm.fetch_start_after(host, x)
            flat = torch.ops.mm.fetch_wait(token, nbytes)
            # One coalesced fetch over this block's STREAMED leaves only; the
            # resident ones are spliced back into canonical order.
            leaf_args = assemble_leaf_args(plan, block_tensor_views(flat, pack))
            lora_args = (
                SingleStreamDiT._block_lora_tuple(loras, multiplier) if loras else None
            )
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

        return fn

    @staticmethod
    def _block_lora_tuple(loras, multiplier=None):
        """(a, b, scale) per canonical leaf, in `_block_linear_entries` order.

        Call this INSIDE the traced block fn: with a live `multiplier` tensor
        the scale is a traced op, so a per-image multiplier change costs no
        recompile. Hoisting it to build time would freeze the first value."""
        out = []
        for name, _ in SingleStreamDiT._block_linear_entries_for_lora_order():
            entry = loras.get(name)
            if entry is None:
                out.append(None)
            elif multiplier is None:
                out.append((entry.a, entry.b, entry.scale))
            else:
                out.append((entry.a, entry.b, entry.scale * multiplier))
        return tuple(out)

    @staticmethod
    def _block_linear_entries_for_lora_order():
        return (
            ("attn.wq", None),
            ("attn.wk", None),
            ("attn.wv", None),
            ("attn.gate", None),
            ("attn.wo", None),
            ("mlp.gate", None),
            ("mlp.up", None),
            ("mlp.down", None),
        )

    def _make_ingraph_train_block_fn(self, index):
        block = self.blocks[index]
        plan = self._ingraph_training_plans[index]
        loras = self._ingraph_training_loras.get(index, {})
        fp8_flags = plan.fp8_flags
        multiplier = self._ingraph_lora_multiplier

        if not plan.streams:
            # Every leaf is already on the device: no flat, no fetch, no token.
            # The planner keeps blocks resident when VRAM allows, and a resident
            # leaf's Parameter is an ordinary graph input.
            leaf_args = assemble_leaf_args(plan)

            def fn(x, tvec, freqs, mask):
                lora_args = (
                    SingleStreamDiT._block_lora_tuple(loras, multiplier)
                    if loras
                    else None
                )
                return block.forward_streamed(
                    x,
                    tvec,
                    freqs,
                    mask,
                    leaf_args,
                    fp8_flags,
                    training=True,
                    loras=lora_args,
                )

            return fn

        pack = plan.pack
        host = pack.host_flat
        nbytes = int(pack.required_pin_bytes)

        def fn(x, tvec, freqs, mask):
            compiling = torch.compiler.is_compiling()
            if compiling:
                token = torch.ops.mm.fetch_start_after(host, x)
            else:
                token = torch.ops.mm.fetch_start(host)
            flat = torch.ops.mm.fetch_wait(token, nbytes)
            # One coalesced fetch over this block's STREAMED leaves only; the
            # resident ones are spliced back into canonical order.
            leaf_args = assemble_leaf_args(plan, block_tensor_views(flat, pack))
            lora_args = (
                SingleStreamDiT._block_lora_tuple(loras, multiplier) if loras else None
            )
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

        return fn

    def _assert_ingraph_training_current(self, x):
        """A trunk's resident leaves are the tensors captured at enable time.

        `_move_unmanaged_parameters` cannot move a quantized Parameter in place;
        it swaps in a new object. For a FROZEN base that churn is benign (same
        values, old storage still alive), so Parameter identity is the wrong
        thing to police -- `get_noise_prediction` re-runs the move on every call
        whenever any weight is streamed, since the model then reports device=cpu.

        What is NOT benign is a cpu->cuda move after enable: the trunk keeps the
        host tensors and the fp8 path feeds `_scaled_mm` a cuda activation and a
        cpu weight. Compare devices, O(1), and fail closed."""
        expected = getattr(self, "_ingraph_training_resident_device", None)
        if expected is not None and expected != x.device:
            raise RuntimeError(
                "in-graph training trunk is stale: its resident leaves are on "
                f"{expected} but activations are on {x.device} (the model moved "
                "after enable_ingraph_training). Re-enable the trunk after "
                "moving the model."
            )

    def _ingraph_training_trunk(self, combined, tvec, freqs, mask):
        context_fn = (
            compiled_checkpoint_context
            if torch.compiler.is_compiling()
            else checkpoint_recompute_context
        )
        for fn in self._ingraph_training_block_fns:
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

    def enable_ingraph_training(self, depth: int = 2, compile: bool = True):
        """Phase 4a: compiled fully-streamed TRAINING trunk (all blocks).

        Each block is checkpoint(fetch + leaves-passing block call); backward
        re-fetch falls out of checkpoint recompute; LoRA A/B enter as
        ordinary trainable graph inputs. Fail-closed like the sampler enable.
        Call AFTER the LoRA network is applied and configured (its multiplier
        must be a scalar; its value is read live, not folded)."""
        self.disable_ingraph_training()
        # Every block runs through the trunk; only the FETCH is conditional. A
        # block whose leaves the planner left resident simply has no pack.
        streamed_blocks = tuple(range(len(self.blocks)))
        # Snapshot which leaves the manager streams, BEFORE the strip removes
        # `_layer_memory_manager` and makes every leaf look resident.
        streamed_ids = {
            id(child)
            for index in streamed_blocks
            for _, child in self._block_linear_entries(self.blocks[index])
            if is_streamed_module(child)
        }
        # Collect LoRA entries BEFORE stripping compile contaminants: the
        # strip deletes instance forwards, including the LoRA hijacks the
        # entries are read from (learned the hard way: 232/512 LoRA grads).
        try:
            loras, network = self._collect_block_loras(streamed_blocks)
        except CompileRegionError as error:
            self._ingraph_unavailable_reasons = error.reasons
            raise RuntimeError(
                "in-graph training unavailable: " + ",".join(error.reasons)
            ) from error
        # _collect_lora_entry reads the hijack out of the Linear's instance
        # forward and returns None when it is absent -- so anything that eats the
        # hijack (see _strip_ingraph_compile_contaminants) yields an empty `loras`
        # and a trunk of pure frozen math, whose loss silently has no grad_fn.
        # A re-enable must never see fewer adapters than the enable before it.
        collected_loras = sum(len(entries) for entries in loras.values())
        expected_loras = getattr(self, "_ingraph_training_lora_leaf_count", None)
        if expected_loras is not None and collected_loras < expected_loras:
            self._ingraph_unavailable_reasons = ("lora_hijack_missing",)
            details = []
            for index in streamed_blocks:
                for name, child in self._block_linear_entries(self.blocks[index]):
                    if name in loras.get(index, {}):
                        continue
                    details.append(
                        f"blocks.{index}.{name}: "
                        + self._describe_lora_chain(child)
                    )
                    if len(details) >= 4:
                        break
                if len(details) >= 4:
                    break
            raise RuntimeError(
                "in-graph training unavailable: lora_hijack_missing "
                f"(collected {collected_loras} LoRA leaves, expected {expected_loras}) "
                "first missing leaves: [" + "; ".join(details) + "]"
            )
        if collected_loras:
            self._ingraph_training_lora_leaf_count = collected_loras
        self._ingraph_training_restores = self._strip_ingraph_compile_contaminants(
            streamed_blocks
        )
        reasons = []
        for index in streamed_blocks:
            packed = {id(m) for _, m in self._block_linear_entries(self.blocks[index])}
            for name, child in self.blocks[index].named_modules():
                if hasattr(child, "_layer_memory_manager"):
                    reasons.append("legacy_layer_manager_present")
                if (
                    bool(getattr(child, "_forward_pre_hooks", None))
                    or bool(getattr(child, "_forward_hooks", None))
                    or bool(getattr(child, "_forward_hooks_with_kwargs", None))
                ):
                    reasons.append("hook_present")
                if "forward" in getattr(child, "__dict__", {}) and id(child) not in packed:
                    reasons.append("forward_hijack_present")
        if reasons:
            self._ingraph_unavailable_reasons = tuple(dict.fromkeys(reasons))
            raise RuntimeError(
                "in-graph training unavailable: "
                + ",".join(self._ingraph_unavailable_reasons)
            )
        # Phase 3 Slice B: BORROW the persistent arena flat for a block whose
        # frozen base is already pinned by the arena (the same mechanic the
        # sampler enable uses), instead of pinning a second independent copy of
        # the base weights. LoRA composition happens in the block fn below;
        # grads flow only to the resident adapters, so the borrowed frozen flat
        # is a read-only source. An owned pack is the fallback for any block the
        # arena does not currently cover.
        # pin_mechanism="register": exact-size cudaHostRegister for any OWNED
        # fallback pack. The default "alloc" path goes through torch's caching
        # host allocator, which rounds each request up to a power-of-two bucket
        # -- a 0.40 GiB block can commit up to 2x that against the DXGI shared
        # budget. Over the full 28-block streamed set that overhead starves the
        # last packs ("pin refused (ingraph_pack): 0.40 GiB > 0.33 GiB
        # available") and fails the whole compile with non_pinned_pack.
        #
        # Only the STREAMED leaves are packed. The memory planner splits
        # residency per-Linear, so a block is routinely part streamed / part
        # resident, and the arena only ever holds the offloaded leaves --
        # demanding all 8 is what produced `borrow refused: stale_modules=3/8`.
        entries_by_block = {
            f"blocks.{index}": list(self._block_linear_entries(self.blocks[index]))
            for index in streamed_blocks
        }
        try:
            result = build_block_leaf_plans(
                getattr(self, "_mm_weight_arena", None),
                entries_by_block,
                is_streamed=lambda module: id(module) in streamed_ids,
                repoint=False,
                pin_mechanism="register",
            )
        except IngraphPackError as error:
            self._ingraph_unavailable_reasons = error.reasons
            raise RuntimeError(f"in-graph training unavailable: {error}") from error
        plans = {index: result.plans[f"blocks.{index}"] for index in streamed_blocks}
        for index in streamed_blocks:
            for _, child in self._block_linear_entries(self.blocks[index]):
                if id(child) not in streamed_ids:
                    # A resident leaf is supposed to be on the device, and the
                    # manager already put it there. Marking it a pack source
                    # would pin it to the host and make a promoted sampler
                    # block run F.linear against a CPU weight ("mat2 is on cpu").
                    continue
                # Keep unmanaged-parameter moves (model.to) off the pack
                # sources: their CPU residency is the design, the trunk
                # streams them from the pinned pack.
                child._mm_ingraph_pack_source = True
        # Diagnostics (Slice B): under strict pinned-arena validation the smoke
        # harness treats any owned-fallback pack as a FAILURE -- every streamed
        # base block must be borrowed from the arena. Exposed like the sampler's
        # borrowed/owned counts.
        self._ingraph_training_borrowed_count = result.borrowed
        self._ingraph_training_owned_count = result.owned
        self._ingraph_training_resident_blocks = result.fully_resident
        self._ingraph_training_streamed_leaves = result.streamed_leaves
        self._ingraph_training_resident_leaves = result.resident_leaves
        self._ingraph_training_plans = plans
        self._ingraph_training_loras = loras
        self._ensure_ingraph_lora_multiplier(network, loras)
        # Remember where this trunk's resident leaves live, so a later device
        # move fails closed instead of reaching _scaled_mm (see
        # _assert_ingraph_training_current). A split across devices is nonsense.
        resident_devices = {
            tensor.device
            for plan in plans.values()
            for triple in plan.resident_args
            for tensor in triple
            if tensor is not None
        }
        if len(resident_devices) > 1:
            self._ingraph_unavailable_reasons = ("resident_leaf_device_split",)
            raise RuntimeError(
                "in-graph training unavailable: resident_leaf_device_split "
                f"({sorted(str(d) for d in resident_devices)})"
            )
        self._ingraph_training_resident_device = (
            next(iter(resident_devices)) if resident_devices else None
        )
        self._ingraph_unavailable_reasons = ()
        self._ingraph_training_block_fns = [
            self._make_ingraph_train_block_fn(i) for i in streamed_blocks
        ]
        configure_fetch_runtime(depth=max(1, int(depth)))
        if compile:
            from toolkit.memory_management.ingraph_stream_scheduling import (
                install_ordering_pass,
            )

            install_ordering_pass()
            self._compiled_ingraph_training = torch.compile(
                self._ingraph_training_trunk,
                fullgraph=True,
                dynamic=False,
                mode="default",
            )
        else:
            self._compiled_ingraph_training = self._ingraph_training_trunk
        return len(plans)

    def disable_ingraph_training(self):
        for block in self.blocks:
            for _, child in self._block_linear_entries(block):
                if hasattr(child, "_mm_ingraph_pack_source"):
                    del child._mm_ingraph_pack_source
        restores = getattr(self, "_ingraph_training_restores", [])
        if restores:
            self._restore_ingraph_compile_contaminants(restores)
        self._ingraph_training_restores = []
        for plan in getattr(self, "_ingraph_training_plans", {}).values():
            # No-ops on a fully-resident block (pack is None) and on a borrowed
            # arena flat (owns_flat=False).
            release_pack(plan.pack)
        self._ingraph_training_plans = {}
        self._ingraph_training_loras = {}
        self._ingraph_training_block_fns = []
        self._ingraph_training_borrowed_count = 0
        self._ingraph_training_owned_count = 0
        self._ingraph_training_resident_blocks = 0
        self._ingraph_training_streamed_leaves = 0
        self._ingraph_training_resident_leaves = 0
        self._ingraph_training_resident_device = None
        self._compiled_ingraph_training = None

    def _enable_lora_compile_fast_path(self):
        restores = []
        for module in self.modules():
            ready = getattr(module, "_memory_management_compile_fast_lora_ready", None)
            if ready is None:
                continue
            try:
                if not ready():
                    continue
            except Exception:
                continue
            previous = getattr(module, "_memory_management_compile_lora_fast", None)
            module._memory_management_compile_lora_fast = True
            restores.append((module, previous))
        return restores

    @staticmethod
    def _disable_lora_compile_fast_path(restores):
        for module, previous in reversed(restores):
            if previous is None:
                if hasattr(module, "_memory_management_compile_lora_fast"):
                    del module._memory_management_compile_lora_fast
            else:
                module._memory_management_compile_lora_fast = previous

    def enable_compiled_training(self, pinned_keys: set[str] | None = None):
        """Compile permanent-resident blocks for grad-enabled training."""
        self.disable_compiled_training()
        mm = getattr(self, "_memory_manager", None)
        fp8_restores = []
        if mm is not None:
            try:
                fp8_restores, _ = mm.__class__._enable_fp8_training_compile(self)
            except Exception:
                fp8_restores = []
        lora_restores = self._enable_lora_compile_fast_path()
        try:
            readiness = self.training_compile_readiness(pinned_keys)
            clean = tuple(
                item["index"] for item in readiness["statuses"]
                if item["ready"]
            )
            compiled: list = [None] * len(self.blocks)
            for i in clean:
                compiled[i] = torch.compile(
                    self.blocks[i],
                    fullgraph=False,
                    dynamic=False,
                    mode="default",
                )
            self._compiled_training_blocks = compiled
            self._compiled_training_fingerprint = clean
            self._compiled_training_fp8_restores = fp8_restores
            self._compiled_training_lora_restores = lora_restores
            return len(clean), readiness["blocked_blocks"]
        except Exception:
            self._disable_lora_compile_fast_path(lora_restores)
            if mm is not None:
                mm.__class__._disable_fp8_training_compile(self, fp8_restores)
            self._compiled_training_blocks = None
            self._compiled_training_fingerprint = None
            self._compiled_training_fp8_restores = []
            self._compiled_training_lora_restores = []
            raise

    def disable_compiled_training(self):
        mm = getattr(self, "_memory_manager", None)
        if mm is not None:
            try:
                mm.__class__._disable_fp8_training_compile(
                    self,
                    getattr(self, "_compiled_training_fp8_restores", []),
                )
            except Exception:
                pass
        self._disable_lora_compile_fast_path(
            getattr(self, "_compiled_training_lora_restores", [])
        )
        self._compiled_training_blocks = None
        self._compiled_training_fingerprint = None
        self._compiled_training_fp8_restores = []
        self._compiled_training_lora_restores = []

    def forward(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
        reflen: int = 0,
        isolate_refs: bool = False,
        ref_kv_capture: list | None = None,
        ref_kv_cache: tuple[list, Tensor] | None = None,
    ) -> Tensor:
        return self._forward_impl(
            img,
            context,
            t,
            pos,
            mask,
            reflen=reflen,
            isolate_refs=isolate_refs,
            ref_kv_capture=ref_kv_capture,
            ref_kv_cache=ref_kv_cache,
        )

    def _forward_impl(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
        *,
        force_ingraph: bool = False,
        reflen: int = 0,
        isolate_refs: bool = False,
        ref_kv_capture: list | None = None,
        ref_kv_cache: tuple[list, Tensor] | None = None,
    ) -> Tensor:
        img = self.first(img)
        t = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)

        txtmask = _mask(mask[:, : context.shape[1]])

        context = self.txtfusion(context, mask=txtmask)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)

        reference_mode = (
            reflen > 0 or ref_kv_capture is not None or ref_kv_cache is not None
        )
        use_compiled = (
            self._compiled_blocks is not None
            and not torch.is_grad_enabled()
            and not reference_mode
        )
        use_ingraph = (
            bool(self._ingraph_sampling_plans)
            and not torch.is_grad_enabled()
            and not reference_mode
        )
        use_ingraph_train = (
            self._compiled_ingraph_training is not None
            and torch.is_grad_enabled()
            and not reference_mode
        )

        # Pad the combined sequence to a multiple of 256 when a compiled block
        # region will run. The pad slots are appended after the image tokens,
        # masked False, and sliced off below, so this is numerically identical.
        if use_compiled or use_ingraph or use_ingraph_train:
            fulllen = combined.shape[1]
            _padlen = (-fulllen) % 256
            if _padlen > 0:
                combined = F.pad(combined, (0, 0, 0, _padlen))
                mask = F.pad(mask, (0, _padlen), value=False)
                pos = F.pad(pos, (0, 0, 0, _padlen))

        blockvec = tvec
        if reflen > 0:
            # The last ``reflen`` image tokens are clean reference tokens: they
            # get t=0 modulation (ComfyUI Kontext "index_timestep_zero") while
            # text + noisy image tokens keep the real t. Padding tokens fall in
            # the t=0 span, but they are masked from attention and sliced off
            # the output, so their values never matter.
            t0 = self.tmlp(
                temb(
                    torch.zeros_like(t[:, 0, 0]),
                    self.config.tdim,
                    device=img.device,
                    dtype=img.dtype,
                )
            )
            blockvec = (tvec, self.tproj(t0), txtlen + imglen - reflen)

        padmask = mask  # (B, L) key-padding mask, incl. the 256-alignment pad
        mask = _mask(mask)
        if reflen > 0 and isolate_refs:
            split = txtlen + imglen - reflen
            is_ref = torch.zeros(
                combined.shape[1], dtype=torch.bool, device=combined.device
            )
            is_ref[split : split + reflen] = True
            mask = mask & (~is_ref[:, None] | is_ref[None, :])

        ref_span = None
        if ref_kv_capture is not None and reflen > 0:
            if not isolate_refs:
                raise ValueError("ref K/V capture requires isolate_refs")
            split = txtlen + imglen - reflen
            ref_span = (split, split + reflen)

        blockcaches = None
        if ref_kv_cache is not None:
            blockcaches, refmask = ref_kv_cache
            extra = padmask.unsqueeze(1).unsqueeze(3) & refmask.unsqueeze(1).unsqueeze(2)
            mask = torch.cat((mask, extra), dim=3)
        freqs = self.posemb(pos)

        if use_ingraph or use_ingraph_train:
            # generate_images reassigns network.multiplier per image, and the
            # network can be deactivated around a forward. Pull the current
            # value into the trunk's live scalar here, outside every compiled
            # region, so neither costs a recompile.
            self._refresh_ingraph_lora_multiplier()

        if use_ingraph_train:
            self._assert_ingraph_training_current(combined)
            combined = self._compiled_ingraph_training(combined, tvec, freqs, mask)
        elif use_ingraph and self._compiled_ingraph_sampling is not None:
            if getattr(self, "_ingraph_sampling_measure", False):
                if combined.device.type == "cuda":
                    torch.cuda.synchronize(combined.device)
                started = time.perf_counter()
                combined = self._compiled_ingraph_sampling(combined, tvec, freqs, mask)
                if combined.device.type == "cuda":
                    torch.cuda.synchronize(combined.device)
                self._record_ingraph_sampling_timing(time.perf_counter() - started)
            else:
                combined = self._compiled_ingraph_sampling(combined, tvec, freqs, mask)
        else:
            combined = self._blocks_trunk(
                combined,
                blockvec,
                freqs,
                mask,
                force_ingraph=force_ingraph,
                ref_span=ref_span,
                ref_kv_capture=ref_kv_capture,
                blockcaches=blockcaches,
            )

        final = self.last(combined, t)
        output = final[:, txtlen : txtlen + imglen - reflen, :]

        return output

    def _blocks_trunk(
        self,
        combined: Tensor,
        tvec: Tensor,
        freqs: Tensor,
        mask: Tensor | None,
        *,
        force_ingraph: bool = False,
        ref_span: tuple[int, int] | None = None,
        ref_kv_capture: list | None = None,
        blockcaches: list | None = None,
    ) -> Tensor:
        reference_mode = (
            isinstance(tvec, tuple)
            or ref_kv_capture is not None
            or blockcaches is not None
        )
        use_compiled = (
            self._compiled_blocks is not None
            and not torch.is_grad_enabled()
            and not reference_mode
        )
        use_ingraph = (
            force_ingraph or bool(self._ingraph_sampling_plans)
        ) and not torch.is_grad_enabled() and not reference_mode
        checkpoint_cutoff = len(self.blocks) - self._checkpoint_keep_last
        use_compiled_training = (
            self._compiled_training_blocks is not None
            and torch.is_grad_enabled()
            and not reference_mode
        )
        if blockcaches is None:
            blockcaches = [None] * len(self.blocks)
        for i, (block, blockkv) in enumerate(zip(self.blocks, blockcaches)):
            block_call = block
            if use_compiled_training and self._compiled_training_blocks[i] is not None:
                block_call = self._compiled_training_blocks[i]
            if (
                self.gradient_checkpointing
                and torch.is_grad_enabled()
                and i < checkpoint_cutoff
            ):
                combined = checkpoint(
                    block_call,
                    combined,
                    tvec,
                    freqs,
                    mask,
                    use_reentrant=False,
                    ref_span=ref_span,
                    kv_capture=ref_kv_capture,
                    kv_cache=blockkv,
                )
            elif use_compiled and self._compiled_blocks[i] is not None:
                # Hook-free block: run its compiled graph. The block that still
                # streams weights falls through to the eager call below.
                combined = self._compiled_blocks[i](combined, tvec, freqs, mask)
            elif use_ingraph and i in self._ingraph_sampling_plans:
                compiled_ingraph = self._compiled_ingraph_sampling_blocks.get(i)
                if compiled_ingraph is not None and not force_ingraph:
                    combined = compiled_ingraph(combined, tvec, freqs, mask)
                else:
                    plan = self._ingraph_sampling_plans[i]
                    block_loras = self._ingraph_sampling_loras.get(i)
                    lora_args = (
                        self._block_lora_tuple(
                            block_loras, self._ingraph_lora_multiplier
                        )
                        if block_loras
                        else None
                    )
                    if plan.streams:
                        pack = plan.pack
                        token = torch.ops.mm.fetch_start_after(
                            pack.host_flat, combined
                        )
                        flat = torch.ops.mm.fetch_wait(
                            token, int(pack.required_pin_bytes)
                        )
                        leaf_args = assemble_leaf_args(
                            plan, block_tensor_views(flat, pack)
                        )
                        combined = block.forward_streamed(
                            combined,
                            tvec,
                            freqs,
                            mask,
                            leaf_args,
                            plan.fp8_flags,
                            loras=lora_args,
                        )
                        torch.ops.mm.fetch_free_after(token, combined)
                    else:
                        combined = block.forward_streamed(
                            combined,
                            tvec,
                            freqs,
                            mask,
                            assemble_leaf_args(plan),
                            plan.fp8_flags,
                            loras=lora_args,
                        )
            else:
                combined = block_call(
                    combined,
                    tvec,
                    freqs,
                    mask,
                    ref_span=ref_span,
                    kv_capture=ref_kv_capture,
                    kv_cache=blockkv,
                )
        return combined
