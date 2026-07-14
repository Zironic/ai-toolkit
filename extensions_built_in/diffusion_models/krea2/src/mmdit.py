"""Krea 2 (K2) single-stream MMDiT backbone.

Vendored from the reference ``mmdit.py`` for ai-toolkit. This is a single-stream
MMDiT: Qwen3-VL text features are fused by a small ``TextFusionTransformer`` and
then concatenated with the patchified image latent tokens into one sequence that
flows through ``SingleStreamBlock`` layers. The model predicts the flow-matching
velocity on the image tokens.

Differences from the reference (all training-driven, numerically equivalent):
  - ``torch.compile`` decorators are dropped (they fight gradient checkpointing,
    LoRA module swapping and variable shapes during training).
  - Attention prefers native cuDNN GQA for eligible CUDA signatures and keeps
    the expanded-KV memory-efficient fallback for unsupported builds.
  - ``enable_gradient_checkpointing`` / ``disable_gradient_checkpointing`` and a
    per-block ``torch.utils.checkpoint`` wrapper are added (gated on
    ``torch.is_grad_enabled()`` so eval/sampling never pays for it).
"""

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from toolkit.memory_management.runtime import get_memory_runtime
from torch.nn.attention import SDPBackend, sdpa_kernel

from toolkit.sdpa_patch import can_use_native_cudnn_gqa, get_gqa_backend_mode


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
        forced_cudnn = can_use_native_cudnn_gqa(q, k, v, mask)
        selected = (
            "CUDNN_ATTENTION (forced native GQA)"
            if forced_cudnn
            else next((name for name in order if eligible.get(name)), "NONE")
        )
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
    # Native cuDNN GQA avoids both the math backend's score tensor and physical
    # KV expansion. The process-wide SDPA wrapper forces cuDNN for eligible
    # signatures. Expand only when cuDNN is unavailable or rejects this call;
    # that makes the memory-efficient backend eligible as the fallback. Masked
    # calls and builds without Flash otherwise need this fallback.
    backend_mode = get_gqa_backend_mode()
    requested_gqa = gqa and k.shape[1] != q.shape[1]
    native_cudnn_gqa = (
        requested_gqa
        and backend_mode != "expanded_efficient"
        and can_use_native_cudnn_gqa(q, k, v, mask)
    )
    if gqa and backend_mode == "cudnn" and not native_cudnn_gqa:
        raise RuntimeError(
            "sdpa_gqa_backend=cudnn was requested, but cuDNN rejected the "
            "unexpanded Krea2 GQA signature"
        )
    if gqa and not native_cudnn_gqa and k.shape[1] != q.shape[1] and (
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
    if backend_mode == "expanded_efficient" and requested_gqa:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=scale, enable_gqa=False
            )
    else:
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

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


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
        ref_span: tuple[int, int] | None = None,
        kv_capture: list | None = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)

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
        return self.wo(out)


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
        x = x + pregate * self.attn(
            (1 + prescale) * self.prenorm(x) + preshift,
            freqs,
            mask,
            **attn_kwargs,
        )
        x = x + postgate * self.mlp(
            (1 + postscale) * self.postnorm(x) + postshift,
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
        # One live scalar shared by the generic train/sample programs. Its
        # identity stays stable while the current network multiplier is copied
        # into it outside compiled regions.
        self._runtime_lora_multiplier = None
        self._runtime_lora_multiplier_value = None
        self._runtime_lora_network = None
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

    def enable_gradient_checkpointing(self, keep_last: int | None = None):
        self.gradient_checkpointing = True
        if keep_last is not None:
            self._checkpoint_keep_last = max(0, int(keep_last))

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self._checkpoint_keep_last = 0

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
            reflen > 0
            or ref_kv_capture is not None
            or ref_kv_cache is not None
        )
        memory_runtime = get_memory_runtime(self)
        use_runtime = memory_runtime is not None and not reference_mode

        # Permanent runtime programs use 256-token buckets. Reference calls
        # retain their native eager shapes and alternate block ABI.
        if use_runtime:
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
            # _mask() deliberately returns None for an all-valid sequence and a
            # key-only (B, 1, 1, L) mask otherwise. Reference isolation is
            # query-dependent, so materialize the live query rows only here.
            if mask is None:
                mask = padmask[:, None, None, :]
            mask = mask.expand(-1, 1, combined.shape[1], -1)
            isolation_mask = ~is_ref[:, None] | is_ref[None, :]
            mask = mask & isolation_mask[None, None, :, :]

        ref_span = None
        if ref_kv_capture is not None and reflen > 0:
            if not isolate_refs:
                raise ValueError("ref K/V capture requires isolate_refs")
            split = txtlen + imglen - reflen
            ref_span = (split, split + reflen)

        blockcaches = None
        if ref_kv_cache is not None:
            blockcaches, refmask = ref_kv_cache
            # Cached reference keys add an (R) key span to every live query.
            # Expand the local key-only/None mask to (B, 1, L, L) before
            # concatenating the reference-key mask (B, 1, L, R).
            if mask is None:
                mask = padmask[:, None, None, :]
            mask = mask.expand(-1, 1, combined.shape[1], -1)
            extra = padmask.unsqueeze(1).unsqueeze(3) & refmask.unsqueeze(1).unsqueeze(2)
            mask = torch.cat((mask, extra), dim=3)
        freqs = self.posemb(pos)

        combined = self._blocks_trunk(
            combined,
            blockvec,
            freqs,
            mask,
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
        ref_span: tuple[int, int] | None = None,
        ref_kv_capture: list | None = None,
        blockcaches: list | None = None,
    ) -> Tensor:
        # The ordinary model loop remains the sole owner of execution order and
        # checkpointing. Arena offload, when active, intercepts each selected
        # block at its existing call boundary.
        checkpoint_cutoff = len(self.blocks) - self._checkpoint_keep_last
        if blockcaches is None:
            blockcaches = [None] * len(self.blocks)
        for i, (block, blockkv) in enumerate(zip(self.blocks, blockcaches)):
            if (
                self.gradient_checkpointing
                and torch.is_grad_enabled()
                and i < checkpoint_cutoff
            ):
                combined = checkpoint(
                    block,
                    combined,
                    tvec,
                    freqs,
                    mask,
                    use_reentrant=False,
                    ref_span=ref_span,
                    kv_capture=ref_kv_capture,
                    kv_cache=blockkv,
                )
            else:
                combined = block(
                    combined,
                    tvec,
                    freqs,
                    mask,
                    ref_span=ref_span,
                    kv_capture=ref_kv_capture,
                    kv_cache=blockkv,
                )
        return combined
