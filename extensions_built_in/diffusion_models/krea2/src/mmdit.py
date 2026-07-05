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
    LoraEntry,
    TrainLeaf,
    assert_compile_region_clean,
    block_linear_views,
    checkpoint_recompute_context,
    compiled_checkpoint_context,
    configure_fetch_runtime,
    free_on_backward,
    in_recompute,
    pack_block_host,
    streamed_linear,
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
    # But enable_gqa=True AND an explicit attn_mask disqualify *both* fast
    # backends at once — Flash rejects arbitrary masks, the memory-efficient
    # (cutlass) backend rejects enable_gqa — so dispatch silently falls to the
    # math backend, which materializes the full (B, heads, L, L) score tensor
    # (multiple GiB at sampling resolutions). When we have a mask, expand the KV
    # heads to match Q here and drop enable_gqa, so the memory-efficient backend
    # (which does accept masks) becomes eligible. This is numerically identical
    # to enable_gqa=True: it repeats each KV head across its query-head group.
    if gqa and mask is not None and k.shape[1] != q.shape[1]:
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        gqa = False
    x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
    )
    return rearrange(x, "B H L D -> B L (H D)")


def _mask(mask: Tensor) -> Tensor:
    """Expand a (B, L) key-padding mask into a (B, 1, L, L) attention mask."""
    return mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)


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
    ) -> Tensor:
        if leaves is None:
            q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)
        else:
            q = streamed_linear(qkv, leaves["wq"])
            k = streamed_linear(qkv, leaves["wk"])
            v = streamed_linear(qkv, leaves["wv"])
            gate = streamed_linear(qkv, leaves["gate"])

        q, k, v = (
            rearrange(q, "B L (H D) -> B H L D", H=self.heads),
            rearrange(k, "B L (H D) -> B H L D", H=self.kvheads),
            rearrange(v, "B L (H D) -> B H L D", H=self.kvheads),
        )

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        out = attention(q, k, v, mask=mask, gqa=self.gqa) * F.sigmoid(gate)
        if leaves is None:
            out = self.wo(out)
        else:
            out = streamed_linear(out, leaves["wo"])

        return out


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
    ) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        attn_leaves = None if leaves is None else leaves["attn"]
        mlp_leaves = None if leaves is None else leaves["mlp"]
        x = x + pregate * self.attn(
            (1 + prescale) * self.prenorm(x) + preshift,
            freqs,
            mask,
            leaves=attn_leaves,
        )
        x = x + postgate * self.mlp(
            (1 + postscale) * self.postnorm(x) + postshift,
            leaves=mlp_leaves,
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
        self._ingraph_sampling_restores: list = []
        self._ingraph_unavailable_reasons: tuple[str, ...] = ()
        self._ingraph_sampling_depth = 2
        self._compiled_ingraph_sampling = None
        self._compiled_ingraph_fingerprint: tuple | None = None
        self._ingraph_sampling_measure = False
        self._ingraph_training_packs: dict[int, object] = {}
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
        if any(hasattr(sub, "_layer_memory_manager") for sub in block.modules()):
            reasons.append("streaming_hook")
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

        `dynamic=None` (torch's automatic-dynamic-shapes mode) lets Dynamo
        specialize to the first shape seen and only pay for a symbolic-shape
        upgrade if a second distinct shape shows up -- the caller is expected
        to run the actual invocation under
        `torch.compiler.set_stance("eager_then_compile")` so that upgrade
        decision is made from real eager-mode shape history instead of
        wasting a static compile on the very first call.

        Returns (compiled_count, eager_count).
        """
        clean = tuple(
            i for i, block in enumerate(self.blocks)
            if self._block_compile_safe(block)
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
                dynamic=None,
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
                restores.append((child, lmm, container, attribute, managed_forward, saved_attrs))
            if hook_state:
                restores.append(("hooks", hook_state))
        return restores

    @staticmethod
    def _restore_ingraph_compile_contaminants(restores):
        for item in reversed(restores):
            if item and item[0] == "hooks":
                SingleStreamDiT._restore_forward_hooks(item[1])
                continue
            child, lmm, container, attribute, managed_forward, saved_attrs = item
            for attr, value in saved_attrs.items():
                setattr(child, attr, value)
            if managed_forward is not None:
                if container is not None and attribute is not None:
                    setattr(container, attribute, managed_forward)
                else:
                    child.forward = managed_forward

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
        packs = {}
        for index in streamed_blocks:
            try:
                packs[index] = pack_block_host(
                    f"blocks.{index}",
                    self._block_linear_entries(self.blocks[index]),
                    repoint=False,
                )
            except ValueError as error:
                message = str(error)
                reason = "wrapper_pack_missing" if "wrapper packing" in message else "unsupported_quant_wrapper"
                self._ingraph_unavailable_reasons = (reason,)
                raise RuntimeError(
                    "in-graph sampling unavailable: " + reason + f" ({message})"
                ) from error
        if streamed_blocks and len(packs) != len(streamed_blocks):
            self._ingraph_unavailable_reasons = ("dynamic_streamed_block_set",)
            raise RuntimeError("in-graph sampling unavailable: dynamic_streamed_block_set")
        for pack in packs.values():
            if not pack.pinned:
                self._ingraph_unavailable_reasons = ("non_pinned_pack",)
                raise RuntimeError("in-graph sampling unavailable: non_pinned_pack")
        self._ingraph_sampling_packs = packs
        self._ingraph_unavailable_reasons = ()
        self._ingraph_sampling_depth = max(1, int(depth))
        if getattr(self, "_ingraph_sampling_measure", False):
            self.reset_ingraph_sampling_timing()
        configure_fetch_runtime(depth=self._ingraph_sampling_depth)
        fingerprint = tuple(sorted(packs))
        self._compiled_ingraph_fingerprint = fingerprint
        if compile and packs:
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
        return len(packs)

    def disable_ingraph_sampling(self):
        snapshot = self.ingraph_sampling_timing_snapshot()
        if snapshot is not None:
            self._last_ingraph_sampling_timing = snapshot
        restores = getattr(self, "_ingraph_sampling_restores", [])
        if restores:
            self._restore_ingraph_compile_contaminants(restores)
        self._ingraph_sampling_restores = []
        self._ingraph_sampling_packs = {}
        self._ingraph_unavailable_reasons = ()
        self._ingraph_sampling_depth = 2
        self._compiled_ingraph_sampling = None
        self._compiled_ingraph_fingerprint = None

    @staticmethod
    def _collect_lora_entry(child):
        """LoraEntry from a Linear's LoRA hijack, None if no LoRA, raise if
        a LoRA is present but not expressible as pure traced math."""
        fwd = getattr(child, "__dict__", {}).get("forward")
        owner = getattr(fwd, "__self__", None)
        if owner is None or not hasattr(owner, "lora_down"):
            return None
        network_ref = getattr(owner, "network_ref", None)
        network = network_ref() if network_ref is not None else None
        multiplier = getattr(network, "torch_multiplier", None)
        dropout = getattr(owner, "dropout", None)
        if (
            network is None
            or multiplier is None
            or getattr(multiplier, "numel", lambda: 0)() != 1
            or owner.__class__.__name__ in ("DoRAModule", "LokrModule")
            or getattr(owner, "module_dropout", None) is not None
            or getattr(owner, "rank_dropout", None) not in (None, 0)
            or (dropout is not None and not isinstance(dropout, torch.nn.Identity))
        ):
            raise CompileRegionError(["lora_untraceable"])
        m = float(multiplier.reshape(()).item())
        if m == 0.0:
            # A zero multiplier at enable time would silently bake LoRA out
            # of the trunk for the whole session.
            raise CompileRegionError(["lora_untraceable"])
        return LoraEntry(
            a=owner.lora_down.weight,
            b=owner.lora_up.weight,
            scale=float(owner.scale) * m,
        )

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

    def _make_ingraph_train_block_fn(self, index):
        block = self.blocks[index]
        pack = self._ingraph_training_packs[index]
        loras = self._ingraph_training_loras.get(index, {})
        host = pack.host_flat
        nbytes = int(pack.required_pin_bytes)

        def fn(x, tvec, freqs, mask):
            compiling = torch.compiler.is_compiling()
            if compiling:
                token = torch.ops.mm.fetch_start_after(host, x)
            else:
                token = torch.ops.mm.fetch_start(host)
            flat = torch.ops.mm.fetch_wait(token, nbytes)
            leaves = self._nest_block_train_leaves(
                block_linear_views(flat, pack), loras
            )
            if torch.is_grad_enabled():
                # Saved-token swap: backward frees the recompute generation.
                x = free_on_backward(x, token)
            out = block(x, tvec, freqs, mask, leaves=leaves)
            if not in_recompute():
                if compiling:
                    torch.ops.mm.fetch_free_after(token, out)
                else:
                    torch.ops.mm.fetch_free(token)
            return out

        return fn

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
        Call AFTER the LoRA network is applied and configured (multiplier
        must be a nonzero scalar at enable time)."""
        self.disable_ingraph_training()
        streamed_blocks = tuple(range(len(self.blocks)))
        # Collect LoRA entries BEFORE stripping compile contaminants: the
        # strip deletes instance forwards, including the LoRA hijacks the
        # entries are read from (learned the hard way: 232/512 LoRA grads).
        try:
            loras = {}
            for index in streamed_blocks:
                block_loras = {}
                for name, child in self._block_linear_entries(self.blocks[index]):
                    entry = self._collect_lora_entry(child)
                    if entry is not None:
                        block_loras[name] = entry
                if block_loras:
                    loras[index] = block_loras
        except CompileRegionError as error:
            self._ingraph_unavailable_reasons = error.reasons
            raise RuntimeError(
                "in-graph training unavailable: " + ",".join(error.reasons)
            ) from error
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
        try:
            packs = {}
            for index in streamed_blocks:
                packs[index] = pack_block_host(
                    f"blocks.{index}",
                    self._block_linear_entries(self.blocks[index]),
                    repoint=False,
                )
        except ValueError as error:
            reason = (
                "wrapper_pack_missing"
                if "wrapper packing" in str(error)
                else "unsupported_quant_wrapper"
            )
            self._ingraph_unavailable_reasons = (reason,)
            raise RuntimeError(
                f"in-graph training unavailable: {reason} ({error})"
            ) from error
        for pack in packs.values():
            if not pack.pinned:
                self._ingraph_unavailable_reasons = ("non_pinned_pack",)
                raise RuntimeError("in-graph training unavailable: non_pinned_pack")
        for index in streamed_blocks:
            for _, child in self._block_linear_entries(self.blocks[index]):
                # Keep unmanaged-parameter moves (model.to) off the pack
                # sources: their CPU residency is the design, the trunk
                # streams them from the pinned pack.
                child._mm_ingraph_pack_source = True
        self._ingraph_training_packs = packs
        self._ingraph_training_loras = loras
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
        return len(packs)

    def disable_ingraph_training(self):
        for block in self.blocks:
            for _, child in self._block_linear_entries(block):
                if hasattr(child, "_mm_ingraph_pack_source"):
                    del child._mm_ingraph_pack_source
        restores = getattr(self, "_ingraph_training_restores", [])
        if restores:
            self._restore_ingraph_compile_contaminants(restores)
        self._ingraph_training_restores = []
        self._ingraph_training_packs = {}
        self._ingraph_training_loras = {}
        self._ingraph_training_block_fns = []
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
    ) -> Tensor:
        return self._forward_impl(img, context, t, pos, mask)

    def _forward_impl(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
        *,
        force_ingraph: bool = False,
    ) -> Tensor:
        img = self.first(img)
        t = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)

        txtmask = _mask(mask[:, : context.shape[1]])

        context = self.txtfusion(context, mask=txtmask)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)

        use_compiled = (
            self._compiled_blocks is not None
            and not torch.is_grad_enabled()
        )
        use_ingraph = bool(self._ingraph_sampling_packs) and not torch.is_grad_enabled()
        use_ingraph_train = (
            self._compiled_ingraph_training is not None and torch.is_grad_enabled()
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

        mask = _mask(mask)
        freqs = self.posemb(pos)

        if use_ingraph_train:
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
                tvec,
                freqs,
                mask,
                force_ingraph=force_ingraph,
            )

        final = self.last(combined, t)
        output = final[:, txtlen : txtlen + imglen, :]

        return output

    def _blocks_trunk(
        self,
        combined: Tensor,
        tvec: Tensor,
        freqs: Tensor,
        mask: Tensor | None,
        *,
        force_ingraph: bool = False,
    ) -> Tensor:
        use_compiled = (
            self._compiled_blocks is not None
            and not torch.is_grad_enabled()
        )
        use_ingraph = (
            force_ingraph or bool(self._ingraph_sampling_packs)
        ) and not torch.is_grad_enabled()
        checkpoint_cutoff = len(self.blocks) - self._checkpoint_keep_last
        use_compiled_training = (
            self._compiled_training_blocks is not None
            and torch.is_grad_enabled()
        )
        for i, block in enumerate(self.blocks):
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
                )
            elif use_compiled and self._compiled_blocks[i] is not None:
                # Hook-free block: run its compiled graph. The block that still
                # streams weights falls through to the eager call below.
                combined = self._compiled_blocks[i](combined, tvec, freqs, mask)
            elif use_ingraph and i in self._ingraph_sampling_packs:
                pack = self._ingraph_sampling_packs[i]
                token = torch.ops.mm.fetch_start_after(pack.host_flat, combined)
                flat = torch.ops.mm.fetch_wait(token, int(pack.required_pin_bytes))
                leaves = self._nest_block_leaves(flat, pack)
                combined = block_call(combined, tvec, freqs, mask, leaves=leaves)
                torch.ops.mm.fetch_free_after(token, combined)
            else:
                combined = block_call(combined, tvec, freqs, mask)
        return combined
