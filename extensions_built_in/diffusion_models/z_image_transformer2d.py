"""
Minimal, safe shim for `z_image_transformer2d` used by the vendored
`z_image_transformer2d_control.py`.

This intentionally implements a very small subset of the original API
so the control file can import and perform smoke tests without pulling
all upstream dependencies. **This is a temporary shim**; replace with
upstream implementation for correctness and performance.
"""

import glob
import inspect
import json
import os
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.utils import (USE_PEFT_BACKEND, is_torch_version, logging,
                             scale_lora_layers, unscale_lora_layers)

# Small constants used by the shimmed classes.
ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


def initialize_missing_parameters(missing_keys, model_state_dict, external_state_dict, torch_dtype=None):
    """Create sensible defaults for missing parameters when partially loading weights.

    - Copies matching non-control keys into `control_` counterparts when sizes match.
    - Zeros for projection / bias-like params where appropriate.
    - Ones for normalization weights and running_var, zeros for running_mean.
    - Xavier init for matrix weights and small Gaussian for embeddings.
    """
    initialized = {}
    with torch.no_grad():
        for key in missing_keys:
            target = model_state_dict[key]
            shape = target.shape
            dtype = torch_dtype if torch_dtype is not None else target.dtype

            # 1) Copy from non-control counterpart if present
            if key.startswith("control_"):
                alt = key.replace("control_", "")
                if alt in external_state_dict and external_state_dict[alt].shape == tuple(shape):
                    initialized[key] = external_state_dict[alt].clone().to(dtype)
                    continue

            # 2) Special-case projection initializations
            if "after_proj" in key or "before_proj" in key:
                initialized[key] = torch.zeros(shape, dtype=dtype)
                continue

            # 3) Weight heuristics
            if "weight" in key:
                lname = key.lower()
                if any(norm_type in lname for norm_type in ["norm", "ln_", "layer_norm", "group_norm", "batch_norm"]):
                    initialized[key] = torch.ones(shape, dtype=dtype)
                    continue
                if "embedding" in lname or "embed" in lname:
                    initialized[key] = torch.randn(shape, dtype=dtype) * 0.02
                    continue
                if any(h in lname for h in ["head", "output", "proj_out"]):
                    initialized[key] = torch.zeros(shape, dtype=dtype)
                    continue
                if len(shape) >= 2:
                    t = torch.empty(shape, dtype=dtype)
                    nn.init.xavier_uniform_(t)
                    initialized[key] = t
                    continue
                # fallback
                initialized[key] = torch.randn(shape, dtype=dtype) * 0.02
                continue

            # 4) Biases and stats
            if "bias" in key:
                initialized[key] = torch.zeros(shape, dtype=dtype)
                continue
            if "running_mean" in key:
                initialized[key] = torch.zeros(shape, dtype=dtype)
                continue
            if "running_var" in key:
                initialized[key] = torch.ones(shape, dtype=dtype)
                continue
            if "num_batches_tracked" in key:
                initialized[key] = torch.zeros(shape, dtype=torch.long)
                continue

            # Default fallback
            initialized[key] = torch.zeros(shape, dtype=dtype)
    print(f"initialize_missing_parameters: initialized {len(initialized)} missing keys")
    return initialized


class TimestepEmbedder(nn.Module):
    """Minimal timestep embedding used by the shimmed transformer."""

    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        self.out_size = out_size
        self.mid_size = mid_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, out_size), nn.SiLU())

    def forward(self, t):
        if isinstance(t, torch.Tensor):
            b = t.shape[0]
        else:
            b = 1
        return torch.zeros((b, self.out_size), dtype=torch.float32)


def register_to_config(fn=None, **kwargs):
    """Simple no-op stand-in for diffusers.register_to_config used in shim.

    Returns a decorator when called, or applies directly when used without
    parentheses.
    """
    if fn is None:
        def _decorator(f):
            return f
        return _decorator
    return fn


class ZImageTransformerBlock(nn.Module):
    """Tiny stand-in for the transformer's block.

    Forward is a no-op passthrough; attributes like `.dim` exist so
    downstream code can inspect them.
    """

    def __init__(self, layer_id: int = 0, dim: int = 16, *args, **kwargs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x


class ZImageTransformer2DModel(nn.Module):
    """Minimal transformer base used by control module.

    Provides `in_channels`, basic containers, and a no-op forward. Also
    implements a simple `from_pretrained` classmethod used by the loader.
    """

    def __init__(self, in_channels: int = 16, dim: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.layers = nn.ModuleList([ZImageTransformerBlock(0, dim)])

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x

    @classmethod
    def from_pretrained(cls, name_or_path: Optional[str] = None, **kwargs):
        # Minimal behaviour: ignore name_or_path and instantiate a fresh model.
        return cls()


class FinalLayer(nn.Module):
    """Small final-layer shim implementing an identity-like projection.

    The upstream implementation is more complex; this is just enough for
    import-time and smoke-test usage.
    """

    def __init__(self, in_dim: int = 16, out_dim: int = 16):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if x.ndim > 2:
            b = x.shape[0]
            x_f = x.view(b, -1)
            return self.proj(x_f)
        return self.proj(x)


__all__ = ["ZImageTransformer2DModel", "ZImageTransformerBlock", "FinalLayer"]


class ZSingleStreamAttnProcessor:
    """Minimal stub of the attention processor used only for import/time smoke tests.

    The real implementation is intentionally omitted in this shim; an explicit
   , production-ready implementation should be provided in the complete
    upstream module.
    """

    def __init__(self):
        # Keep construction cheap for tests.
        return

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("ZSingleStreamAttnProcessor is a test stub; production implementation required")


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Keep a minimal feed-forward sufficient for smoke tests.
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


@maybe_allow_in_graph
class ZImageTransformerBlock(nn.Module):
    """Minimal transformer block for shimmed usage by the control adapter.

    This implementation intentionally omits attention internals. It exposes the
    same constructor and `forward` signature the control code calls and
    performs a lightweight, deterministic pass-through / residual feed-forward
    to preserve shapes and behavior during smoke tests.
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        # minimal attributes expected by downstream code
        self.head_dim = max(1, dim // max(1, n_heads))
        self.layer_id = layer_id

        # lightweight modules for a small residual-style update
        self.feed_forward = FeedForward(dim=dim, hidden_dim=max(4, int(dim / 3 * 8)))
        # pass through the normalization epsilon from constructor to match upstream API
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    @property
    def attn_processors(self) -> Dict[str, "AttentionProcessor"]:
        # shim: no attention processors available
        return {}

    def set_attn_processor(self, processor):
        # no-op in shim
        return

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, freqs_cis: Optional[torch.Tensor] = None, adaln_input: Optional[torch.Tensor] = None):
        # Simple, predictable forward: residual feed-forward with optional modulation
        if self.modulation:
            if adaln_input is None:
                # fallback zero vector when caller doesn't pass modulation input
                adaln_input = torch.zeros((x.shape[0], min(self.dim, ADALN_EMBED_DIM)), dtype=x.dtype, device=x.device)
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_mlp = gate_mlp.tanh()
            scale_mlp = 1.0 + scale_mlp
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c=None):
        # Accept optional modulation `c` for compatibility with upstream API.
        if c is None:
            return self.linear(self.norm_final(x))
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)  # complex64
                freqs_cis.append(freqs_cis_i)

            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            # Ensure freqs_cis are on the same device as ids
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


class ZImageTransformer2DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    def __call__(self, x, *args, **kwargs):
        # Simple passthrough entry point used in smoke tests
        return x

    _supports_gradient_checkpointing = True
    # _no_split_modules = ["ZImageTransformerBlock"]
    # _skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]  # precision sensitive layers

    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

        self.sp_world_size = 1
        self.sp_world_rank = 0

        # Allow simple callable usage in smoke tests: treat the instance as a passthrough.
        self.__call__ = lambda x, *a, **k: x

    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")

    def enable_multi_gpus_inference(self,):
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather
        for layer in self.layers:
            layer.set_attn_processor(ZMultiGPUsSingleStreamAttnProcessor())

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify(
        self,
        all_image: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
        cap_padding_len: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []

        for i, image in enumerate(all_image):
            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_image_size,
            all_image_pos_ids,
            all_image_pad_mask,
        )

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            ### Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            all_cap_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            cap_padded_feat = torch.cat(
                [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
                dim=0,
            )
            all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        # Match t_embedder output dtype to x for layerwise casting compatibility
        adaln_input = t.type_as(x)
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.noise_refiner:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x, x_attn_mask, x_freqs_cis, adaln_input,
                    **ckpt_kwargs,
                )
        else:
            for layer in self.noise_refiner:
                x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.context_refiner:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                cap_feats = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    cap_feats, 
                    cap_attn_mask, 
                    cap_freqs_cis,
                    **ckpt_kwargs,
                )
        else:
            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]

            x_item_seqlens = [len(_) for _ in x]
            assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
            x_max_item_seqlen = max(x_item_seqlens)
            x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(x_item_seqlens):
                x_attn_mask[i, :seq_len] = 1

            if x_freqs_cis is not None:
                x_freqs_cis = torch.chunk(x_freqs_cis, self.sp_world_size, dim=1)[self.sp_world_rank]

        # unified
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)
        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.layers:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                unified = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    unified, 
                    unified_attn_mask, 
                    unified_freqs_cis, 
                    adaln_input,
                    **ckpt_kwargs,
                )
        else:
            for layer in self.layers:
                unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        if self.sp_world_size > 1:
            unified_out = []
            for i in range(bsz):
                x_len = x_item_seqlens[i]
                unified_out.append(unified[i, :x_len])
            unified = torch.stack(unified_out)
            unified = self.all_gather(unified, dim=1)
            
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        x = torch.stack(x)
        return x, {}
    

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        subfolder: Optional[str] = None,
        transformer_additional_kwargs: Optional[Dict[str, Any]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        low_cpu_mem_usage: bool = True,
        load_control_only: bool = False,
        materialize_device: str = 'cpu',
        base_transformer_path: Optional[str] = None,
    ):
        """Workspace-friendly `from_pretrained` for the shim.

        Behaviour:
        - Prefer local directories: expects `config.json` and optional weight files
          (`*.safetensors`, `pytorch_model.bin`). If none are found, the method will
          instantiate the model from the config and return it (weights uninitialized).
        - Keep implementation small and deterministic (no external HF downloads,
          no accelerate/flash-attn assumptions).
        - Provide clear, actionable errors when paths are missing or invalid.
        
        Args:
            base_transformer_path: Path to base transformer weights. Required for 
                control-only checkpoints (<8GB) which only contain delta weights.
                The base weights are loaded first, then control weights override them.
        
        Args:
            materialize_device: Device to load weights onto (default 'cpu'). 
                               Set to 'cuda' to load directly to GPU (avoids CPU->GPU copy).
        """
        transformer_additional_kwargs = transformer_additional_kwargs or {}

        if pretrained_model_path is None:
            # No path -> return a fresh instance using any provided kwargs
            try:
                return cls(**transformer_additional_kwargs)
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate {cls.__name__} without a path: {e}") from e

        model_dir = pretrained_model_path
        if subfolder is not None:
            model_dir = os.path.join(model_dir, subfolder)

        if not os.path.isdir(model_dir):
            raise RuntimeError(f"from_pretrained expects a local directory path, got: {pretrained_model_path}")

        # Load config.json if present
        config_file = os.path.join(model_dir, "config.json")
        config = {}
        if os.path.isfile(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
        else:
            # Use provided kwargs as a minimal config if none present
            if transformer_additional_kwargs:
                config = dict(transformer_additional_kwargs)
            else:
                print(f"Warning: no config.json in {model_dir}; instantiating with defaults")

        # Find weight files first
        weight_files = []
        for ext in ("*.safetensors", "pytorch_model.bin", "*.bin"):
            found = glob.glob(os.path.join(model_dir, ext))
            if found:
                weight_files.extend(found)
        
        # Detect control_in_dim from checkpoint if not already set
        # Z-Image standard: control_tensor(16) + mask(1) + latent_tensor(16) = 33
        # Z-Image Union: 4 control types * 33 = 132
        safetensors = [p for p in weight_files if p.endswith(".safetensors")]
        detected_control_in_dim = None
        if safetensors:
            try:
                from safetensors import safe_open
                for sf_path in safetensors:
                    with safe_open(sf_path, framework='pt', device='cpu') as f:
                        for k in f.keys():
                            # Look for control embedder weight to detect control_in_dim
                            if 'control' in k and 'embedder' in k and 'weight' in k:
                                shape = f.get_slice(k).get_shape()
                                if len(shape) == 2:
                                    detected_control_in_dim = shape[1]
                                    print(f"from_pretrained: detected control_in_dim={detected_control_in_dim} from {k}")
                                    break
                    if detected_control_in_dim:
                        break
            except Exception as e:
                print(f"from_pretrained: failed to detect control_in_dim: {e}")
        
        # Set control_in_dim - prefer detected, then config, then default 33
        final_control_in_dim = detected_control_in_dim or config.get('control_in_dim') or (transformer_additional_kwargs or {}).get('control_in_dim') or 33
        if config.get('control_in_dim') != final_control_in_dim:
            print(f"from_pretrained: setting control_in_dim={final_control_in_dim}")
            config['control_in_dim'] = final_control_in_dim
        if transformer_additional_kwargs is None:
            transformer_additional_kwargs = {}
        transformer_additional_kwargs['control_in_dim'] = final_control_in_dim

        # Optionally construct model with accelerate.init_empty_weights to avoid allocating
        # full parameter tensors. Do this *before* any normal instantiation when
        # `low_cpu_mem_usage=True` so we don't allocate large tensors inadvertently.
        model = None
        if low_cpu_mem_usage:
            try:
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate
                    print(f"from_pretrained: [1/4] using accelerate.init_empty_weights to construct {cls.__name__} (no memory allocation)")
                    print(f"from_pretrained: config control_in_dim={config.get('control_in_dim')}")
                    with accelerate.init_empty_weights():
                        model = cls.from_config(config, **(transformer_additional_kwargs or {}))
                    print(f"from_pretrained: [1/4] meta model created successfully")
                else:
                    print("from_pretrained: accelerate available but is_accelerate_available() returned False")
            except Exception as e:
                # accelerate not available or failed; fall back to normal instantiation
                print(f"from_pretrained: accelerate.init_empty_weights failed: {e}; falling back to normal constructor")

        # Instantiate model normally (if empty-weight construction didn't succeed)
        if model is None:
            print(f"from_pretrained: [1/4] WARNING - using normal constructor (will allocate ~20GB)")
            if hasattr(cls, "from_config") and callable(getattr(cls, "from_config")):
                try:
                    model = cls.from_config(config, **(transformer_additional_kwargs or {}))
                except Exception as e:
                    print(f"from_config failed for {cls.__name__}; falling back to constructor: {e}")
            if model is None:
                try:
                    # Try to extract matching constructor args from config/kwargs
                    ctor_kwargs = {}
                    ctor_kwargs.update(transformer_additional_kwargs)
                    # Common keys
                    for k in ("in_channels", "dim", "n_layers", "n_heads", "n_refiner_layers"):
                        if k in config and k not in ctor_kwargs:
                            ctor_kwargs[k] = config[k]
                    model = cls(**ctor_kwargs)
                except Exception as e:
                    raise RuntimeError(f"Failed to construct model {cls.__name__}: {e}") from e

        if not weight_files:
            # No loadable weights found; return the instantiated model
            print(f"No weights found in {model_dir}; returning model instantiated from config")
            if torch_dtype is not None:
                model = model.to(torch_dtype)
            return model

        # Check if model has meta tensors (from init_empty_weights)
        has_meta = any(p.device.type == 'meta' for p in model.parameters())
        
        # Count checkpoint keys to determine if this is a complete control model or a delta
        # A complete Z-Image control transformer has ~295 control-specific keys
        # A delta checkpoint would have fewer keys and need base weights
        total_file_size = sum(os.path.getsize(f) for f in weight_files)
        checkpoint_key_count = 0
        safetensors_files = [p for p in weight_files if p.endswith(".safetensors")]
        if safetensors_files:
            try:
                from safetensors import safe_open
                for sf in safetensors_files:
                    with safe_open(sf, framework='pt', device='cpu') as f:
                        checkpoint_key_count += len(f.keys())
            except:
                pass
        
        # A complete control transformer has 250+ keys - doesn't need base weights
        # A delta checkpoint would have <200 keys and need base weights
        is_complete_control_model = checkpoint_key_count >= 250
        needs_base_weights = not is_complete_control_model and total_file_size < 15 * 1024**3
        
        print(f"from_pretrained: checkpoint has {checkpoint_key_count} keys, size={total_file_size/(1024**3):.2f}GB")
        print(f"from_pretrained: is_complete_control_model={is_complete_control_model}, needs_base_weights={needs_base_weights}")
        
        # For meta tensor models loading to GPU, use streaming to avoid loading entire file to RAM
        if has_meta and materialize_device != 'cpu':
            print(f"from_pretrained: [2/4] streaming weights directly to {materialize_device}...")
            safetensors = safetensors_files
            if safetensors:
                try:
                    from safetensors import safe_open
                    from accelerate.utils import set_module_tensor_to_device
                    import gc
                    
                    model_state = model.state_dict()
                    loaded_keys = set()
                    
                    # Only load base weights if this is a delta checkpoint
                    if needs_base_weights:
                        print(f"from_pretrained: [2.5/4] delta checkpoint detected, loading base weights first...")
                        # Use explicitly provided base_transformer_path if available
                        base_path = None
                        if base_transformer_path:
                            if os.path.isdir(base_transformer_path):
                                base_sfs = glob.glob(os.path.join(base_transformer_path, "*.safetensors"))
                                if base_sfs:
                                    base_path = base_transformer_path
                                    print(f"from_pretrained: using provided base_transformer_path: {base_path}")
                            elif os.path.isfile(base_transformer_path):
                                # Single file provided
                                base_path = os.path.dirname(base_transformer_path)
                                print(f"from_pretrained: using provided base_transformer_path dir: {base_path}")
                        
                        # Fallback: try to find base transformer in parent directory
                        if not base_path:
                            parent_dir = os.path.dirname(model_dir.rstrip('/\\'))
                            transformer_dir = os.path.join(parent_dir, 'transformer')
                            if os.path.isdir(transformer_dir):
                                base_sfs = glob.glob(os.path.join(transformer_dir, "*.safetensors"))
                                if base_sfs:
                                    base_path = transformer_dir
                                    print(f"from_pretrained: found base transformer at {base_path}")
                        
                        if base_path:
                            # Stream base weights to device first
                            base_sfs = glob.glob(os.path.join(base_path, "*.safetensors"))
                            for p in base_sfs:
                                with safe_open(p, framework='pt', device='cpu') as f:
                                    for k in f.keys():
                                        if k in model_state and model_state[k].size() == f.get_slice(k).get_shape():
                                            t = f.get_tensor(k)
                                            if torch_dtype is not None and t.dtype != torch_dtype:
                                                t = t.to(dtype=torch_dtype)
                                            set_module_tensor_to_device(model, k, materialize_device, value=t)
                                            loaded_keys.add(k)
                                            del t
                            print(f"from_pretrained: [2.5/4] loaded {len(loaded_keys)} base keys to {materialize_device}")
                        else:
                            print(f"from_pretrained: WARNING - delta checkpoint but no base transformer found!")
                            print(f"from_pretrained: Provide base_transformer_path arg or place transformer/ dir next to controlnet")
                    
                    # Now load checkpoint weights directly
                    # For complete control models, these are all the weights needed
                    # For delta checkpoints, these overwrite base weights for control-specific params
                    for p in safetensors:
                        with safe_open(p, framework='pt', device='cpu') as f:
                            for k in f.keys():
                                # Skip non-control keys if load_control_only
                                if load_control_only and "control" not in k:
                                    continue
                                
                                # Check if key matches model directly
                                if k in model_state and model_state[k].size() == f.get_slice(k).get_shape():
                                    # Load tensor, convert dtype if needed, send directly to device
                                    t = f.get_tensor(k)
                                    if torch_dtype is not None and t.dtype != torch_dtype:
                                        t = t.to(dtype=torch_dtype)
                                    set_module_tensor_to_device(model, k, materialize_device, value=t)
                                    loaded_keys.add(k)
                                    del t
                    
                    # GC to free any intermediate memory
                    gc.collect()
                    
                    print(f"from_pretrained: [4/4] loaded total {len(loaded_keys)} keys to {materialize_device}")
                    
                    # For remaining meta tensors, materialize to CPU with zeros (NOT GPU)
                    # This keeps GPU memory minimal - only actual weights go to GPU
                    remaining_meta = [n for n, p in model.named_parameters() if p.device.type == 'meta']
                    if remaining_meta:
                        if needs_base_weights and len(remaining_meta) > 50:
                            # Something is wrong - delta checkpoint should have loaded base weights
                            print(f"from_pretrained: WARNING - {len(remaining_meta)} params still meta after delta load!")
                            print(f"from_pretrained: This suggests base transformer weights were not found.")
                            print(f"from_pretrained: Model may not function correctly without base weights.")
                        print(f"from_pretrained: {len(remaining_meta)} params not in checkpoint, initializing to CPU (not GPU)")
                        for name in remaining_meta:
                            param = model_state[name]
                            # Initialize on CPU to save GPU memory - these are unused anyway
                            zeros = torch.zeros(param.shape, dtype=torch_dtype or torch.float32, device='cpu')
                            set_module_tensor_to_device(model, name, 'cpu', value=zeros)
                    
                    return model
                except Exception as e:
                    import traceback
                    print(f"from_pretrained: streaming load failed: {e}")
                    traceback.print_exc()
                    print("from_pretrained: falling back to standard load")
        
        # Standard path: load state_dict then transfer
        print(f"from_pretrained: [2/4] loading weights from disk...")
        state_dict = None
        try:
            safetensors = [p for p in weight_files if p.endswith(".safetensors")]
            bin_files = [p for p in weight_files if p.endswith(".bin") or p.endswith('.pth')]


            # Prefer safetensors; they allow incremental loading and are generally
            # faster for selective key extraction.
            if safetensors:
                print(f"from_pretrained: [2/4] loading from safetensors: {safetensors}")
                try:
                    from safetensors.torch import load_file

                    if load_control_only:
                        # Load and filter on the fly, convert to target dtype only if needed
                        state_dict = {}
                        for p in safetensors:
                            sd = load_file(p)
                            for k, v in sd.items():
                                if ("control" in k) or (not load_control_only):
                                    # Only convert if dtype doesn't match (avoid unnecessary copy)
                                    if torch_dtype is not None and v.dtype != torch_dtype:
                                        state_dict[k] = v.to(dtype=torch_dtype)
                                    else:
                                        state_dict[k] = v
                    else:
                        state_dict = {}
                        for p in safetensors:
                            sd = load_file(p)
                            for k, v in sd.items():
                                # Only convert if dtype doesn't match (avoid unnecessary copy)
                                if torch_dtype is not None and v.dtype != torch_dtype:
                                    state_dict[k] = v.to(dtype=torch_dtype)
                                else:
                                    state_dict[k] = v
                    print(f"from_pretrained: [2/4] loaded {len(state_dict)} keys from safetensors")
                except Exception as e:
                    print(f"Failed to read safetensors from {model_dir}: {e}")
                    state_dict = None
            elif bin_files:
                # Bins can be large; try to load only when necessary and warn.
                try:
                    if load_control_only:
                        raw = torch.load(bin_files[0], map_location="cpu")
                        state_dict = {}
                        for k, v in raw.items():
                            if ("control" in k) or (not load_control_only):
                                # Only convert if dtype doesn't match
                                if torch_dtype is not None and hasattr(v, 'dtype') and v.dtype != torch_dtype:
                                    state_dict[k] = v.to(dtype=torch_dtype)
                                else:
                                    state_dict[k] = v
                        # free raw if possible
                        del raw
                    else:
                        raw = torch.load(bin_files[0], map_location="cpu")
                        state_dict = {}
                        for k, v in raw.items():
                            # Only convert if dtype doesn't match
                            if torch_dtype is not None and hasattr(v, 'dtype') and v.dtype != torch_dtype:
                                state_dict[k] = v.to(dtype=torch_dtype)
                            else:
                                state_dict[k] = v
                        del raw
                except Exception as e:
                    print(f"Failed to load bin weights from {model_dir}: {e}")
                    state_dict = None
        except Exception as e:
            print(f"Error scanning weight files in {model_dir}: {e}")
            state_dict = None

        if state_dict is None:
            print(f"No usable weights found in {model_dir}; returning model instantiated from config")
            if torch_dtype is not None:
                model = model.to(torch_dtype)
            return model

        # Filter and load matching keys only (and try to map control_ keys)
        # NOTE: We avoid creating copies - just reference same tensors from state_dict
        print(f"from_pretrained: [3/4] filtering keys to match model...")
        model_state = model.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if k in model_state and model_state[k].size() == v.size():
                filtered[k] = v

        # Try control->non-control key mapping e.g., control.weight <- weight
        # NOTE: No .clone() - just reference the same tensor
        for k in list(model_state.keys()):
            if k not in filtered and k.startswith("control_"):
                alt = k.replace("control_", "")
                if alt in state_dict and model_state[k].size() == state_dict[alt].size():
                    filtered[k] = state_dict[alt]  # No clone - same tensor

        # Initialize missing parameters to sensible defaults to support partial loads
        missing_keys = [k for k in model_state.keys() if k not in filtered]
        if missing_keys:
            init_params = initialize_missing_parameters(missing_keys, model_state, state_dict, torch_dtype=torch_dtype)
            # Merge in initializations, preferring already-loaded keys
            for k, v in init_params.items():
                if k not in filtered:
                    filtered[k] = v

        print(f"from_pretrained: [3/4] filtered to {len(filtered)} keys, freeing unused...")
        # Free state_dict keys we won't use to reduce peak memory
        keys_to_keep = set(filtered.keys())
        for k in list(state_dict.keys()):
            if k not in keys_to_keep:
                del state_dict[k]
        
        # GC before GPU transfer to free any intermediate allocations
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Check if model has meta tensors (from init_empty_weights)
        has_meta = any(p.device.type == 'meta' for p in model.parameters())
        print(f"from_pretrained: [4/4] loading into model (has_meta={has_meta}, target_device={materialize_device})...")
        
        if has_meta:
            # Use accelerate's set_module_tensor_to_device for proper meta tensor handling
            try:
                from accelerate.utils import set_module_tensor_to_device
                for name, param in filtered.items():
                    set_module_tensor_to_device(model, name, materialize_device, value=param)
                    del param  # Free the tensor after it's assigned
                # Clear filtered dict and GC
                filtered.clear()
                gc.collect()
                torch.cuda.empty_cache()
                print(f"from_pretrained: [4/4] materialized keys from meta tensors to {materialize_device} using accelerate")
            except ImportError:
                # Fallback: use to_empty + load_state_dict
                print("from_pretrained: accelerate.set_module_tensor_to_device not available, using to_empty fallback")
                model.to_empty(device=materialize_device)
                model.load_state_dict(filtered, strict=False)
        else:
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print(f"from_pretrained: loaded {len(filtered)} keys; missing: {len(missing)}; unexpected: {len(unexpected)}")
            # Free filtered before device transfer
            filtered.clear()
            gc.collect()
            # Move to target device if not already there
            if materialize_device != 'cpu':
                model = model.to(materialize_device)

        # Don't call model.to(dtype) - tensors are already the right dtype from loading
        # Calling .to() would create a full copy of the model, doubling memory
        return model