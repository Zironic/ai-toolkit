import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers import (
    ZImageTransformer2DModel as DiffusersZImageTransformer2DModel,
)
from diffusers.models.transformers.transformer_z_image import (
    ZSingleStreamAttnProcessor,
)

from ._mixin import OstrisModelMixin


class CacheableZSingleStreamAttnProcessor(ZSingleStreamAttnProcessor):
    """Z-Image attention without an autocast context inside the block graph."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if freqs_cis is not None:
            # The explicit float conversion is sufficient to keep complex RoPE
            # math in float32. A nested autocast context here introduces
            # its context-entry op into AOTAutograd and bypasses MegaCache.
            def apply_rotary_emb(x_in, frequencies):
                value = torch.view_as_complex(
                    x_in.float().reshape(*x_in.shape[:-1], -1, 2)
                )
                frequencies = frequencies.unsqueeze(2)
                return torch.view_as_real(value * frequencies).flatten(3).type_as(
                    x_in
                )

            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(dtype)
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output


class ZImageTransformer2DModel(DiffusersZImageTransformer2DModel, OstrisModelMixin):
    aitk_subfolder = "transformer"
    # repo to pull the config from when loading a single-file checkpoint
    aitk_config_repo = "Tongyi-MAI/Z-Image-Turbo"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for module in self.modules():
            if not isinstance(module, Attention):
                continue
            processor = module.processor
            if type(processor) is not ZSingleStreamAttnProcessor:
                continue
            replacement = CacheableZSingleStreamAttnProcessor()
            replacement._attention_backend = processor._attention_backend
            replacement._parallel_config = processor._parallel_config
            module.set_processor(replacement)

    @classmethod
    def get_quantization_block_names(cls):
        return ["layers"]

    @classmethod
    def get_quantization_exclude_modules(cls):
        # sensitive modules kept in full precision (fnmatch patterns on module
        # names within ZImageTransformer2DModel):
        #   t_embedder*      - timestep embedder; feeds every block's
        #                      adaLN_modulation and the final layers
        #   cap_embedder*    - caption feature -> model width projection
        #   all_x_embedder*  - patchified latent input projections
        #   all_final_layer* - final adaLN-modulated output projections
        #   siglip_embedder* - siglip feature projection (edit models only)
        return [
            "t_embedder*",
            "cap_embedder*",
            "all_x_embedder*",
            "all_final_layer*",
            "siglip_embedder*",
        ]

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        """Convert a single-file Z-Image checkpoint to diffusers transformer keys."""
        new_sd = {}
        for key, value in state_dict.items():
            k = key
            if k.endswith(".attention.qkv.weight"):
                # the single file fuses q,k,v into one tensor (in that order); diffusers keeps them split
                prefix = k[: -len(".attention.qkv.weight")]
                q, k_proj, v = torch.chunk(value, 3, dim=0)
                new_sd[prefix + ".attention.to_q.weight"] = q
                new_sd[prefix + ".attention.to_k.weight"] = k_proj
                new_sd[prefix + ".attention.to_v.weight"] = v
                continue
            k = k.replace(".attention.out.weight", ".attention.to_out.0.weight")
            k = k.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
            k = k.replace(".attention.k_norm.weight", ".attention.norm_k.weight")
            if k.startswith("x_embedder."):
                k = "all_x_embedder.2-1." + k[len("x_embedder.") :]
            elif k.startswith("final_layer."):
                k = "all_final_layer.2-1." + k[len("final_layer.") :]
            new_sd[k] = value
        return new_sd

    @classmethod
    def convert_state_dict_on_save(cls, state_dict):
        """Convert a diffusers transformer state dict back to the single-file layout."""
        new_sd = {}
        qkv_cache = {}
        for key, value in state_dict.items():
            k = key
            matched = False
            for suffix in (
                ".attention.to_q.weight",
                ".attention.to_k.weight",
                ".attention.to_v.weight",
            ):
                if k.endswith(suffix):
                    prefix = k[: -len(suffix)]
                    cache = qkv_cache.setdefault(prefix, {})
                    cache[suffix] = value
                    if len(cache) == 3:
                        # the single file expects q,k,v fused in that order
                        qkv = torch.cat(
                            [
                                cache[".attention.to_q.weight"],
                                cache[".attention.to_k.weight"],
                                cache[".attention.to_v.weight"],
                            ],
                            dim=0,
                        )
                        new_sd[prefix + ".attention.qkv.weight"] = qkv
                        del qkv_cache[prefix]
                    matched = True
                    break
            if matched:
                continue
            k = k.replace(".attention.to_out.0.weight", ".attention.out.weight")
            k = k.replace(".attention.norm_q.weight", ".attention.q_norm.weight")
            k = k.replace(".attention.norm_k.weight", ".attention.k_norm.weight")
            if k.startswith("all_x_embedder.2-1."):
                k = "x_embedder." + k[len("all_x_embedder.2-1.") :]
            elif k.startswith("all_final_layer.2-1."):
                k = "final_layer." + k[len("all_final_layer.2-1.") :]
            new_sd[k] = value
        return new_sd
