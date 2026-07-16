"""Architecture profiles for scripts/smoke_transformer_train_cuda.py.

Each profile supplies ONLY what genuinely differs between architectures:
how to build the ModelConfig, construct the model, load the transformer,
validate/batch cached conditioning, and build valid fake latents. Arena
attachment, block discovery, execution, phase transitions, optimizer setup,
telemetry, and teardown are owned by the shared runner.

Profiles must NOT become architecture adapters under another name: no block
paths, no leaf paths, no block execution, no quantizer operations, no
runtime checkpoint policy. (`arena_ignore_modules` is the one concession:
which permanent tokens/projections stay out of the arena is model knowledge
the runner cannot derive until the generic dispatcher's state classification
lands -- see tasks/done/GENERIC_BLOCK_DISPATCHER_PLAN.md.)

Fixed mapping, no plugin registry:

    PROFILES = {"krea2": ..., "zimage": ..., "ideogram4": ..., "anima": ...}
"""

from __future__ import annotations

from pathlib import Path

import torch

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import ModelConfig
from toolkit.prompt_utils import PromptEmbeds

# ---------------------------------------------------------------------------
# Quantized-representation audit (shared by the runner and the contract smoke)
# ---------------------------------------------------------------------------

def audit_quantized_representation(module: torch.nn.Module) -> dict:
    """Count the quantized weight representations present under `module`.

    Returns observed counts for optimum.quanto QBytesTensor weights, torchao
    Float8Tensor weights, OstrisLinear modules (with per-qtype counts), plain
    Linear modules, and declared physical payload bytes where they are cheap
    to read (quanto qdata/scale, ostris registered buffers).
    """
    try:
        from optimum.quanto.tensor.qbytes import QBytesTensor
    except Exception:
        QBytesTensor = ()  # sentinel: isinstance(x, ()) is False
    try:
        from torchao.quantization import Float8Tensor
    except Exception:
        Float8Tensor = ()  # sentinel: isinstance(x, ()) is False
    from toolkit.util.ostris_quant import OstrisLinear

    counts = {
        "qbytes_tensor": 0,
        "float8_tensor": 0,
        "torchao_subclass": 0,
        "ostris_linear": 0,
        "plain_linear": 0,
        "ostris_qtype_counts": {},
        "quanto_qdata_bytes": 0,
        "quanto_scale_bytes": 0,
        "ostris_buffer_bytes": 0,
        "logical_dense_bytes": 0,
    }
    for m in module.modules():
        if isinstance(m, OstrisLinear):
            counts["ostris_linear"] += 1
            qtype = getattr(m.ostris_quantizer, "qtype", None)
            counts["ostris_qtype_counts"][str(qtype)] = (
                counts["ostris_qtype_counts"].get(str(qtype), 0) + 1
            )
            for _, buf in m.named_buffers(recurse=False):
                counts["ostris_buffer_bytes"] += buf.numel() * buf.element_size()
            counts["logical_dense_bytes"] += (
                m.in_features * m.out_features * 2  # bf16 reference payload
            )
            continue
        if isinstance(m, torch.nn.Linear):
            weight = m._parameters.get("weight", None)
            if weight is None:
                continue
            data = weight.data if isinstance(weight, torch.nn.Parameter) else weight
            if QBytesTensor and isinstance(data, QBytesTensor):
                counts["qbytes_tensor"] += 1
                qdata = getattr(data, "_data", None)
                scale = getattr(data, "_scale", None)
                if qdata is not None:
                    counts["quanto_qdata_bytes"] += qdata.numel() * qdata.element_size()
                if scale is not None:
                    counts["quanto_scale_bytes"] += scale.numel() * scale.element_size()
            elif Float8Tensor and isinstance(data, Float8Tensor):
                counts["float8_tensor"] += 1
            elif "torchao" in type(data).__module__:
                counts["torchao_subclass"] += 1
            else:
                counts["plain_linear"] += 1
            counts["logical_dense_bytes"] += m.in_features * m.out_features * 2
    return counts


def assert_representation(counts: dict, requested_qtype: str) -> list:
    """The plan's hard representation gates, as a list of failure strings."""
    failures = []
    if requested_qtype == "qfloat8":
        if counts["qbytes_tensor"] == 0:
            failures.append("qfloat8 requested but no QBytesTensor weights found")
        if counts["float8_tensor"] > 0:
            failures.append(
                "qfloat8 requested but torchao Float8Tensor weights present "
                f"({counts['float8_tensor']}) -- the qtype was silently replaced"
            )
        if counts["ostris_linear"] > 0:
            failures.append("qfloat8 requested but OstrisLinear modules present")
    elif requested_qtype == "float8":
        if counts["float8_tensor"] == 0:
            failures.append("float8 requested but no torchao Float8Tensor weights")
    elif requested_qtype.startswith(("convrot", "orbit")):
        if counts["ostris_linear"] == 0:
            failures.append(
                f"{requested_qtype} requested but no OstrisLinear modules found"
            )
        wrong = {
            q: n
            for q, n in counts["ostris_qtype_counts"].items()
            if q != requested_qtype
        }
        if wrong:
            failures.append(
                f"OstrisLinear qtype mismatch: requested {requested_qtype}, "
                f"observed {wrong}"
            )
    return failures


# ---------------------------------------------------------------------------
# Conditioning loader (generic)
# ---------------------------------------------------------------------------

def load_condition_caches(paths):
    """Load each --cond-cache through PromptEmbeds.load (which dispatches to
    AdvancedPromptEmbeds via safetensors metadata). Returns the raw list; the
    profile validates and batches."""
    loaded = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        loaded.append(PromptEmbeds.load(str(p)))
    if not loaded:
        raise SystemExit("at least one --cond-cache is required")
    return loaded


def _require_plain_embeds(caches, profile_name):
    for c in caches:
        if isinstance(c, AdvancedPromptEmbeds):
            raise SystemExit(
                f"profile {profile_name} expects ordinary PromptEmbeds caches, "
                "got an AdvancedPromptEmbeds file"
            )
    return caches


# ---------------------------------------------------------------------------
# Profile contract
# ---------------------------------------------------------------------------

class SmokeProfile:
    """Narrow architecture contract for the shared runner. See module docstring."""

    name: str = ""
    expected_block_container: str = ""

    def build_model_config(self, args) -> ModelConfig:
        raise NotImplementedError

    def construct_model(self, args, config):
        raise NotImplementedError

    def load_transformer(self, model, args):
        raise NotImplementedError

    def arena_ignore_modules(self, transformer) -> list:
        return []

    def load_conditioning(self, paths, batch_size):
        raise NotImplementedError

    def make_latents(self, resolution, batch_size, generator, model) -> torch.Tensor:
        raise NotImplementedError

    def default_fullmodule_target(self, transformer) -> str:
        raise NotImplementedError

    def enable_model_checkpointing(self, transformer, keep_last=0) -> bool:
        """Turn on model-owned gradient checkpointing if the architecture has
        it. Returns True when something was enabled."""
        enable = getattr(transformer, "enable_gradient_checkpointing", None)
        if callable(enable):
            if keep_last:
                try:
                    enable(keep_last=keep_last)
                except TypeError as error:
                    raise SystemExit(
                        f"profile {self.name} does not support checkpoint keep-last"
                    ) from error
            else:
                enable()
            return True
        if hasattr(transformer, "gradient_checkpointing"):
            transformer.gradient_checkpointing = True
            return True
        return False

    def prediction_shape_reference(self, latents: torch.Tensor):
        """The shape the noise prediction must match for these latents."""
        return tuple(latents.shape)

    def _shared_offload_config_kwargs(self, args) -> dict:
        return dict(
            dtype=args.dtype,
            quantize=True,
            qtype=args.qtype,
            layer_offloading=True,
            layer_offloading_smart=True,
            layer_offloading_smart_working_reserve_gb=args.working_reserve_gib,
            layer_offloading_smart_wddm_margin_gb=args.wddm_margin_gib,
            layer_offloading_smart_wddm_hard_gb=args.wddm_hard_gib,
            layer_offloading_simulated_vram_gb=args.simulated_vram_gib,
            layer_offloading_checkpoint_keep_last=args.checkpoint_keep_last,
            layer_offloading_prefetch_depth=args.prefetch_depth,
            compile=not args.no_compile,
            compile_dynamic=args.compile_dynamic_resolved,
            compile_fullgraph=args.compile_fullgraph,
            compile_coordinate_descent=args.compile_coordinate_descent_resolved,
        )


class Krea2SmokeProfile(SmokeProfile):
    name = "krea2"
    expected_block_container = "blocks"

    def build_model_config(self, args) -> ModelConfig:
        model_kwargs = {
            "max_text_length": args.max_text_length,
            "local_files_only": not args.allow_download,
        }
        if args.cache_dir:
            model_kwargs["quantized_transformer_cache_dir"] = args.cache_dir
        return ModelConfig(
            name_or_path=args.model_path or "krea/Krea-2-Raw",
            arch="krea2",
            model_kwargs=model_kwargs,
            **self._shared_offload_config_kwargs(args),
        )

    def construct_model(self, args, config):
        from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model

        model = Krea2Model(device=args.device, model_config=config, dtype=args.dtype)
        model.skip_te = True
        return model

    def load_transformer(self, model, args):
        transformer = model._load_transformer()
        model.model = transformer
        return transformer

    def arena_ignore_modules(self, transformer) -> list:
        from extensions_built_in.diffusion_models.krea2.krea2 import (
            DoubleSharedModulation,
            SimpleModulation,
        )

        return [
            module
            for module in transformer.modules()
            if isinstance(module, (SimpleModulation, DoubleSharedModulation))
        ]

    def load_conditioning(self, paths, batch_size):
        caches = load_condition_caches(paths)
        embeds = caches[0]
        text_embeds = embeds.text_embeds
        if isinstance(text_embeds, torch.Tensor):
            if text_embeds.dim() == 3 and text_embeds.shape[0] == 1:
                text_embeds = text_embeds[0]
            if text_embeds.dim() != 2:
                raise SystemExit(
                    f"unsupported krea2 text_embeds shape {tuple(text_embeds.shape)}"
                )
            embeds.text_embeds = [text_embeds]
        embeds.text_embeds = [embeds.text_embeds[0]] * batch_size
        return embeds

    def make_latents(self, resolution, batch_size, generator, model) -> torch.Tensor:
        width, height = resolution
        return torch.randn(
            batch_size, 16, height // 8, width // 8, generator=generator
        )

    def default_fullmodule_target(self, transformer) -> str:
        return "blocks.0.attn.wq"


class ZImageSmokeProfile(SmokeProfile):
    name = "zimage"
    expected_block_container = "layers"

    def build_model_config(self, args) -> ModelConfig:
        if not args.model_path:
            raise SystemExit("--model-path is required for the zimage profile")
        return ModelConfig(
            name_or_path=args.model_path,
            arch="zimage",
            assistant_lora_path=args.assistant_lora,
            **self._shared_offload_config_kwargs(args),
        )

    def construct_model(self, args, config):
        from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel

        model = ZImageModel(device=args.device, model_config=config, dtype=args.dtype)
        model.skip_te = True
        return model

    def load_transformer(self, model, args):
        # The universal ZImage loader normally quantizes before returning. In
        # smoke-direct mode the shared runner owns blockwise quantization so it
        # can publish each final representation straight into canonical arena
        # storage and release the source block immediately.
        direct_arena = bool(getattr(model, "_smoke_direct_arena_load", False))
        quantize_requested = bool(model.model_config.quantize)
        if direct_arena:
            model.model_config.quantize = False
        try:
            transformer, _base = model.load_transformer(
                model.model_config.name_or_path,
                model.model_config.extras_name_or_path,
                model.torch_dtype,
            )
        finally:
            model.model_config.quantize = quantize_requested
        if model.model_config.assistant_lora_path is not None:
            # Documented behaviour: the assistant adapter is merged before
            # quantization and converts qfloat8 -> float8. The runner asserts
            # the conversion instead of claiming to test Quanto.
            model.load_training_adapter(transformer)
            if model.model_config.qtype == "qfloat8":
                model.model_config.qtype = "float8"
        model._transformer_quantized_during_load = bool(
            getattr(transformer, "aitk_is_quantized", False)
        )
        model.model = transformer
        return transformer

    def arena_ignore_modules(self, transformer) -> list:
        return [transformer.x_pad_token, transformer.cap_pad_token]

    def load_conditioning(self, paths, batch_size):
        caches = _require_plain_embeds(load_condition_caches(paths), self.name)
        embeds = caches[0]
        text_embeds = embeds.text_embeds
        if isinstance(text_embeds, torch.Tensor):
            if text_embeds.dim() == 3 and text_embeds.shape[0] == 1:
                text_embeds = text_embeds[0]
            embeds.text_embeds = [text_embeds]
        embeds.text_embeds = [embeds.text_embeds[0]] * batch_size
        return embeds

    def make_latents(self, resolution, batch_size, generator, model) -> torch.Tensor:
        width, height = resolution
        transformer = model.model
        channels = 16
        config = getattr(transformer, "config", None)
        if config is not None:
            channels = int(getattr(config, "in_channels", channels))
        return torch.randn(
            batch_size, channels, height // 8, width // 8, generator=generator
        )

    def default_fullmodule_target(self, transformer) -> str:
        return "layers.0.attention.to_q"


class Ideogram4SmokeProfile(SmokeProfile):
    name = "ideogram4"
    expected_block_container = "layers"

    def build_model_config(self, args) -> ModelConfig:
        if not args.model_path:
            raise SystemExit("--model-path is required for the ideogram4 profile")
        return ModelConfig(
            name_or_path=args.model_path,
            arch="ideogram4",
            model_kwargs={"max_text_length": args.max_text_length},
            **self._shared_offload_config_kwargs(args),
        )

    def construct_model(self, args, config):
        from extensions_built_in.diffusion_models.ideogram4.ideogram4 import (
            Ideogram4Model,
        )

        model = Ideogram4Model(
            device=args.device, model_config=config, dtype=args.dtype
        )
        model.skip_te = True
        return model

    def load_transformer(self, model, args):
        transformer = model._load_transformer(model.model_config.name_or_path)
        model.model = transformer
        return transformer

    def arena_ignore_modules(self, transformer) -> list:
        # input_proj / llm_cond_proj are permanent (outside the repeated
        # blocks); the rotary inv_freq is a non-persistent buffer and must not
        # be treated as managed immutable block storage.
        return [transformer.input_proj, transformer.llm_cond_proj]

    def load_conditioning(self, paths, batch_size):
        caches = load_condition_caches(paths)
        for c in caches:
            if not isinstance(c, AdvancedPromptEmbeds):
                raise SystemExit(
                    "ideogram4 requires AdvancedPromptEmbeds caches "
                    "(safetensors with class_name metadata)"
                )
        if len(caches) == 1:
            return caches[0].expand_to_batch(batch_size)
        if len(caches) != batch_size:
            raise SystemExit(
                f"got {len(caches)} ideogram4 caches for batch size {batch_size}; "
                "pass one cache (expanded) or exactly batch-size caches "
                "(kept separate to preserve natural token lengths)"
            )
        # Multiple caches stay separate samples so unequal token lengths reach
        # the model and exercise its call-time padding.
        return AdvancedPromptEmbeds.concat_prompt_embeds(caches)

    def make_latents(self, resolution, batch_size, generator, model) -> torch.Tensor:
        width, height = resolution
        # 32 VAE channels at /8, patchified 2x2 -> 128 channels at /16.
        return torch.randn(
            batch_size, 128, height // 16, width // 16, generator=generator
        )

    def default_fullmodule_target(self, transformer) -> str:
        return "layers.0.attention.to_q"

    def enable_model_checkpointing(self, transformer, keep_last=0) -> bool:
        enable = getattr(transformer, "enable_gradient_checkpointing", None)
        if callable(enable):
            if keep_last:
                try:
                    enable(keep_last=keep_last)
                except TypeError as error:
                    raise SystemExit(
                        f"profile {self.name} does not support checkpoint keep-last"
                    ) from error
            else:
                enable()
            return True
        if hasattr(transformer, "gradient_checkpointing"):
            transformer.gradient_checkpointing = True
            return True
        return False


class AnimaSmokeProfile(SmokeProfile):
    name = "anima"
    expected_block_container = "transformer_blocks"

    def build_model_config(self, args) -> ModelConfig:
        return ModelConfig(
            name_or_path=(
                args.model_path
                or "circlestone-labs/Anima-Base-v1.0-Diffusers"
            ),
            arch="anima",
            model_kwargs={"max_sequence_length": args.max_text_length},
            **self._shared_offload_config_kwargs(args),
        )

    def construct_model(self, args, config):
        from extensions_built_in.diffusion_models.anima import AnimaModel

        model = AnimaModel(
            device=args.device,
            model_config=config,
            dtype=args.dtype,
            noise_scheduler=AnimaModel.get_train_scheduler(),
        )
        model.skip_te = True
        return model

    def load_transformer(self, model, args):
        from diffusers import CosmosTransformer3DModel

        transformer = CosmosTransformer3DModel.from_pretrained(
            model.model_config.name_or_path,
            subfolder="transformer",
            torch_dtype=model.torch_dtype,
            local_files_only=not args.allow_download,
        )
        transformer.all_patch_size = [model.patch_size]
        model.model = transformer
        return transformer

    def load_conditioning(self, paths, batch_size):
        caches = load_condition_caches(paths)
        for cache in caches:
            if not isinstance(cache, AdvancedPromptEmbeds):
                raise SystemExit(
                    "anima requires AdvancedPromptEmbeds conditioning caches"
                )
        if len(caches) == 1:
            return caches[0].expand_to_batch(batch_size)
        if len(caches) != batch_size:
            raise SystemExit(
                f"got {len(caches)} anima caches for batch size {batch_size}; "
                "pass one cache or exactly batch-size caches"
            )
        return AdvancedPromptEmbeds.concat_prompt_embeds(caches)

    def make_latents(self, resolution, batch_size, generator, model) -> torch.Tensor:
        width, height = resolution
        channels = int(getattr(model.model.config, "in_channels", 16))
        return torch.randn(
            batch_size, channels, height // 8, width // 8, generator=generator
        )

    def default_fullmodule_target(self, transformer) -> str:
        return "transformer_blocks.0.attn1.to_q"


PROFILES = {
    "krea2": Krea2SmokeProfile(),
    "zimage": ZImageSmokeProfile(),
    "ideogram4": Ideogram4SmokeProfile(),
    "anima": AnimaSmokeProfile(),
}
