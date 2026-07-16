import os
from typing import List, Optional

import numpy as np
import torch
import yaml
from diffusers import AnimaTextConditioner
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKLQwenImage, CosmosTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor
from optimum.quanto import QTensor
from PIL import Image
from torchvision.transforms import Resize
from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.memory_management import MemoryManager
from toolkit.models.base_model import BaseModel
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.quantize import quantize_model


scheduler_config = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.30.0.dev0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": False,
}


SAVE_RENAME = {
    "transformer_blocks.": "blocks.",
    "attn1.to_q": "self_attn.q_proj",
    "attn1.to_k": "self_attn.k_proj",
    "attn1.to_v": "self_attn.v_proj",
    "attn1.to_out.0": "self_attn.output_proj",
    "attn2.to_q": "cross_attn.q_proj",
    "attn2.to_k": "cross_attn.k_proj",
    "attn2.to_v": "cross_attn.v_proj",
    "attn2.to_out.0": "cross_attn.output_proj",
    "ff.net.0.proj": "mlp.layer1",
    "ff.net.2": "mlp.layer2",
    "norm1.linear_1": "adaln_modulation_self_attn.1",
    "norm1.linear_2": "adaln_modulation_self_attn.2",
    "norm2.linear_1": "adaln_modulation_cross_attn.1",
    "norm2.linear_2": "adaln_modulation_cross_attn.2",
    "norm3.linear_1": "adaln_modulation_mlp.1",
    "norm3.linear_2": "adaln_modulation_mlp.2",
    "norm_out.linear_1": "final_layer.adaln_modulation.1",
    "norm_out.linear_2": "final_layer.adaln_modulation.2",
    "proj_out": "final_layer.linear",
    "time_embed.t_embedder": "t_embedder.1",
    "time_embed.norm": "t_embedding_norm",
    "patch_embed.proj": "x_embedder.proj.1",
}
LOAD_RENAME = {value: key for key, value in SAVE_RENAME.items()}


def _pad_prompt_embeds(
    embeds: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    sequence_length: Optional[int] = None,
) -> torch.Tensor:
    if not embeds:
        raise ValueError("Anima prompt embeddings cannot be empty")
    max_length = sequence_length or max(item.shape[0] for item in embeds)
    feature_dim = embeds[0].shape[-1]
    padded = torch.zeros(
        len(embeds), max_length, feature_dim, device=device, dtype=dtype
    )
    for index, item in enumerate(embeds):
        length = min(item.shape[0], max_length)
        padded[index, :length] = item[:length].to(device=device, dtype=dtype)
    return padded


class AnimaPipeline:
    """Lightweight embeds-only sampler facade used by BaseModel previews."""

    def __init__(self, model: "AnimaModel"):
        self.model = model

    @property
    def vae_scale_factor(self):
        return self.model.vae_scale_factor

    def set_progress_bar_config(self, **kwargs):
        return None

    def to(self, *args, **kwargs):
        return self


class AnimaModel(BaseModel):
    arch = "anima"
    use_old_lokr_format = False
    text_embedding_space_version = "anima_te_v2"
    text_embed_dim = 1024

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.use_old_lokr_format = False
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["CosmosTransformer3DModel"]
        self.patch_size = 2
        self.cosmos_patch_size = (1, 2, 2)
        self.vae_scale_factor = 8
        self.max_sequence_length = int(
            self.model_config.model_kwargs.get("max_sequence_length", 512)
        )
        self.t5_tokenizer = None
        self.text_conditioner = None

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return self.vae_scale_factor * self.patch_size

    def load_model(self):
        if self.is_loaded:
            return

        dtype = self.torch_dtype
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.name_or_path_original
        if os.path.isdir(model_path) and os.path.isdir(
            os.path.join(model_path, "transformer")
        ):
            base_model_path = model_path

        self.print_and_status_update("Loading Anima model")
        transformer = None
        if self.te_only:
            self.print_and_status_update("Skipping Anima transformer (te_only load)")
        else:
            self.print_and_status_update("Loading Anima transformer")
            transformer = CosmosTransformer3DModel.from_pretrained(
                base_model_path,
                subfolder="transformer",
                torch_dtype=dtype,
            )
            transformer.all_patch_size = [self.patch_size]
            if self.model_config.quantize:
                self.print_and_status_update("Quantizing Anima transformer")
                quantize_model(self, transformer)
            else:
                transformer.to(self.device_torch, dtype=dtype)
            flush()

            if (
                self.model_config.layer_offloading
                and self.model_config.layer_offloading_transformer_percent > 0
            ):
                MemoryManager.attach(
                    transformer,
                    self.device_torch,
                    offload_percent=self.model_config.layer_offloading_transformer_percent,
                )
            elif self.model_config.low_vram:
                transformer.to("cpu")
            else:
                transformer.to(self.device_torch)
            transformer.requires_grad_(False)
            transformer.eval()

        vae = None
        if self.te_only:
            self.print_and_status_update("Skipping Anima VAE (te_only load)")
        else:
            self.print_and_status_update("Loading Anima VAE")
            vae = AutoencoderKLQwenImage.from_pretrained(
                base_model_path,
                subfolder="vae",
                torch_dtype=self.vae_torch_dtype,
            )
            vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)
            vae.requires_grad_(False)
            vae.eval()
            flush()

        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer"
        )
        t5_tokenizer = T5TokenizerFast.from_pretrained(
            base_model_path, subfolder="t5_tokenizer"
        )

        if self.skip_te:
            from toolkit.unloader import FakeTextEncoder

            self.print_and_status_update(
                "Skipping Anima text encoder + conditioner (skip_te load)"
            )
            text_encoder = FakeTextEncoder(device=self.device_torch, dtype=dtype)
            text_conditioner = FakeTextEncoder(device=self.device_torch, dtype=dtype)
        else:
            self.print_and_status_update("Loading Anima text encoder (Qwen3)")
            text_encoder = Qwen3Model.from_pretrained(
                base_model_path,
                subfolder="text_encoder",
                torch_dtype=dtype,
            )
            text_encoder.to(self.te_device_torch, dtype=dtype)
            text_encoder.requires_grad_(False)
            text_encoder.eval()
            flush()

            self.print_and_status_update("Loading Anima text conditioner")
            text_conditioner = AnimaTextConditioner.from_pretrained(
                base_model_path,
                subfolder="text_conditioner",
                torch_dtype=dtype,
            )
            text_conditioner.to(self.te_device_torch, dtype=dtype)
            text_conditioner.requires_grad_(False)
            text_conditioner.eval()
            flush()

        self.noise_scheduler = self.noise_scheduler or self.get_train_scheduler()
        self.model = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_conditioner = text_conditioner
        self.tokenizer = tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.pipeline = AnimaPipeline(self)
        self.is_loaded = True
        self.print_and_status_update("Anima model loaded")

    def text_encoder_to(self, *args, **kwargs):
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        if self.text_conditioner is not None:
            self.text_conditioner.to(*args, **kwargs)

    def set_device_state(self, state):
        super().set_device_state(state)
        if self.text_conditioner is None:
            return
        text_state = state["text_encoder"]
        if isinstance(text_state, list):
            text_state = text_state[0]
        self.text_conditioner.to(text_state["device"])
        self.text_conditioner.requires_grad_(False)
        self.text_conditioner.eval()

    def encode_prompt(self, prompt, *args, dropout_prob=0.0, **kwargs):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if dropout_prob > 0.0:
            prompts = [
                item if torch.rand(1).item() > dropout_prob else ""
                for item in prompts
            ]
        return self.get_prompt_embeds(prompts)

    @torch.no_grad()
    def get_prompt_embeds(self, prompt, control_images=None):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        qwen_inputs = self.tokenizer(
            prompts,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        qwen_ids = qwen_inputs.input_ids.to(self.te_device_torch)
        qwen_mask = qwen_inputs.attention_mask.to(self.te_device_torch)
        if qwen_ids.shape[-1] == 0:
            qwen_ids = qwen_ids.new_zeros((qwen_ids.shape[0], 1))
            qwen_mask = qwen_mask.new_zeros((qwen_mask.shape[0], 1))

        qwen_embeds = self.text_encoder(
            input_ids=qwen_ids,
            attention_mask=qwen_mask,
            output_hidden_states=False,
        ).last_hidden_state.to(
            device=self.te_device_torch, dtype=self.text_conditioner.dtype
        )
        qwen_embeds = qwen_embeds * qwen_mask.to(qwen_embeds).unsqueeze(-1)

        t5_inputs = self.t5_tokenizer(
            prompts,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        t5_ids = t5_inputs.input_ids.to(self.te_device_torch)
        t5_mask = t5_inputs.attention_mask.to(self.te_device_torch)
        conditioning = self.text_conditioner(
            source_hidden_states=qwen_embeds,
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=qwen_mask,
        ).to(dtype=self.torch_dtype, device=self.te_device_torch)

        per_item = []
        for index in range(conditioning.shape[0]):
            length = max(int(t5_mask[index].sum().item()), 1)
            per_item.append(conditioning[index, :length])
        return AdvancedPromptEmbeds(text_embeds=per_item)

    @torch.no_grad()
    def encode_images(self, image_list, device=None, dtype=None):
        device = device or self.vae_device_torch
        dtype = dtype or self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        images = [image.to(device=device, dtype=dtype) for image in image_list]
        for index, image in enumerate(images):
            height, width = image.shape[-2:]
            if height % self.vae_scale_factor or width % self.vae_scale_factor:
                images[index] = Resize(
                    (
                        height // self.vae_scale_factor * self.vae_scale_factor,
                        width // self.vae_scale_factor * self.vae_scale_factor,
                    )
                )(image)
        images = torch.stack(images).unsqueeze(2)
        raw_latents = self.vae.encode(images).latent_dist.sample()
        mean, std = self._latent_stats(raw_latents)
        return ((raw_latents - mean) / std).squeeze(2).to(device=device, dtype=dtype)

    @torch.no_grad()
    def decode_latents(self, latents, device=None, dtype=None):
        device = device or self.vae_device_torch
        dtype = dtype or self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        latents = latents.to(device=device, dtype=dtype)
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)
        elif latents.ndim != 5:
            raise ValueError("Anima latents must be 4D or 5D")
        mean, std = self._latent_stats(latents)
        images = self.vae.decode(latents * std + mean, return_dict=False)[0]
        if images.shape[2] == 1:
            images = images.squeeze(2)
        return images.to(device=device, dtype=dtype)

    def _latent_stats(self, latents):
        channels = latents.shape[1]
        mean = torch.as_tensor(self.vae.config.latents_mean).view(
            1, channels, 1, 1, 1
        ).to(latents)
        std = torch.as_tensor(self.vae.config.latents_std).view(
            1, channels, 1, 1, 1
        ).to(latents)
        return mean, std

    def get_noise_prediction(
        self, latent_model_input, timestep, text_embeddings, **kwargs
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        latents_5d = latent_model_input.to(
            self.device_torch, self.torch_dtype
        ).unsqueeze(2)
        timestep = timestep.float() / self.noise_scheduler.config.num_train_timesteps
        timestep = timestep.expand(latents_5d.shape[0]).to(
            self.device_torch, self.torch_dtype
        )
        encoder_hidden_states = _pad_prompt_embeds(
            text_embeddings.text_embeds, self.device_torch, self.torch_dtype
        )
        height, width = latents_5d.shape[-2:]
        padding_mask = latents_5d.new_zeros(
            1,
            1,
            height * self.vae_scale_factor,
            width * self.vae_scale_factor,
            dtype=self.torch_dtype,
        )
        with self.accelerator.autocast():
            prediction = self.model(
                hidden_states=latents_5d,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]
        if isinstance(prediction, QTensor):
            prediction = prediction.dequantize()
        return prediction.squeeze(2)

    def get_generation_pipeline(self):
        return AnimaPipeline(self)

    def generate_single_image(
        self,
        pipeline: AnimaPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ) -> Image.Image:
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        height = gen_config.height
        width = gen_config.width
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        channels = self.model.config.in_channels
        if gen_config.latents is None:
            latents = randn_tensor(
                (1, channels, 1, latent_height, latent_width),
                generator=generator,
                device=self.device_torch,
                dtype=self.torch_dtype,
            )
        else:
            latents = gen_config.latents.to(self.device_torch, self.torch_dtype)
            if latents.ndim == 4:
                latents = latents.unsqueeze(2)

        sequence_length = max(
            max(item.shape[0] for item in conditional_embeds.text_embeds),
            max(item.shape[0] for item in unconditional_embeds.text_embeds),
        )
        conditional = _pad_prompt_embeds(
            conditional_embeds.text_embeds,
            self.device_torch,
            self.torch_dtype,
            sequence_length,
        )
        unconditional = _pad_prompt_embeds(
            unconditional_embeds.text_embeds,
            self.device_torch,
            self.torch_dtype,
            sequence_length,
        )
        encoder_hidden_states = torch.cat([unconditional, conditional])
        padding_mask = latents.new_zeros(
            1, 1, height, width, dtype=self.torch_dtype
        )

        scheduler = self.get_train_scheduler()
        steps = gen_config.num_inference_steps
        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        scheduler.set_timesteps(sigmas=sigmas, device=self.device_torch)
        scheduler.set_begin_index(0)
        for timestep in scheduler.timesteps:
            model_timestep = timestep.expand(1).to(self.torch_dtype)
            model_timestep = model_timestep / scheduler.config.num_train_timesteps
            with torch.no_grad():
                prediction = self.model(
                    hidden_states=torch.cat([latents, latents]),
                    timestep=model_timestep.repeat(2),
                    encoder_hidden_states=encoder_hidden_states,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
            if isinstance(prediction, QTensor):
                prediction = prediction.dequantize()
            unconditional_prediction, conditional_prediction = prediction.chunk(2)
            prediction = unconditional_prediction + gen_config.guidance_scale * (
                conditional_prediction - unconditional_prediction
            )
            latents = scheduler.step(
                prediction, timestep, latents, return_dict=False
            )[0]

        decoded = self.decode_latents(latents)
        image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        return image_processor.postprocess(decoded, output_type="pil")[0]

    def get_loss_target(self, *args, **kwargs):
        return (kwargs["noise"] - kwargs["batch"].latents).detach()

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def get_transformer_block_names(self):
        return ["transformer_blocks"]

    def get_quantization_exclude_modules(self):
        # CosmosPatchEmbed applies this Linear directly to a 5D video tensor.
        # Quanto Linear kernels accept only 2D/3D activations, and this small
        # input projection is outside the repeated transformer blocks anyway.
        return ["patch_embed.proj"]

    def save_model(self, output_path, meta, save_dtype):
        transformer = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "aitk_meta.yaml"), "w") as handle:
            yaml.dump(meta, handle)

    def convert_lora_weights_before_save(self, state_dict):
        return self._convert_lora_weights(
            state_dict,
            prefix_from="transformer.",
            prefix_to="diffusion_model.",
            rename=SAVE_RENAME,
            lora_from=(".lora_A.", ".lora_B."),
            lora_to=(".lora_down.", ".lora_up."),
        )

    def convert_lora_weights_before_load(self, state_dict):
        return self._convert_lora_weights(
            state_dict,
            prefix_from="diffusion_model.",
            prefix_to="transformer.",
            rename=LOAD_RENAME,
            lora_from=(".lora_down.", ".lora_up."),
            lora_to=(".lora_A.", ".lora_B."),
        )

    @staticmethod
    def _convert_lora_weights(
        state_dict, prefix_from, prefix_to, rename, lora_from, lora_to
    ):
        converted = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith(prefix_from):
                new_key = prefix_to + new_key[len(prefix_from) :]
            for old, new in rename.items():
                new_key = new_key.replace(old, new)
            new_key = new_key.replace(lora_from[0], lora_to[0])
            new_key = new_key.replace(lora_from[1], lora_to[1])
            converted[new_key] = value
        return converted
