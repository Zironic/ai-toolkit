# Anima BaseModel Reintegration Plan

Ticket: `8e5e883` (`Re-integrate Anima correctly`)

## Goal

Make Anima a first-class `BaseModel` implementation under
`extensions_built_in/diffusion_models/anima/`, registered through
`AI_TOOLKIT_MODELS`, instead of a legacy `StableDiffusion` arch with scattered
`is_anima` branches.

The finished integration should keep the known working behavior:

- `CosmosTransformer3DModel` backbone.
- Qwen3 text encoder plus `AnimaTextConditioner`.
- T5 tokenizer inputs consumed by the conditioner.
- Cosmos/Qwen VAE with 5D encode/decode and latent mean/std normalization.
- Single-frame image training exposed to the trainer as 4D latents.
- Transformer forward using 5D latents: `(B, C, T, H, W)`.
- Temporal/spatial patching equivalent to `patch_size=[1, 2, 2]`.
- Manual preview sampling with batched CFG.
- FlowMatch timestep scaling from ai-toolkit's `0..1000` API to Cosmos `0..1`.
- Padding-mask behavior, including the `concat_padding_mask`/channel-17
  interaction if the loaded diffusers config requires it.
- Existing Anima LoRA save naming, plus a matching load conversion.
- TE-worker and `skip_te` compatibility.

## Current Defect

Anima is currently wired by special cases spread across core files:

- `toolkit/stable_diffusion_model.py`
  - Anima component loading.
  - Pipeline construction.
  - Prompt encoding dispatch.
  - Image latent encoding dispatch.
  - Training forward dispatch.
  - Manual sampling dispatch.
  - Full transformer save dispatch.
  - Anima LoRA key conversion.
  - Text-conditioner device moves.
- `toolkit/train_tools.py`
  - `encode_prompts_anima`.
- `toolkit/models/anima.py`
  - `encode_images_anima`.
- `toolkit/lora_special.py`
  - `is_anima` argument and `CosmosTransformer3DModel` target selection.
- `jobs/process/BaseSDTrainProcess.py`
  - `model_config.is_anima` scheduler selection.
  - `is_anima` forwarding into LoRA construction.
- `toolkit/config_modules.py`
  - `is_anima` compatibility flag and arch backfill.
- `toolkit/util/get_model.py`
  - `anima` is still listed as a legacy arch, so `get_model_class()` returns
    `StableDiffusion`.

The correct shape is the same as `krea2`, `ideogram4`, `z_image`, and the
documented `example_model` template: a self-contained model class that exposes
the standard `BaseModel` hooks and lets the trainer use generic model
capabilities.

## Design Target

Create:

- `extensions_built_in/diffusion_models/anima/__init__.py`
- `extensions_built_in/diffusion_models/anima/anima.py`
- Optional helper:
  `extensions_built_in/diffusion_models/anima/pipeline.py`

Register `AnimaModel` in `extensions_built_in/diffusion_models/__init__.py`.

`AnimaModel` should define:

- `arch = "anima"`
- `use_old_lokr_format = False`
- `is_flow_matching = True`
- `is_transformer = True`
- `target_lora_modules = ["CosmosTransformer3DModel"]`
- `patch_size = 2` for trainer/bucket math
- `cosmos_patch_size = [1, 2, 2]` for transformer semantics
- `vae_scale_factor = 8`
- `max_sequence_length` from `model.model_kwargs.max_sequence_length`, default
  `512`
- `text_embedding_space_version = "anima_te_v2"` to invalidate old
  `PromptEmbeds` Anima caches if the migration changes the cache container

Use `AdvancedPromptEmbeds` for new Anima text caches. Store one 2D tensor per
prompt and pad only at the model call. If the conditioner requires an attention
mask to preserve padded-token semantics, store that mask as an
`AdvancedPromptEmbeds` key and mark it in `frozen_dtype_keys`.

Keep trainer-facing latents 4D for compatibility with the existing image
training loop:

- `encode_images()` returns `(B, C, H, W)`.
- `get_noise_prediction()` expands to `(B, C, 1, H, W)` for the transformer.
- `decode_latents()` accepts 4D or 5D, denormalizes with VAE mean/std, decodes
  as 5D, and returns image tensors.

## Low-Effort Agent Handoff

This section is normative. If later exploratory language says "decide", "audit", or "if needed", do the concrete thing specified here first. Only deviate when a test proves the concrete thing is wrong, and document the proof in the ticket.

### Required Work Order

1. Add tests that describe the desired new behavior while the old code still exists.
2. Create and register the new model class while leaving the legacy Anima path intact.
3. Move prompt, latent, forward, sampling, LoRA, and save behavior into `AnimaModel`.
4. Make the new tests pass.
5. Remove the legacy Anima branches.
6. Run static cleanup checks after removal.

Do not delete legacy code before the new `AnimaModel` tests pass. Do not continue through shape, scheduler, or LoRA uncertainty by guessing.

### Exact Files

Create:

- `extensions_built_in/diffusion_models/anima/__init__.py`
- `extensions_built_in/diffusion_models/anima/anima.py`
- `tests/test_anima_model_registration.py`
- `tests/test_anima_shapes.py`
- `tests/test_anima_lora_keys.py`

Edit:

- `extensions_built_in/diffusion_models/__init__.py`
- `toolkit/util/get_model.py`
- `toolkit/stable_diffusion_model.py`
- `toolkit/train_tools.py`
- `toolkit/lora_special.py`
- `jobs/process/BaseSDTrainProcess.py`
- `toolkit/config_modules.py`

Only edit these if a real field/default must change:

- `config/examples/train_lora_anima_24gb.yaml`
- `ui/src/app/jobs/new/options.ts`

Delete only after `rg "encode_prompts_anima|is_anima|toolkit.models.anima"` shows no remaining production imports:

- `toolkit/models/anima.py`

### Required `AnimaModel` Skeleton

Use this shape. Fill the bodies by moving the current legacy logic, not by rewriting behavior from memory.

```python
scheduler_config = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.30.0.dev0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}


class AnimaModel(BaseModel):
    arch = "anima"
    use_old_lokr_format = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    @property
    def text_embedding_space_version(self):
        return "anima_te_v2"

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return self.vae_scale_factor * self.patch_size

    def load_model(self):
        ...

    def text_encoder_to(self, *args, **kwargs):
        ...

    def encode_prompt(self, prompt, *args, dropout_prob=0.0, **kwargs):
        ...

    def get_prompt_embeds(self, prompt):
        ...

    def encode_images(self, image_list, device=None, dtype=None):
        ...

    def decode_latents(self, latents, device=None, dtype=None):
        ...

    def get_noise_prediction(self, latent_model_input, timestep, text_embeddings, **kwargs):
        ...

    def get_generation_pipeline(self):
        return AnimaPipeline(self)

    def generate_single_image(self, pipeline, gen_config, conditional_embeds,
                              unconditional_embeds, generator, extra):
        ...

    def get_loss_target(self, *args, **kwargs):
        return (kwargs["noise"] - kwargs["batch"].latents).detach()

    def get_transformer_block_names(self):
        return ["transformer_blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        ...

    def convert_lora_weights_before_load(self, state_dict):
        ...
```

### Tensor Shape Contract

- Image input item: `(C, H, W)`.
- VAE encode input: `(B, C, 1, H, W)`.
- Cached/trainer latents: `(B, Z, h, w)`.
- Transformer input: `(B, Z, 1, h, w)`.
- Padding mask: `(1, 1, H, W)` in full pixel dimensions.
- Transformer output: `(B, Z, 1, h, w)`.
- Model prediction returned to trainer: `(B, Z, h, w)`.
- Per-item text embed in `AdvancedPromptEmbeds`: `(L, D)`.
- Batched text embed passed to transformer: `(B, Lmax, D)`.

Never store 3D per-item text tensors in `AdvancedPromptEmbeds`; BaseModel uses 2D per-item tensors to infer batch size.

### Prompt Encoding Contract

Do this exactly:

1. Accept `str` or `list[str]`.
2. Apply prompt dropout in `AnimaModel.encode_prompt()`, because `BaseModel.encode_prompt()` accepts `dropout_prob` but does not pass it to `get_prompt_embeds()`.
3. Tokenize Qwen with `padding="longest"`, `max_length=max_sequence_length`, `truncation=True`, `return_tensors="pt"`.
4. Run Qwen3 with `input_ids`, `attention_mask`, and `output_hidden_states=False`.
5. Convert Qwen hidden states to `text_conditioner.dtype`.
6. Multiply Qwen hidden states by `qwen_mask.unsqueeze(-1)`.
7. Tokenize T5 with the same prompt list and the same max length.
8. Run `AnimaTextConditioner(...)`.
9. For each batch item, trim conditioner output to `max(int(t5_mask[i].sum().item()), 1)`.
10. Return `AdvancedPromptEmbeds(text_embeds=[item0, item1, ...])`.

Do not store a text attention mask unless a failing test proves the transformer needs it. The conditioner consumes the masks; the transformer receives only the conditioned hidden states.

### Scheduler Contract

Use `CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)` with the exact config shown in the skeleton. This matches the legacy `get_sampler(..., arch="flux")` path.

Do not use the checkpoint `FlowMatchEulerDiscreteScheduler` as the training scheduler. It can be loaded into a thin generation helper if useful, but training must use `CustomFlowMatchEulerDiscreteScheduler` because `BaseSDTrainProcess` calls `set_train_timesteps(...)`.

### Padding-Mask Contract

Do not manually concatenate the padding mask to `hidden_states`.

The installed diffusers `CosmosTransformer3DModel` has `config.concat_padding_mask`, and when that is true its `forward()` resizes and concatenates `padding_mask` internally. Therefore:

- Pass latent hidden states with exactly `transformer.config.in_channels` channels.
- Pass `padding_mask=latents.new_zeros(1, 1, height, width, dtype=model_dtype)`.
- Keep `height = latent_h * vae_scale_factor`.
- Keep `width = latent_w * vae_scale_factor`.
- Add a fake-transformer test proving the model receives hidden states without the extra mask channel.
- Add a config/assertion test proving that when `concat_padding_mask == True`, the expected patch embed input channels are `in_channels + 1`.

The "channel 17" concern is internal to Cosmos patch embedding when `concat_padding_mask` is true. The ai-toolkit integration should pass a separate mask argument, not pre-concatenate channel 17 itself.

### LoRA Key Conversion Contract

Use this mapping from the current legacy implementation:

```python
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
LOAD_RENAME = {v: k for k, v in SAVE_RENAME.items()}
```

Save conversion order:

1. Prefix: `transformer.` to `diffusion_model.`
2. Apply `SAVE_RENAME`.
3. `.lora_A.` to `.lora_down.`
4. `.lora_B.` to `.lora_up.`

Load conversion order:

1. Prefix: `diffusion_model.` to `transformer.`
2. Apply `LOAD_RENAME`.
3. `.lora_down.` to `.lora_A.`
4. `.lora_up.` to `.lora_B.`

Required test examples:

- Save: `transformer.transformer_blocks.0.attn1.to_q.lora_A.weight` becomes `diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight`.
- Save: `transformer.patch_embed.proj.lora_B.weight` becomes `diffusion_model.x_embedder.proj.1.lora_up.weight`.
- Load maps both examples back to the original internal keys.

### Required Tests

Add focused tests before cleanup:

1. `tests/test_anima_model_registration.py`
   - `get_model_class(ModelConfig(arch="anima")).__name__ == "AnimaModel"`.
   - `"anima" not in LEGACY_MODEL_ARCHES` after cleanup.
2. `tests/test_anima_shapes.py`
   - Fake VAE encode returns 5D raw latents; `encode_images()` returns 4D.
   - `decode_latents()` accepts 4D and 5D.
   - Fake transformer receives 5D hidden states and separate 4D padding mask.
   - `get_noise_prediction()` returns 4D.
   - `AdvancedPromptEmbeds` contains only 2D per-item text tensors.
3. `tests/test_anima_lora_keys.py`
   - Save examples map exactly as specified above.
   - Load examples round-trip exactly.
4. Static cleanup test or `rg` check:
   - No `is_anima` remains in `toolkit/stable_diffusion_model.py`, `toolkit/lora_special.py`, or `jobs/process/BaseSDTrainProcess.py`.
   - No `encode_prompts_anima` remains in `toolkit/train_tools.py`.

## Step-by-Step Plan

### 1. Baseline the Existing Behavior

1. Read and annotate the current Anima branches in:
   - `toolkit/stable_diffusion_model.py`
   - `toolkit/train_tools.py`
   - `toolkit/models/anima.py`
   - `toolkit/lora_special.py`
   - `jobs/process/BaseSDTrainProcess.py`
   - `toolkit/config_modules.py`
   - `toolkit/util/get_model.py`
2. Record the currently loaded component names and subfolders:
   - `transformer/` -> `CosmosTransformer3DModel`
   - `vae/` -> `AutoencoderKLQwenImage`
   - `text_encoder/` -> `Qwen3Model`
   - `text_conditioner/` -> `AnimaTextConditioner`
   - `tokenizer/` -> `Qwen2Tokenizer`
   - `t5_tokenizer/` -> `T5TokenizerFast`
   - `scheduler/` -> `FlowMatchEulerDiscreteScheduler`
3. Before changing code, add or run a tiny discovery probe that confirms the
   current baseline:
   - `ModelConfig(arch="anima")`
   - `get_model_class(config) is StableDiffusion`
4. Keep this baseline probe so it can be flipped after registration to assert:
   - `get_model_class(config) is AnimaModel`

### 2. Add the Anima Extension Skeleton

1. Create `extensions_built_in/diffusion_models/anima/__init__.py`.
2. Create `extensions_built_in/diffusion_models/anima/anima.py`.
3. Define `AnimaModel(BaseModel)`.
4. Import and append `AnimaModel` in
   `extensions_built_in/diffusion_models/__init__.py`.
5. Add an import/discovery test that proves `arch: anima` resolves to
   `AnimaModel`.
6. Do not remove the legacy code yet. First make the new class loadable in
   isolation.

### 3. Move Component Loading Into `AnimaModel.load_model()`

1. Port the current `StableDiffusion` Anima load branch into
   `AnimaModel.load_model()`.
2. Preserve local checkpoint resolution:
   - If `name_or_path` is a local directory containing `transformer/`, treat
     that directory as the full base checkpoint.
   - Otherwise load subfolders from the configured repository/path.
3. Implement transformer loading:
   - Skip transformer when `self.te_only` is true.
   - Load `CosmosTransformer3DModel.from_pretrained(...)`.
   - Move to `self.quantize_device` before quantization when needed.
   - Respect `model.quantize`, `qtype`, and `quantize_kwargs`.
   - Use the current repo quantization pattern for BaseModel extensions
     (`quantize_model(self, transformer)` where possible); preserve
     dequantization-on-save behavior if direct `quantize()` remains required.
4. Implement VAE loading:
   - Skip VAE when `self.te_only` is true.
   - Load `AutoencoderKLQwenImage`.
   - Move to `self.vae_device_torch` with `self.vae_torch_dtype`.
   - Set eval and `requires_grad_(False)`.
5. Implement tokenizer loading:
   - Always load Qwen tokenizer and T5 tokenizer because they are small and keep
     the object well formed.
6. Implement text stack loading:
   - If `self.skip_te` is true, load no Qwen3 model and no
     `AnimaTextConditioner`; use `FakeTextEncoder` placeholders for both.
   - If `self.skip_te` is false, load `Qwen3Model` and
     `AnimaTextConditioner`, move both to `self.device_torch`, set eval, and
     freeze.
7. Store the standard BaseModel fields:
   - `self.model = transformer`
   - `self.vae = vae`
   - `self.text_encoder = text_encoder`
   - `self.text_conditioner = text_conditioner`
   - `self.tokenizer = tokenizer`
   - `self.t5_tokenizer = t5_tokenizer`
   - `self.noise_scheduler = AnimaModel.get_train_scheduler()` unless one was
     injected
   - `self.pipeline = AnimaPipeline(self)` or a thin Anima helper object
8. Implement `text_encoder_to()` override so both Qwen3 and the conditioner move
   together.

### 4. Define Scheduler Behavior

1. Add `AnimaModel.get_train_scheduler()`.
2. Return the same scheduler family the current integration expects:
   `CustomFlowMatchEulerDiscreteScheduler` if training needs
   `set_train_timesteps()`, or a checkpoint-derived FlowMatch scheduler only if
   it supports the trainer's calls.
3. Keep timestep conversion inside `get_noise_prediction()`:
   - Input timestep is ai-toolkit `0..1000`.
   - Cosmos timestep is `timestep / num_train_timesteps`.
4. Remove the Anima-specific scheduler branch in `BaseSDTrainProcess` only after
   `get_train_scheduler()` is registered and covered by a test.

### 5. Move Prompt Encoding

1. Move `encode_prompts_anima()` out of `toolkit/train_tools.py` and into the
   Anima extension, either as a private helper or as `AnimaModel` methods.
2. Implement `get_prompt_embeds(prompt)`:
   - Accept `str` or `list[str]`.
   - Tokenize with Qwen tokenizer using `padding="longest"`.
   - Run Qwen3 to get hidden states.
   - Mask Qwen hidden states with the Qwen attention mask.
   - Tokenize the same prompt list with T5 tokenizer.
   - Run `AnimaTextConditioner` with Qwen hidden states, Qwen mask, T5 IDs, and
     T5 mask.
   - Return `AdvancedPromptEmbeds(text_embeds=[...])`.
3. Decide how to store variable-length conditioning:
   - Preferred: trim each conditioner output to the real T5 length and store one
     2D tensor per prompt.
   - Conservative fallback: keep padded 2D tensors but still store them per
     prompt in `AdvancedPromptEmbeds`.
4. Add a small helper to batch prompt embeds for the transformer call:
   - Pad conditional/unconditional batches to a common sequence length.
   - Preserve dtype/device.
   - If text masks are stored, pad masks with zeros and keep integer/bool dtype.
5. Preserve prompt dropout intentionally:
   - `BaseModel.encode_prompt()` currently accepts `dropout_prob` but does not
     pass it to `get_prompt_embeds()`.
   - If Anima must keep prompt-level dropout from the old helper, override
     `encode_prompt()` in `AnimaModel` and apply dropout before
     `get_prompt_embeds()`.
   - Otherwise rely on dataset caption dropout and document the behavior.

### 6. Move Image Encoding and Decoding

1. Move `encode_images_anima()` from `toolkit/models/anima.py` into
   `AnimaModel.encode_images()`.
2. Preserve the current encode path:
   - Input list items are `(C, H, W)` in `[-1, 1]`.
   - Resize to multiples of `vae_scale_factor=8`.
   - Stack to `(B, C, H, W)`.
   - Unsqueeze time to `(B, C, 1, H, W)`.
   - Run VAE encode.
   - Normalize with `vae.config.latents_mean` and `vae.config.latents_std`.
   - Squeeze time and return 4D latents.
3. Implement `decode_latents()`:
   - Accept 4D latents and unsqueeze time.
   - Accept 5D latents as-is.
   - Denormalize with mean/std.
   - Decode through the VAE.
   - Squeeze the single-frame time dimension before returning images.
4. Implement `get_bucket_divisibility()` returning `16`
   (`vae_scale_factor=8 * spatial_patch_size=2`).
5. Decide whether to bump `latent_space_version`:
   - Do not bump if the cached latent tensor values remain identical.
   - Bump only if normalization, shape, or resize behavior changes.

### 7. Move the Training Forward

1. Implement `get_noise_prediction(...)` on `AnimaModel`.
2. Keep the trainer-facing input as 4D latents.
3. Expand to 5D before transformer call:
   - `(B, C, H, W)` -> `(B, C, 1, H, W)`.
4. Convert timestep:
   - `t = timestep.float() / self.noise_scheduler.config.num_train_timesteps`
   - Expand to batch size and cast to model dtype/device.
5. Build the image padding mask:
   - Shape should match the official Anima/Cosmos expectation.
   - Start from full pixel dimensions: `H_latent * 8`, `W_latent * 8`.
   - Use batch size `B` unless diffusers explicitly accepts broadcasting from
     batch size `1`.
6. Audit the loaded transformer config for `concat_padding_mask`:
   - If diffusers concatenates the mask internally, pass hidden states with the
     latent channel count only and pass `padding_mask` separately.
   - If the local model expects an already-concatenated mask channel, concatenate
     it explicitly and verify the input channel count is 17.
   - Add a shape test so this behavior cannot silently regress.
7. Call `CosmosTransformer3DModel` with:
   - `hidden_states`
   - `timestep`
   - `encoder_hidden_states`
   - `padding_mask`
   - `return_dict=False`
8. Dequantize `QTensor` outputs if needed.
9. Squeeze the time dimension back to 4D for loss computation.
10. Keep `get_loss_target()` on the BaseModel flow-matching convention unless a
    validation run proves Anima needs a sign flip:
    - Expected ai-toolkit target: `noise - clean`.

### 8. Move Preview Sampling

1. Move the current manual sampling loop out of `StableDiffusion.generate_images`
   into `AnimaModel.generate_single_image()` or an `AnimaPipeline` helper.
2. Keep sampling embeds-only:
   - `generate_single_image()` receives already encoded positive and negative
     embeds from `BaseModel.generate_images()`.
   - It must not re-tokenize prompt text.
3. Initialize latents as 5D:
   - `(1, latent_channels, 1, H/8, W/8)`.
   - Use `gen_config.latents` when supplied; otherwise use seeded `randn_tensor`.
4. Initialize scheduler timesteps the same way as the old code first:
   - `sigmas = linspace(1.0, 1.0 / n_steps, n_steps)`.
   - Then validate against the official diffusers Anima pipeline.
5. Batch CFG manually:
   - Concatenate unconditional and conditional latents.
   - Concatenate unconditional and conditional text embeds after padding to the
     same length.
   - Run one transformer call.
   - Compute `uncond + guidance_scale * (cond - uncond)`.
6. Step the scheduler.
7. Denormalize and decode with the Anima VAE.
8. Postprocess with `VaeImageProcessor`.
9. Restore train state and devices through the existing BaseModel generation
   harness.

### 9. Move LoRA Behavior

1. Rely on `self.is_transformer = True` and `target_lora_modules` instead of
   `is_anima` in `LoRASpecialNetwork`.
2. Remove `is_anima` from `LoRASpecialNetwork.__init__` once no caller uses it.
3. Set `AnimaModel.target_lora_modules = ["CosmosTransformer3DModel"]`.
4. Implement `get_transformer_block_names()` after inspecting loaded Anima module
   names:
   - Expected candidate: `["transformer_blocks"]`.
   - Confirm by checking `named_modules()` on a loaded transformer.
5. Move `convert_lora_weights_before_save()` into `AnimaModel`.
6. Add `convert_lora_weights_before_load()` as the inverse mapping so saved
   Anima LoRAs can resume cleanly.
7. Preserve the public save convention:
   - Internal PEFT prefix: `transformer.`
   - Public Anima/Circlestone prefix: `diffusion_model.`
   - Diffusers block names mapped to native names.
   - PEFT `.lora_A.`/`.lora_B.` mapped to `.lora_down.`/`.lora_up.` on save.
8. Add unit tests for save and load key conversion.

### 10. Move Full-Model Save Behavior

1. Implement `save_model()` on `AnimaModel`.
2. Preserve current full-save behavior:
   - Save the transformer under `output_path/transformer`.
   - Use safe serialization.
3. Decide whether a full fine-tune should also save scheduler, VAE/tokenizer, or
   text components:
   - Keep current transformer-only behavior for the first migration.
   - Document that full checkpoint completeness is a later enhancement unless a
     current workflow depends on it.
4. Ensure quantized transformer save still dequantizes or writes valid weights.

### 11. Remove Legacy Core Branches

After the new model class passes focused tests, remove legacy Anima branches:

1. `toolkit/stable_diffusion_model.py`
   - Remove Anima imports from diffusers/transformers.
   - Remove `anima_t5_tokenizer` and `anima_text_conditioner` fields.
   - Remove `is_anima` property if it only exists for the legacy path.
   - Remove Anima load branch.
   - Remove Anima generation branch.
   - Remove Anima prompt-encode branch.
   - Remove Anima image-encode branch.
   - Remove Anima training-forward branch.
   - Remove Anima save branch.
   - Remove Anima LoRA conversion branch.
   - Remove Anima-specific `text_encoder_to()` behavior.
2. `toolkit/train_tools.py`
   - Remove `encode_prompts_anima`.
3. `toolkit/models/anima.py`
   - Delete the file if no imports remain.
4. `toolkit/lora_special.py`
   - Remove `is_anima` constructor parameter and instance field.
   - Remove Anima-specific target module branch.
   - Keep generic transformer PEFT behavior through `is_transformer`.
5. `jobs/process/BaseSDTrainProcess.py`
   - Remove `model_config.is_anima` scheduler fallback.
   - Remove `is_anima=` from LoRA constructor arguments.
   - Rely on `ModelClass.get_train_scheduler()`, `self.sd.is_transformer`, and
     `self.sd.target_lora_modules`.
6. `toolkit/config_modules.py`
   - Remove `is_anima` flag and arch backfill if no compatibility requirement
     remains.
   - Keep `"anima"` in `ModelArch`.
7. `toolkit/util/get_model.py`
   - Remove `"anima"` from `LEGACY_MODEL_ARCHES`.
8. `ui/src/app/jobs/new/options.ts`
   - Keep `arch: anima`.
   - Update copy/defaults only if the new model needs different config fields.
9. `config/examples/train_lora_anima_24gb.yaml`
   - Keep `arch: anima`.
   - Add cache/TE-worker fields only if validation shows they are required.

### 12. TE-Worker and `skip_te` Validation

1. Confirm `run_te_cache_worker()` resolves `AnimaModel`, not `StableDiffusion`.
2. Confirm `te_only=True` loads:
   - Qwen tokenizer.
   - T5 tokenizer.
   - Qwen3 text encoder.
   - `AnimaTextConditioner`.
   - No transformer.
   - No VAE.
3. Confirm trainer load with `skip_te=True` loads:
   - Transformer.
   - VAE.
   - Tokenizers.
   - Fake Qwen/text-conditioner placeholders only.
   - No real Qwen3 text encoder.
   - No real `AnimaTextConditioner`.
4. Confirm `AdvancedPromptEmbeds` Anima caches round-trip through:
   - `PromptEmbeds.load(...)`
   - Dataset text-cache loading.
   - DOP cache loading if DOP is enabled.
   - Sample prompt cache loading.
5. Fix the pre-worker cache-signature path if needed:
   - `SDTrainer._prepare_file_item_text_cache_signature()` currently uses
     `str(model_config.arch)`.
   - If Anima uses `text_embedding_space_version = "anima_te_v2"`, make the
     pre-worker signature derive the model class text-cache version rather than
     hard-coding the arch string.
6. Treat end-to-end TE-worker validation as shared acceptance with ticket
   `98ffdf9`.

### 13. Tests and Smoke Checks

Focused non-GPU tests:

1. Model discovery:
   - `get_model_class(ModelConfig(arch="anima")).__name__ == "AnimaModel"`.
2. Legacy fallback removal:
   - `"anima" not in LEGACY_MODEL_ARCHES`.
3. Static cleanup:
   - No `is_anima` branches remain in the legacy core files.
4. Prompt embed batching helper:
   - Pads variable-length 2D tensors.
   - Keeps dtype/device.
   - Preserves masks if masks are stored.
5. Latent helpers with a fake VAE:
   - Encode path returns 4D latents.
   - Decode path accepts 4D and 5D latents.
   - Mean/std normalization and inverse are consistent.
6. Training forward shape test with a fake transformer:
   - Input 4D latents.
   - Transformer receives 5D latents.
   - Padding mask has expected pixel dimensions.
   - Output returns 4D.
7. LoRA key mapping:
   - Save conversion maps internal keys to public Anima keys.
   - Load conversion maps public keys back to internal keys.
8. Compile/import checks:
   - New Anima extension.
   - Touched core files.

Suggested commands:

```powershell
venv\Scripts\python.exe -m pytest tests\test_anima_model_registration.py -q
venv\Scripts\python.exe -m pytest tests\test_anima_prompt_embeds.py -q
venv\Scripts\python.exe -m pytest tests\test_anima_lora_keys.py -q
venv\Scripts\python.exe -m py_compile extensions_built_in\diffusion_models\anima\anima.py
venv\Scripts\python.exe -m py_compile toolkit\stable_diffusion_model.py toolkit\lora_special.py jobs\process\BaseSDTrainProcess.py
```

GPU/manual checks:

1. Load Anima without TE worker and encode one prompt.
2. Encode one image to latents and decode it back.
3. Run one `get_noise_prediction()` call on synthetic latents.
4. Generate one sample preview with positive/negative embeds and guidance.
5. Run a tiny LoRA smoke:
   - 1 image or tiny dataset.
   - 1 to 5 training steps.
   - sampling disabled first.
   - verify finite loss.
6. Run the same tiny smoke with TE worker enabled:
   - Worker builds text caches.
   - Trainer starts with `skip_te=True`.
   - No real Qwen3/text-conditioner load in the trainer process.
7. Save a LoRA and resume from it.
8. Test `model.quantize: true` for the transformer.
9. Test `model.quantize_te: true` only if the Qwen3/conditioner path supports
   it cleanly.

## Acceptance Criteria

The ticket can be closed when:

1. `arch: anima` resolves to `AnimaModel`, not `StableDiffusion`.
2. Anima is registered through `AI_TOOLKIT_MODELS`.
3. `anima` is removed from `LEGACY_MODEL_ARCHES`.
4. Legacy Anima branches are gone from:
   - `toolkit/stable_diffusion_model.py`
   - `toolkit/train_tools.py`
   - `toolkit/models/anima.py`
   - `toolkit/lora_special.py`
   - `jobs/process/BaseSDTrainProcess.py`
   - `toolkit/config_modules.py`
5. The example Anima config still uses `model.arch: anima`.
6. Prompt caches are valid under the new text embedding space version.
7. Latent caches are either bit-compatible with the old Anima encode path or
   invalidated with a new latent space version.
8. Training forward returns the expected 4D prediction and uses the correct
   Cosmos 5D input internally.
9. Manual preview sampling produces an image with CFG.
10. Anima LoRAs save in the expected public key format and can be loaded back.
11. A tiny LoRA run reaches at least one optimizer step with finite loss.
12. TE-worker validation passes or the remaining work is explicitly handed to
    ticket `98ffdf9` with a concrete blocker.

