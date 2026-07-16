# Anima model integration

Anima is registered as a self-contained `BaseModel` extension with
`model.arch: anima`. Its implementation lives entirely in this folder; core
training code does not dispatch on Anima-specific flags.

## Components

- `CosmosTransformer3DModel` transformer
- `Qwen3Model` text encoder
- `AnimaTextConditioner` bridge fed by Qwen states and T5 token IDs
- `AutoencoderKLQwenImage` video-shaped VAE
- `CustomFlowMatchEulerDiscreteScheduler` for training and previews

## Shape and time conventions

The trainer and latent cache use single-frame 4D latents `(B, C, H, W)`.
Anima expands these to `(B, C, 1, H, W)` only around the Cosmos transformer
and VAE calls. VAE latents are normalized with the checkpoint's
`latents_mean` and `latents_std` values.

Ai-toolkit timesteps enter the model on the standard `0..1000` scale. The
transformer receives `timestep / num_train_timesteps`, matching Cosmos' `0..1`
convention. The flow-matching target remains `noise - clean`.

Cosmos receives a separate full-pixel padding mask. When
`concat_padding_mask` is enabled, diffusers adds the extra patch-embed channel
internally; callers must not append a seventeenth latent channel themselves.

The Cosmos patch projection consumes a 5D video tensor directly. It remains
dense during quantization because Quanto Linear kernels accept only 2D or 3D
activations; the repeated transformer blocks are still quantized normally.

## Text caches and TE worker

Prompt conditioning is stored as `AdvancedPromptEmbeds`, with one full-length
2D `(512, D)` conditioner output per prompt. The conditioner output after the
T5 attention-mask length is learned conditioning, not padding, so it must not
be trimmed. The cache space is `anima_te_v3`, so truncated or legacy batched
prompt caches are not reused.

`te_only` loads both Qwen3 and `AnimaTextConditioner` without the transformer or
VAE. `skip_te` loads fake placeholders for both text components so the trainer
can consume worker-created disk caches without co-resident text models.
`AnimaModel` advertises that generic partial-load contract through
`supports_te_cache_worker`; worker orchestration contains no Anima dispatch.

The pinned Transformers 5.5.3 tokenizer backend performs an unrelated online
Mistral metadata probe when `Qwen2Tokenizer` receives a Hub ID, even with
`local_files_only=True`. The Anima loader therefore uses a complete cached
snapshot path when available, while retaining the Hub-ID path as a fallback
for a first run whose tokenizer files are not cached yet.

## LoRA and sampling

LoRA targets `CosmosTransformer3DModel`. Save conversion writes the public
`diffusion_model.*` Circlestone/ComfyUI naming, and load conversion maps those
keys back to the internal diffusers PEFT names.

Preview sampling uses an embeds-only manual CFG loop. Conditional and
unconditional variable-length embeddings are padded to a shared length and run
together in one transformer call per scheduler step.
