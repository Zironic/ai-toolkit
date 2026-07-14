---
name: model-integration
description: How a new diffusion model architecture is meant to be wired into ai-toolkit - the BaseModel extension pattern, the legacy anti-pattern that made the Anima integration wrong, and this fork's extra validation seams. Load BEFORE integrating or reworking any model arch (new backbone, new text encoder combo, new pipeline) or when editing extensions_built_in/diffusion_models/.
---

# Model integration: the one right way

## The rule (learned the hard way with Anima)

A model integration is a **self-contained BaseModel subclass** under
`extensions_built_in/diffusion_models/<name>/`, auto-discovered via
`AI_TOOLKIT_MODELS`. It is **never** `is_<arch>` branches edited into the
monolithic core files.

Anima violated this: scattered `is_anima` / `'anima'` conditionals went
directly into `toolkit/stable_diffusion_model.py` (load/encode/forward/
decode/sample/save dispatch), `toolkit/config_modules.py`,
`toolkit/lora_special.py`, and `toolkit/train_tools.py`. It trained and
converged -- and was still WRONG: unmaintainable, invisible to the extension
machinery, and now needs a full redo (git-bug `8e5e883`, plan in
`tasks/open/ANIMA_BASE_MODEL_REINTEGRATION_PLAN.md`). "It runs" does not
mean "it is integrated correctly."

If you find yourself adding an arch string check to any core `toolkit/`
file, stop: that logic belongs in the model class, or (rarely) in a new
overridable hook on `BaseModel` that defaults to current behavior.

## The authoritative how-to

`extensions_built_in/diffusion_models/example_model/README.md` is the full,
maintained contract: registration flow, lifecycle (load -> cache -> train
step -> sample -> save), tensor/timestep/loss conventions,
AdvancedPromptEmbeds rules (per-item 2D `(L, D)` -- violating this breaks
batch-size inference), gradient checkpointing, quanto's 2D/3D activation
limit, and the optional-attention-backend pattern. **Read it before writing
any code**; copy from the closest worked example (`z_image`, `ideogram4`,
`krea2`, `wan22`, ...). Sharpest edges, as reminders only:

- Timesteps cross the BaseModel API on `0..1000` (1000 = pure noise);
  convert to the model's native convention inside `get_noise_prediction`,
  including reversed-time models.
- Flow-matching loss target here is `noise - clean` (velocity).
- Pipelines receive embeds, never raw text.
- Bump `text_embedding_space_version` when `get_prompt_embeds` output
  changes, or users' disk caches go stale silently.
- `use_old_lokr_format = False` on every new model class.
- No top-level `flash_attn`-style imports; default to SDPA, make other
  kernels opt-in flags.

## Fork-specific seams the README does not cover

- **TE-worker / skip_te handoff**: this fork runs a throwaway text-encoder
  worker that caches all prompt embeds to disk and dies; the trainer then
  loads with skip_te so the TE never co-resides with the transformer. A new
  model (especially multi-TE ones like Anima's Qwen3 + AnimaTextConditioner)
  is not done until validated against this path (see ticket `98ffdf9` for
  the Anima case).
- **Offload/memory-manager attach**: model integrations opt into streaming
  via `attach_smart_training` (`toolkit/memory_management/`). Offload flags
  live on `ModelConfig` (`layer_offloading_*`). Any offload-aware change
  must be off-by-default and behavior-preserving with flags off
  (`docs/decisions/UPSTREAM_PR_PLAN.md`).
- **UI exposure**: add the arch to `ui/src/app/jobs/new/options.ts` or it
  cannot be selected in the web UI.
- **Config over env**: model behavior is controlled by job config only; env
  vars do not exist in UI-launched runs.

## Definition of done for an integration

1. Self-contained extension folder; zero new arch conditionals in core files.
2. Registered in `AI_TOOLKIT_MODELS` and the UI options.
3. Trains a short LoRA run AND samples correctly (both directions of the
   timestep/latent conventions -- training can converge while sampling is
   subtly wrong, and vice versa).
4. Validated with quantization on (quanto activation-ndim trap) and with
   the TE-worker handoff.
5. Model-specific gotchas written down: in the model folder's README and,
   if they were expensive to learn, a git-bug ticket comment.
