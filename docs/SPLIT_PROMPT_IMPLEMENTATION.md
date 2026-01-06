# SplitPrompt Implementation Design

Goal: implement reliable per-block SplitPrompt routing so that when a dataset has `split_prompt_enabled`:
- Blocks 20–29 use the dataset sample's *normal prompt* (per-sample conditional prompt).
- Blocks 30–31 receive *no prompt* (empty/unconditional prompt embedding).
- Blocks 32–57 use the dataset **split_prompt** embedding (cached per-dataset).

Requirements:
- Must **apply** the mapping during training and inference.
- Must **fail fast** with a clear RuntimeError when routing cannot be applied (missing embeddings, shape mismatch, etc.).
- Add tests that validate correct routing and failure modes.

Two implementation approaches are described below. Option A is recommended (precise & testable). Option B provides a less-invasive, modular alternative that localizes changes to attention modules via a small hook.

---

## Option A — Full per-block routing in model pipeline (Recommended) ✅

Overview
- Implement an explicit per-block prompt routing path that constructs a "per-block PromptEmbeds" set for the batch and passes it through the `predict_noise` → UNet forward call.
- UNet cross-attention layers will consult per-block embeddings (if present) and use them instead of the global `text_embeddings` for that block.

Advantages
- Exact, deterministic per-block control; easy to test; matches intended semantics.
- Fine-grained checks and clear failure modes.

Files to modify
- extensions_built_in/sd_trainer/SDTrainer.py
  - After `conditional_embeds` and after `dataset_split_prompt_embeds` is available, build `per_block_prompt_embeds` (see snippet below).
  - Add runtime checks: existence, batch-size alignment, token length compatibility where applicable. Raise `RuntimeError` on any mismatch.
  - Pass `per_block_prompt_embeds` into the existing call to `sd.predict_noise(...)` (add a new arg).
- toolkit/stable_diffusion_model.py (and/or toolkit/pipelines.py)
  - Modify `predict_noise(...)` signature to accept an optional `per_block_prompt_embeds: Optional[Dict[int, PromptEmbeds]]` (or List indexed by block index).
  - Validate shapes vs latents (batch size) and prepare them for UNet forwarding.
  - Forward the `per_block_prompt_embeds` down to UNet call site.
- UNet implementation (where cross-attention runs)
  - Modify the UNet forward to accept `per_block_prompt_embeds` and in each transformer block, if an override exists for that block, use it.
  - Ensure handling of CFG (concatenated unconditional+conditional embeddings): when using per-block embeddings ensure the unconditional/conditional duplication is respected (i.e., double when doing CFG or provider must return already-concatenated embeddings).
- toolkit/prompt_utils.py
  - Utility helpers to build per-sample-per-block prompts, expand to batch, and verify compatibility.

High-level pseudocode in SDTrainer

```py
# after computing `conditional_embeds` (per-sample) and having `split_pe` (per-dataset PromptEmbeds):
batch_size = noisy_latents.shape[0]
per_block = {}

# Helper to expand an embed to batch
def expand_to_batch(pe, n):
    return pe.expand_to_batch(n)

# Blocks 20-29 -> sample conditional prompts
for b in range(20, 30):
    per_block[b] = [conditional_embeds[i] for i in range(batch_size)]

# Blocks 30-31 -> empty prompt
empty_pe = self.cached_blank_embeds
for b in (30, 31):
    per_block[b] = [expand_to_batch(empty_pe, 1) for _ in range(batch_size)]

# Blocks 32-57 -> split prompt (dataset-level): expand to batch size
sp = self.dataset_split_prompt_embeds[ds_key]
if sp is None: raise RuntimeError("Missing split prompt for dataset")
for b in range(32, 58):
    per_block[b] = expand_to_batch(sp, batch_size)

# sanity checks: shape, token length, etc.
# pass into predict_noise
pred = self.sd.predict_noise(latents, ..., per_block_prompt_embeds=per_block)
```

UNet cross-attention snippet (conceptual)

```py
# inside uni transformer block with index block_idx
override = per_block_prompt_embeds.get(block_idx)
if override is not None:
    # use override (already expanded to batch)
    text_embeds_for_block = override
else:
    text_embeds_for_block = text_embeddings
# compute cross-attention with text_embeds_for_block
```

Runtime checks (fail-fast)
- When detecting `_last_batch_has_splitprompt` is True (and `batch_splitprompt_key` is set), create per-block map and verify:
  - `self.dataset_split_prompt_embeds[dataset_key]` exists.
  - The split prompt `PromptEmbeds` seq length is compatible with model/tokenizer expectations (optionally assert token length <= model max or matches conditional_embeds token length when required).
  - After expanding to batch and possible CFG duplication, the shape batch dimension matches latents.
  - If any check fails, raise `RuntimeError` with a clear description.

Tests to add
- Unit: `testing/test_splitprompt_routing.py`
  - Test that when splitprompt is present, UNet receives per-block embeddings for the proper block indices. Use monkeypatches to intercept calls.
  - Test that blocks 30–31 receive the blank embeddings.
  - Test failure modes: missing split prompt raises RuntimeError; token-length mismatch raises.
- Integration: a short smoke training/testing flow that sets split_prompt_enabled and ensures no silent swallowing of errors and graph shows split prompt metadata.

Notes and Caveats
- Changes are invasive: UNet must be modified at attention call sites (careful to keep optimized code paths and device/dtype correctness).
- Performance: small overhead for per-block embedding selection. Ensure attention implementations keep memory locality and avoid copies where possible.

---

## Option B — PromptProvider hook in attention modules (Less invasive, modular) ⚠️

Overview
- Add a small interface/hook that allows the UNet attention block to query a provider for a prompt embedding override for the current block and sample.
- Provider is passed into `predict_noise` from trainer when batch has splitprompt.

Advantages
- Localizes changes to attention modules and a `PromptProvider` class. Simpler than full pipeline modifications.
- More modular: other features can supply alternate providers (e.g., adapters) without changing signature widely.

Files/Changes
- toolkit/prompt_utils.py
  - Add `PromptProvider` protocol / abstract class:

```py
class PromptProvider:
    def get_prompt(self, block_idx: int, sample_idx: int, is_unconditional: bool=False) -> Optional[PromptEmbeds]:
        """Return a PromptEmbeds override or None"""
```

- toolkit/stable_diffusion_model.py
  - Accept an optional `prompt_provider: Optional[PromptProvider]` arg in `predict_noise`.
  - Forward `prompt_provider` to UNet.
- UNet cross-attention
  - At the moment of cross-attention, call `prompt_provider.get_prompt(block_idx, sample_idx, is_unconditional)`.
    - If provider returns a PromptEmbeds for that sample/block, use it. If None, fall back to global embeddings.
  - MUST raise RuntimeError if `prompt_provider` indicates splitprompt but cannot provide embeddings for expected blocks (i.e., when `_last_batch_has_splitprompt` is True but provider returns None for block 32 etc.).
- extensions_built_in/sd_trainer/SDTrainer.py
  - Implement a `DatasetSplitPromptProvider` that wraps `self.dataset_split_prompt_embeds[ds_key]` and `self.cached_blank_embeds` and `conditional_embeds` and returns appropriate PromptEmbeds for each (block, sample).
  - When batch_has_splitprompt true, instantiate the provider and pass to `sd.predict_noise(..., prompt_provider=provider)`.

Runtime checks
- Provider must return PromptEmbeds for blocks expected to be active (32–57). If it returns None, raise `RuntimeError`.
- Check shapes; if mismatch, raise `RuntimeError`.

Tests
- Unit: implement tests for `DatasetSplitPromptProvider` and attention module hooking: ensure provider called with expected (block_idx, sample_idx) and that missing returns cause runtime errors.
- Integration: lightweight smoke run ensuring splitprompt_file name is reported and provider is used.

Pros/Cons
- Smaller, localized change than Option A; avoids touching core UNet pipeline signatures as heavily (only pass a provider object instead of embedding map). 
- Still requires attention call sites to be updated to query provider.
- Simpler to implement and easier to unit test.

---

## Recommended approach
- Implement **Option B (PromptProvider)** if you prefer smaller, safer changes with simple test coverage and quick rollout.
- Implement **Option A** if you want full precision and the cleanest semantics for per-block prompts (recommended if you expect the feature to be central and used heavily; it is more work but more exact).

Either approach must include:
- **Fail-fast behavior** when split prompts are missing or incompatible.
- **Unit tests** covering positive and negative cases.
- **Clear debug output** and documentation in `docs/SPLIT_PROMPT_IMPLEMENTATION.md` and `docs/CHANGELOG.md`.

---

## Example RuntimeError messages (user-friendly)
- "SplitPrompt: missing cached embedding for dataset <path> — re-run dataset caching or supply `split_prompt.safetensors`."
- "SplitPrompt: embedding token length mismatch (split prompt len=XXX vs model expect=YYY)."
- "SplitPrompt routing error: provider failed to return embedding for block 32 (dataset=<path>, sample=0)."

---

## Tests to add (summary)
- testing/test_splitprompt_cache.py (already present) — ensure saving/loading works.
- testing/test_splitprompt_errors_raise.py — ensure encoding/save raises on failures (already added).
- testing/test_splitprompt_routing_optionA.py — intercept UNet calls to assert per-block embeddings are used and failures raise.
- testing/test_splitprompt_provider.py — validate provider returns/raises as expected.

---

If you want, I can:
- Implement Option B (prompt-provider hook) first as a small, well-tested increment, and then follow with Option A later if you want more exact behavior.
- Start by adding tests and minimal provider implementation to exercise the code paths.

Let me know which option to implement first and whether to proceed with a proof-of-concept PR that includes tests and minimal runtime checks.