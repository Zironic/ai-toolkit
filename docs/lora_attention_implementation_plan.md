# LoRA Attention Visualization — Implementation Plan

## Goal
Create a CLI that generates sample images as usual, and additionally produces:
- An English→token mapping JSON for the prompt
- Per-token attention heatmaps overlaid on the generated sample image for the top-N tokens
- Image outputs labeled with the English word(s) corresponding to those tokens

## High-level approach
1. During inference, replace attention processors with a `RecordingAttnProcessor` that records cross-attention tensors [B, H, T, S] per layer and step.
2. Build a prompt→token mapping utility that (a) tokenizes the prompt with offsets, (b) merges subwords into word spans, and (c) finds corresponding embedding rows in `.safetensor` files where available.
3. Aggregate recorded attentions across chosen layers/heads/timesteps, extract per-token attention maps, build spatial masks (H×W), and overlay heatmaps on images.
4. Save artifacts: `mapping.json`, raw attention tensors (NPZ/PT), and PNG overlays for top-N tokens (plus a combined overlay).

## Files to add
- `toolkit/attn_recorder.py` — Recording processor to capture attention tensors. (See previous design notes.)
- `toolkit/prompt_token_map.py` — Tokenize prompts, produce word→token indices mapping, and export JSON files.
- `toolkit/safetensor_utils.py` — Helpers to inspect safetensor files and map token ids to embedding rows.
- `tools/visualize_lora_attention.py` — CLI wrapper to run generation, collect attentions, compute heatmaps, and write outputs.
- `testing/test_prompt_token_map.py` — Unit tests for tokenization and merging behavior.
- `docs/visualize_lora_attention.md` — Short doc with usage examples and recommended aggregation choices.

## CLI design (flags)
- `--prompt` (string)
- `--model` (path/id)
- `--lora` (optional path)
- `--top-tokens N` (default 6)
- `--aggregation {avg,last,per-step}` (default `avg`)
- `--layers` (list)
- `--heads` (list)
- `--map-type {cond,delta,cond-uncond}` (default `delta`)
- `--mapping-out` (path to mapping.json)
- `--out-dir` (output directory)

## Edge cases / Notes
- Handle classifier-free guidance double passes (record cond/uncond separately). Use delta maps by default to highlight prompt-specific/LoRA-specific attention.
- If attention impl uses flash/sage and doesn't expose probabilities, attempt to reconstruct via q/k if possible or fallback with a warning.
- Merge subword tokens into words using offset mappings when available; fall back to BPE-prefix heuristics.

## Testing & Validation
- Unit tests for tokenization and subword merging (mock tokenizer).
- Recorder tests using a mock attention module to confirm recorded shapes and retrieval.
- Smoke CLI test (mock pipeline or fast local pipeline) that confirms artifacts are created.

## Next steps
1. Implement `toolkit/prompt_token_map.py` and `toolkit/safetensor_utils.py` and add unit tests.  
2. Implement `RecordingAttnProcessor` and small integration example.  
3. Add the CLI and finish tests and docs.

---
*Created: 2026-01-07*