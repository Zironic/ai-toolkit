"""CLI to generate samples and visualize token->image attention maps.

Usage (simple):
python tools/visualize_lora_attention.py --prompt "a portrait of a fox" --model <model-path> --out_dir out --top_tokens 6

This script performs:
- Tokenize prompt and export mapping JSON (word->tokens)
- Optionally load a LoRA and run sampling (with or without LoRA for delta)
- Replace UNet attention processors with RecordingAttnProcessor during sampling
- Aggregate attention, select top-N tokens by saliency, and save overlay images
"""
from __future__ import annotations

import argparse
import os
import json
from typing import List

import torch
from PIL import Image
import numpy as np

from toolkit.prompt_token_map import export_prompt_mapping_json, tokenize_prompt, merge_subwords_to_words
from toolkit.attn_recorder import RecordingAttnProcessor, token_to_spatial_masks_from_records
from toolkit.token_attention import token_sigmoid_weights, token_zscore_saliency, token_attention_mask_to_spatial
# local small helper to blend a mask on image (avoid importing heavy utils)

def blend_mask_on_image(img_pil, mask, color=(255, 0, 0), alpha=0.6):
    """Blend a normalized mask (H,W with values 0..1) over `img_pil` and return a PIL image."""
    base = img_pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, color + (0,))
    # Normalize mask to numpy uint8 L image
    import numpy as _np
    import torch as _torch
    if isinstance(mask, _torch.Tensor):
        mask_arr = (mask.detach().cpu().numpy() * 255.0).astype("uint8")
    else:
        mask_arr = (_np.array(mask) * 255.0).astype("uint8")
    mask_img = Image.fromarray(mask_arr, mode="L").resize(base.size)
    # alpha channel from mask (scaled by alpha float)
    alpha_mask = mask_img.point(lambda p: int(p * alpha))
    overlay.putalpha(alpha_mask)
    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument("--top_tokens", type=int, default=6)
    p.add_argument("--layers", type=str, default=None, help="comma-separated layer indices to include")
    p.add_argument("--heads", type=str, default=None, help="comma-separated head indices to include")
    p.add_argument("--aggregation", type=str, choices=("avg", "last"), default="avg")
    p.add_argument("--lora", type=str, default=None, help="path to a LoRA .safetensors file or local path")
    p.add_argument("--map_type", type=str, choices=("cond","uncond","delta"), default=None, help="Type of attention map to produce. If --lora is provided default is 'delta'.")
    return p.parse_args()


def prepare_pipeline_for_inference(pipe, prefer_gpu: bool = False):
    """Apply toolkit-like low-memory settings to a Diffusers pipeline or similar.

    This mirrors the common pattern used across the toolkit: enable attention
    slicing, memory-efficient attention, and model CPU offload where available.
    The default is CPU-friendly (avoid moving weights to GPU) unless
    `prefer_gpu=True` is explicitly requested.
    """
    try:
        if hasattr(pipe, 'enable_attention_slicing'):
            try:
                pipe.enable_attention_slicing()
                print("[DEBUG] enabled attention slicing")
            except Exception:
                pass
        # Try common API for xformers/memory-efficient attention
        for attr in ('enable_xformers_memory_efficient_attention', 'enable_memory_efficient_attention'):
            try:
                if hasattr(pipe, attr):
                    getattr(pipe, attr)()
                    print(f"[DEBUG] called {attr}")
            except Exception:
                pass
        # Try CPU offload helpers that some pipelines expose
        try:
            if hasattr(pipe, 'enable_model_cpu_offload'):
                try:
                    pipe.enable_model_cpu_offload(0)
                    print("[DEBUG] enabled model CPU offload")
                except Exception:
                    pass
        except Exception:
            pass
        # Do not move to GPU automatically: keep pipeline on CPU unless explicitly requested
        if prefer_gpu:
            try:
                pipe.to('cuda')
                print("[DEBUG] moved pipeline to cuda (prefer_gpu=True)")
            except Exception:
                pass
    except Exception:
        pass
    return pipe


def load_pipeline(model_name: str):
    # If the user passed a single-file checkpoint path (safetensors/ckpt/etc.),
    # avoid calling the repo-aware `load_model_for_inference` which may attempt
    # to resolve and unpack components into CPU memory (causing the warning and OOM).
    import os
    _lower = str(model_name).lower() if model_name is not None else ''
    if os.path.exists(model_name) and os.path.isfile(model_name):
        print(f"[DEBUG] detected model path is a single file: {model_name}; skipping repo-aware loader")
        _skip_repo_loader = True
    elif _lower.endswith(('.ckpt', '.safetensors', '.pt', '.pth')):
        print(f"[DEBUG] detected model name looks like a single-file checkpoint; skipping repo-aware loader")
        _skip_repo_loader = True
    else:
        _skip_repo_loader = False

    if not _skip_repo_loader:
        # Prefer the repository's model resolver/loader that knows local model layout and config
        try:
            from toolkit.model_utils import load_model_for_inference
            sd = load_model_for_inference(model_name, device='cpu', dtype='float32', apply_lora=False)
            pipe = getattr(sd, 'pipeline', None)
            if pipe is not None:
                try:
                    prepare_pipeline_for_inference(pipe, prefer_gpu=False)
                except Exception:
                    pass
                return pipe
        except Exception as e:
            print(f"[WARN] load_model_for_inference early load failed: {e}; falling back to diffusers loader")

    # Lazy import to avoid heavy deps at module import; prefer low-memory / offload loading options
    from diffusers import DiffusionPipeline
    import tempfile
    import shutil

    # If this is a single-file model, try the pipeline's `from_single_file` helper first
    if os.path.exists(model_name) and os.path.isfile(model_name):
        try:
            # Prefer the stable-diffusion pipeline single-file loader when possible
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

            print(f"[DEBUG] attempting StableDiffusionPipeline.from_single_file({model_name})")
            pipe = StableDiffusionPipeline.from_single_file(model_name, torch_dtype=None)
            print("[DEBUG] loaded pipeline via from_single_file")
            try:
                prepare_pipeline_for_inference(pipe, prefer_gpu=False)
            except Exception:
                pass
            return pipe
        except Exception as e:
            print(f"[DEBUG] from_single_file attempt failed: {e}; continuing to device_map/offload attempt")

    # Try an offload/device_map-based load next (streams weights and offloads if possible)
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="pipe_offload_")
        print(f"[DEBUG] attempting to load with device_map='auto' and offload_folder={tmpdir}")
        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_name,
                device_map="auto",
                offload_folder=tmpdir,
                low_cpu_mem_usage=True,
                torch_dtype=None,
                use_safetensors=True,
            )
            print("[DEBUG] loaded pipeline with device_map='auto' offload")
        except Exception as e:
            print(f"[DEBUG] device_map/offload load failed: {e}; trying low_cpu_mem_usage fallback")
            try:
                pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=None, low_cpu_mem_usage=True, use_safetensors=True)
            except TypeError:
                pipe = DiffusionPipeline.from_pretrained(model_name)
    except Exception as e:
        print(f"[WARN] offload attempt failed: {e}; falling back to simple from_pretrained")
        try:
            pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=None, low_cpu_mem_usage=True, use_safetensors=True)
        except TypeError:
            pipe = DiffusionPipeline.from_pretrained(model_name)
    finally:
        # We keep the tmpdir around intentionally while the pipeline might reference it; do not remove here.
        pass

    try:
        prepare_pipeline_for_inference(pipe, prefer_gpu=False)
    except Exception:
        pass
    return pipe


def replace_processors_with_recorder(pipe, recorder: RecordingAttnProcessor):
    """Try to install the `recorder` onto any attention processor available.

    Returns True if any replacement was applied, False otherwise (and prints a warning).
    """
    replaced = False
    # First try the common location: pipe.unet.set_attn_processor
    try:
        if hasattr(pipe, 'unet') and hasattr(pipe.unet, 'set_attn_processor'):
            try:
                pipe.unet.set_attn_processor(recorder)
                replaced = True
            except Exception as e:
                print(f"[WARN] failed to set attn processor on pipe.unet: {e}")
    except Exception:
        # defensive: some pipelines might not expose unet
        pass

    # Fallback: search all modules (either under unet or whole pipeline) for set_attn_processor
    search_root = getattr(pipe, 'unet', pipe)
    try:
        for name, module in search_root.named_modules():
            if hasattr(module, 'set_attn_processor'):
                try:
                    module.set_attn_processor(recorder)
                    replaced = True
                except Exception:
                    pass
    except Exception:
        # named_modules may not be available on the search_root (defensive)
        pass

    if not replaced:
        print("[WARN] No `set_attn_processor` was found on the pipeline; attention recording may be unavailable.")
    return replaced


def select_top_words_by_saliency(prompt: str, tokenizer, recorder: RecordingAttnProcessor, top_k: int) -> List[dict]:
    """Return top_k words (merged) by token saliency computed from token embeddings.

    Uses token_zscore_saliency on tokenizer embeddings when available; otherwise
    uses summed attention mass across recorded attention maps as a proxy.
    """
    tok = tokenize_prompt(prompt, tokenizer)
    merged = merge_subwords_to_words(tok['tokens'], tok['offsets'])

    # compute per-token saliency: if tokenizer has embeddings accessible use them
    if hasattr(tokenizer, 'get_input_embeddings'):
        emb_layer = tokenizer.get_input_embeddings()
        try:
            emb = emb_layer.weight if hasattr(emb_layer, 'weight') else None
            if emb is not None:
                token_ids = tok['ids']
                emb_vecs = emb[token_ids]
                sal = token_zscore_saliency(emb_vecs.unsqueeze(0)).squeeze(0)  # [T]
            else:
                raise Exception('no emb weight')
        except Exception:
            sal = _saliency_from_attn(recorder, len(tok['ids']))
    else:
        sal = _saliency_from_attn(recorder, len(tok['ids']))

    # map word->sum of token saliencies
    word_scores = []
    for w in merged:
        token_indices = w['token_indices']
        score = float(sal[token_indices].sum().item())
        word_scores.append({"word": w['word'], "token_indices": token_indices, "score": score, "token_texts": w['token_texts']})
    word_scores = sorted(word_scores, key=lambda x: x['score'], reverse=True)
    return word_scores[:top_k]


def _saliency_from_attn(recorder: RecordingAttnProcessor, T: int):
    # sum recorded attentions over layers/heads and source positions -> [T]
    records = recorder.get_records()
    if len(records) == 0:
        return torch.zeros((T,))
    # take head-avg then layer-average then sum over source positions
    maps = []
    for r in records:
        # r['attn'] [B,H,T,S] -> mean over H, sum over S -> [B,T]
        a = r['attn'].mean(dim=1).sum(dim=-1)  # [B,T]
        maps.append(a)
    stacked = torch.stack(maps, dim=0).mean(dim=0).squeeze(0)  # [T]
    return stacked


def make_overlay_and_save(img_pil: Image.Image, mask: np.ndarray, out_path: str, label: str):
    # mask: [H, W] normalized 0..1
    overlay = blend_mask_on_image(img_pil, mask)
    # label into filename
    out_fname = os.path.splitext(os.path.basename(out_path))[0] + f"_{label}.png"
    out_full = os.path.join(os.path.dirname(out_path), out_fname)
    overlay.save(out_full)
    return out_full


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # load model via toolkit helper (preferred). Do NOT fall back to `load_pipeline` because
    # that path may unpack weights into CPU RAM and cause OOMs. If the toolkit loader fails,
    # abort early and ask the user to provide a path that supports streaming/offload.
    sd = None
    # If the model argument looks like a single-file checkpoint, call the toolkit loader
    # with an explicit hint dict *first* so it receives the component paths it may expect
    # (avoids an unnecessary failure + retry cycle).
    _lower = str(args.model).lower() if args.model is not None else ''
    is_single_file = False
    if os.path.exists(args.model) and os.path.isfile(args.model):
        is_single_file = True
    elif _lower.endswith(('.ckpt', '.safetensors', '.pt', '.pth')):
        is_single_file = True

    try:
        from toolkit.model_utils import load_model_for_inference
        if is_single_file:
            hint_cfg = {
                'name_or_path': args.model,
                'extras_name_or_path': args.model,
                'unet_path': args.model,
                'vae_path': args.model,
                'te_name_or_path': args.model,
                # pass additional components users may have packaged in the single-file
                'feature_extractor': args.model,
                'feature_extractor_path': args.model,
                'image_encoder': args.model,
                'image_encoder_path': args.model,
                'safety_checker': args.model,
                'safety_checker_path': args.model,
                'low_vram': True,
            }
            print(f"[DEBUG] detected single-file model; calling toolkit loader with hint_cfg keys {list(hint_cfg.keys())}")
            sd = load_model_for_inference(hint_cfg, device='cpu', dtype='float32', apply_lora=False)
        else:
            sd = load_model_for_inference(args.model, device='cpu', dtype='float32', apply_lora=False)
    except Exception as e:
        print(f"[ERROR] load_model_for_inference failed: {e}; aborting. Provide a local model path or ensure the model can be loaded with low-memory/offload options.")
        return

    pipe = getattr(sd, 'pipeline', None)
    if pipe is None:
        print(f"[ERROR] model loaded but no `pipeline` attribute found for {args.model}; aborting.")
        return

    # load tokenizer (robust with fallbacks)
    from transformers import AutoTokenizer, CLIPTokenizerFast
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"AutoTokenizer.from_pretrained failed for {args.model}: {e}; trying pipeline/model tokenizer and CLIP fallbacks")
        tokenizer = (getattr(sd, 'tokenizer', None) if sd is not None else None) or getattr(pipe, 'tokenizer', None)
        if tokenizer is None:
            # Try to infer a tokenizer from the model/pipeline's text_encoder config if available
            te = (getattr(sd, 'text_encoder', None) if sd is not None else None) or getattr(pipe, 'text_encoder', None)
            tried = False
            if te is not None and hasattr(te, 'config'):
                name = getattr(te.config, 'name_or_path', None)
                if name:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(name)
                        tried = True
                    except Exception:
                        tried = False
            if not tokenizer:
                # Final fallback: default to a CLIP tokenizer
                try:
                    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
                except Exception:
                    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    # create mapping JSON
    mapping_path = os.path.join(args.out_dir, "mapping.json")
    export_prompt_mapping_json(args.prompt, tokenizer, safetensor_paths=[], out_path=mapping_path)

    # Decide map_type (default delta when LoRA provided)
    map_type = getattr(args, 'map_type', None)
    if map_type is None and args.lora is not None:
        map_type = 'delta'
    if map_type is None:
        map_type = 'cond'

    def run_and_record(pipe, prompt, seed=42):
        from toolkit.attn_recorder import RecordingAttnProcessor
        rec = RecordingAttnProcessor()
        replaced = replace_processors_with_recorder(pipe, rec)
        # Debug: report pipeline and whether attn processor was installed
        try:
            pipe_type = type(pipe)
            device_attr = getattr(pipe, 'device', None)
            print(f"[DEBUG] running pipeline {pipe_type} device={device_attr} unet_present={hasattr(pipe, 'unet')} processor_installed={replaced}")
        except Exception:
            pass

        # Build a safe generator (prefer CPU to avoid OOM; pipeline prepared with low-memory options)
        gen_device = 'cpu'
        try:
            # If the pipeline explicitly moved to CUDA (prefer_gpu), honor it; otherwise keep CPU
            pd = getattr(pipe, 'device', None)
            if pd is not None and str(pd).startswith('cuda'):
                gen_device = 'cuda'
        except Exception:
            pass
        try:
            generator = torch.Generator(device=gen_device)
        except Exception:
            generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)

        try:
            out = pipe(prompt, num_inference_steps=8, generator=generator)
            image = out.images[0]
            records = rec.get_records()
            print(f"[DEBUG] pipeline run completed; recorded {len(records)} records")
            return image, records, rec
        except Exception as e:
            # Print traceback to help diagnose hangs/errors during real runs
            import traceback

            traceback.print_exc()
            print(f"[ERROR] pipeline run failed: {e}")
            # Return a white fallback image and empty records so we still produce outputs
            H = getattr(getattr(pipe, 'vae', None), 'sample_size', 64)
            from PIL import Image as _Image

            image = _Image.new('RGB', (H, H), color='white')
            return image, [], rec
        except Exception as e:
            # Print traceback to help diagnose hangs/errors during real runs
            import traceback

            traceback.print_exc()
            print(f"[ERROR] pipeline run failed: {e}")
            # Return a white fallback image and empty records so we still produce outputs
            H = getattr(getattr(pipe, 'vae', None), 'sample_size', 64)
            from PIL import Image as _Image

            image = _Image.new('RGB', (H, H), color='white')
            return image, [], rec

    # Baseline run (no LoRA applied)
    image_base, records_base, rec_base = run_and_record(pipe, args.prompt)

    records_to_use = records_base
    image_to_use = image_base

    lora_network = None
    if args.lora is not None:
        # load LoRA and perform a second run with it applied
        from safetensors.torch import load_file
        from toolkit.lora_special import LoRASpecialNetwork

        lora_state = load_file(args.lora)
        # infer a simple lora_dim if present in keys similar to assistant loader
        lora_dim = None
        for k in lora_state.keys():
            kn = k.lower()
            if 'lora_a' in kn or 'lora_down' in kn or 'lora_a.weight' in kn:
                v = lora_state[k]
                if hasattr(v, 'shape') and len(v.shape) >= 2:
                    lora_dim = int(v.shape[0])
                    break
        lora_dim = lora_dim or 4

        # Build LoRA network and apply
        text_encoder = getattr(pipe, 'text_encoder', None)
        unet = getattr(pipe, 'unet', None)
        lora_network = LoRASpecialNetwork(text_encoder=text_encoder, unet=unet, lora_dim=lora_dim, multiplier=1.0, alpha=lora_dim)
        # apply network to the model
        try:
            lora_network.apply_to(text_encoder, unet, apply_text_encoder=False, apply_unet=True)
        except Exception:
            # fallback to generic apply_to if signature differs
            try:
                lora_network.apply_to()
            except Exception:
                pass
        # load weights and activate
        try:
            lora_network._update_torch_multiplier()
        except Exception:
            pass
        lora_network.load_weights(lora_state)
        lora_network.is_active = True

        # Run with LoRA active
        image_lora, records_lora, rec_lora = run_and_record(pipe, args.prompt)

        if map_type == 'delta':
            # compute delta maps = lora - base
            records_to_use = []
            # Align records by index if possible
            L = max(len(records_base), len(records_lora))
            for i in range(L):
                a = records_lora[i]['attn'] if i < len(records_lora) else torch.zeros_like(records_lora[0]['attn'])
                b = records_base[i]['attn'] if i < len(records_base) else torch.zeros_like(records_base[0]['attn'])
                # simple delta record
                records_to_use.append({"module": records_lora[i].get('module', None) if i < len(records_lora) else None,
                                       "block_idx": i,
                                       "attn": (a - b).cpu()})
            image_to_use = image_lora
        elif map_type == 'cond':
            records_to_use = records_lora
            image_to_use = image_lora
        elif map_type == 'uncond':
            # not implemented separately yet; default to base
            records_to_use = records_base
            image_to_use = image_base

    # compute top-K words using the baseline tokenizer+recording (use base rec for saliency)
    top_words = select_top_words_by_saliency(args.prompt, tokenizer, rec_base, args.top_tokens)

    # convert recorded attentions to masks for top tokens
    records = records_to_use
    if len(records) == 0:
        print("No attention records captured; saving blank overlays as fallback.")
        H, W = image_to_use.height, image_to_use.width
        blank_mask = np.zeros((H, W))
        for wi, w in enumerate(top_words):
            out_path = os.path.join(args.out_dir, f"sample_0_{wi}.png")
            make_overlay_and_save(image_to_use, blank_mask, out_path, w['word'])
        # still export recorder outputs (may be empty) so downstream tools see artifacts
        rec_base.export_numpy(os.path.join(args.out_dir, "attn_base.npz"))
        # Backwards compat export
        rec_base.export_numpy(os.path.join(args.out_dir, "attn_data.npz"))
        if lora_network is not None:
            try:
                rec_lora.export_numpy(os.path.join(args.out_dir, "attn_lora.npz"))
            except Exception:
                pass
            from toolkit.attn_recorder import RecordingAttnProcessor
            delta_rec = RecordingAttnProcessor()
            delta_rec.records = records_to_use
            delta_rec.export_numpy(os.path.join(args.out_dir, "attn_delta.npz"))
        print(f"Saved mapping to {mapping_path} and overlays to {args.out_dir}")
        return

    token_indices_flat = []
    labels = []
    for w in top_words:
        token_indices_flat.extend(w['token_indices'])
        labels.append(w['word'])

    # compute spatial masks [B, K, H, W]
    H, W = image_to_use.height, image_to_use.width

    maps = token_to_spatial_masks_from_records(records, token_indices_flat, H, W)
    # maps: [B, K, H, W]
    # combine per-word indices
    idx = 0
    for wi, w in enumerate(top_words):
        k = len(w['token_indices'])
        m = maps[0, idx: idx + k].sum(axis=0)  # [H,W]
        # normalize
        m = (m - m.min()) / (m.max() - m.min() + 1e-9)
        out_path = os.path.join(args.out_dir, f"sample_0_{wi}.png")
        make_overlay_and_save(image_to_use, m, out_path, w['word'])
        idx += k

    # save raw recorder output(s)
    rec_base.export_numpy(os.path.join(args.out_dir, "attn_base.npz"))
    # Backwards compatibility: some tools expect attn_data.npz
    rec_base.export_numpy(os.path.join(args.out_dir, "attn_data.npz"))
    if lora_network is not None:
        rec_lora.export_numpy(os.path.join(args.out_dir, "attn_lora.npz"))
        # if delta map created, also dump delta arrays
        from toolkit.attn_recorder import RecordingAttnProcessor
        delta_rec = RecordingAttnProcessor()
        delta_rec.records = records_to_use
        delta_rec.export_numpy(os.path.join(args.out_dir, "attn_delta.npz"))
        # also export a combined attn_data.npz containing both base and lora if desired (kept simple: copy base over)
        try:
            rec_base.export_numpy(os.path.join(args.out_dir, "attn_data.npz"))
        except Exception:
            pass

    print(f"Saved mapping to {mapping_path} and overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
