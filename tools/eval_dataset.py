"""Dataset evaluation CLI and helpers.

This module provides a simple command line tool to run dataset evaluation using the
`toolkit.util.loss_utils.run_dataset_evaluation` orchestration. It constructs a
`DataLoader` for a dataset folder, loads a StableDiffusion model from a ModelConfig,
and builds a compute_fn that reproduces the approximate training-time per-example
loss calculation (noise prediction vs loss target).

The intent is to provide a reusable backend utility that can be invoked by
background workers or manually from the command line.
"""
from typing import Callable, List, Dict, Any, Optional
import argparse
import os
import json
import math

import torch

from toolkit.config_modules import ModelConfig, DatasetConfig
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.data_loader import get_dataloader_from_datasets
from toolkit.util.loss_utils import run_dataset_evaluation, per_sample_from_loss_tensor
from toolkit.prompt_utils import concat_prompt_embeds
from toolkit.paths import MODELS_PATH, ORIG_CONFIGS_ROOT, DIFFUSERS_CONFIGS_ROOT


def resolve_local_model_path(name_or_path: str):
    """Try common local locations (MODELS_PATH, DIFFUSERS_CONFIGS_ROOT, ORIG_CONFIGS_ROOT) for a downloaded model.

    Returns the path if found, otherwise None.
    """
    # if it's already a path, return it
    if os.path.exists(name_or_path):
        return name_or_path

    candidates = []
    # exact under MODELS_PATH
    if MODELS_PATH:
        candidates.append(os.path.join(MODELS_PATH, name_or_path))
        # try stripping user/owner
        if '/' in name_or_path:
            candidates.append(os.path.join(MODELS_PATH, name_or_path.split('/')[-1]))
        candidates.append(os.path.join(MODELS_PATH, name_or_path.replace('/', '_')))

    # DIFFUSERS and ORIG configs
    candidates.append(os.path.join(DIFFUSERS_CONFIGS_ROOT, name_or_path))
    candidates.append(os.path.join(DIFFUSERS_CONFIGS_ROOT, name_or_path.split('/')[-1]))
    candidates.append(os.path.join(ORIG_CONFIGS_ROOT, name_or_path))
    candidates.append(os.path.join(ORIG_CONFIGS_ROOT, name_or_path.split('/')[-1]))

    for c in candidates:
        try:
            if c and os.path.exists(c):
                return c
        except Exception:
            continue
    return None


from toolkit.model_utils import load_model_for_inference


def build_sd_model(name_or_path: str, device: str = "cpu", dtype: str = "float32") -> StableDiffusion:
    """Create and load a StableDiffusion wrapper given a model path/name.

    This uses the inference-safe loader in `toolkit.model_utils` which prefers
    local model folders and avoids training-only side-effects.
    """
    sd = load_model_for_inference(name_or_path, device=device, dtype=dtype)
    return sd


def make_compute_fn_for_sd(sd: StableDiffusion, device: str = "cpu", normalize_loss: bool = True, samples_per_image: int = 8, debug_noise: bool = False, debug_noise_timestep_type: str = 'sigmoid', fixed_noise_std: float | None = 0.6, debug_captions: bool = False, caption_ablation: str = 'none', caption_ablation_compare: bool = False, log_conditioning: bool = False) -> Callable[[Any], List[Dict[str, Any]]]:
    """Return a compute_fn(batch) suitable for evaluate_dataset / run_dataset_evaluation.

    The compute_fn expects a batch-like object with attributes used by the training
    code (see DataLoaderBatchDTO). We implement a conservative approximation of the
    training loss calculation:
      - compute or use latents
      - sample Gaussian noise and timesteps
      - add noise to latents using the model scheduler
      - compute noise prediction with the model
      - compute a per-element MSE loss vs model.get_loss_target (fallback to noise)
      - reduce the loss to a per-sample scalar and return records with 'path','dataset','caption','loss'
    """

    def compute_fn(batch) -> List[Dict[str, Any]]:
        results = []
        # Safety: enforce evaluation mode and disable any configured dropout to ensure
        # deterministic/no-dropout behavior during evaluation (protect against jobs/new defaults)
        try:
            try:
                sd.te_eval()
            except Exception:
                pass
            try:
                sd.vae.eval()
            except Exception:
                pass
            try:
                sd.unet.eval()
            except Exception:
                pass
            # Zero-out nn.Dropout probabilities where possible
            import torch.nn as _nn
            changed = False
            for root in (getattr(sd, 'unet', None), getattr(sd, 'text_encoder', None), getattr(sd, 'vae', None)):
                if root is None:
                    continue
                for m in root.modules():
                    if isinstance(m, _nn.Dropout):
                        try:
                            if getattr(m, 'p', None) is not None and float(m.p) != 0.0:
                                m.p = 0.0
                                changed = True
                        except Exception:
                            pass
            # Also, if model_config includes an explicit dropout field, override it
            try:
                if hasattr(sd, 'model_config') and sd.model_config is not None and hasattr(sd.model_config, 'dropout'):
                    if getattr(sd.model_config, 'dropout') is not None and float(sd.model_config.dropout) != 0.0:
                        sd.model_config.dropout = 0.0
                        changed = True
            except Exception:
                pass
            if changed:
                print('Enforced dropout=0.0 for evaluation (overrode model config or modules)')
        except Exception:
            # do not fail the evaluation if enforcement cannot run
            pass

        with torch.no_grad(): 
            # prepare prompt embeddings
            if getattr(batch, 'prompt_embeds', None) is not None:
                conditional_embeds = batch.prompt_embeds
            else:
                # build from file_items' raw captions
                captions = [getattr(fi, 'raw_caption', '') or '' for fi in batch.file_items]
                try:
                    conditional_embeds = sd.encode_prompt(captions)
                except Exception:
                    # fallback: empty embeddings
                    conditional_embeds = sd.encode_prompt([""] * len(captions))

            # latents: prefer cached latents if present
            if getattr(batch, 'latents', None) is not None:
                latents = batch.latents.to(sd.device_torch)
            elif getattr(batch, 'tensor', None) is not None:
                # convert batched tensor into a list of images and explicitly pass
                # the VAE device/dtype to avoid dtype mismatch between inputs and
                # VAE parameters (e.g., fp16 vs bf16). When running on CUDA we also
                # run the encode step under torch.cuda.amp.autocast with the
                # VAE dtype so any automatic autocast uses bfloat16 (matching VAE
                # weights) instead of defaulting to float16.
                images = [img for img in batch.tensor]
                if ("cuda" in str(device)) and sd.vae_torch_dtype in (torch.float16, torch.bfloat16):
                    try:
                        # Use torch.autocast which accepts a device/string argument on this
                        # runtime (avoids device_type kwarg compatibility issues).
                        with torch.autocast("cuda", dtype=sd.vae_torch_dtype):
                            latents = sd.encode_images(images, device=sd.vae_device_torch, dtype=sd.vae_torch_dtype)
                    except Exception:
                        # fallback to encode without autocast if the context isn't supported
                        latents = sd.encode_images(images, device=sd.vae_device_torch, dtype=sd.vae_torch_dtype)
                else:
                    latents = sd.encode_images(images, device=sd.vae_device_torch, dtype=sd.vae_torch_dtype)
            else:
                # nothing to evaluate
                return results

            bs = latents.shape[0]
            # sample noise and timesteps
            noise = torch.randn_like(latents).to(sd.device_torch)
            # choose timesteps according to selected strategy
            try:
                max_steps = int(sd.noise_scheduler.config.num_train_timesteps)
            except Exception:
                # fallback to 1000
                max_steps = 1000

            def sample_timesteps(bs_local: int):
                ttype = debug_noise_timestep_type
                if ttype == 'uniform':
                    return torch.randint(0, max_steps, (bs_local,), device=sd.device_torch, dtype=torch.long)
                if ttype == 'one_step':
                    return torch.zeros((bs_local,), device=sd.device_torch, dtype=torch.long)
                if ttype == 'cubic_early':
                    r = torch.rand(bs_local, device=sd.device_torch)
                    idx = (r ** 3 * (max_steps)).to(torch.long)
                    return torch.clamp(idx, 0, max_steps - 1)
                if ttype == 'cubic_late':
                    r = torch.rand(bs_local, device=sd.device_torch)
                    idx = ((1.0 - (1.0 - r) ** 3) * (max_steps)).to(torch.long)
                    return torch.clamp(idx, 0, max_steps - 1)
                if ttype == 'sigmoid':
                    # Match trainer default: generate timesteps by applying sigmoid to normal noise
                    # This produces a bell-like distribution biased toward the center when turned into timesteps
                    r = torch.randn((bs_local,), device=sd.device_torch)
                    tvals = torch.sigmoid(r)
                    idx = ((1.0 - tvals) * (max_steps)).to(torch.long)
                    return torch.clamp(idx, 0, max_steps - 1)
                # default
                return torch.randint(0, max_steps, (bs_local,), device=sd.device_torch, dtype=torch.long)

            if fixed_noise_std is not None:
                # Apply direct mixture to achieve fixed noise std s such that
                # noisy = sqrt(1 - s^2) * latents + s * noise. We still supply
                # a placeholder timestep tensor because some model implementations
                # expect a tensor and call `.to()` on it.
                s = float(fixed_noise_std)
                if s < 0 or s > 1:
                    raise ValueError('fixed_noise_std must be between 0 and 1')
                lat_coeff = float((1.0 - s * s) ** 0.5)
                noisy = latents * lat_coeff + noise * s
                # Use a mid-range timestep so model timestep embeddings are valid.
                try:
                    mid_t = max(1, max_steps // 2)
                    timesteps = torch.full((bs,), mid_t, device=sd.device_torch, dtype=torch.long)
                except Exception:
                    timesteps = torch.zeros((bs,), device=sd.device_torch, dtype=torch.long)
            else:
                timesteps = sample_timesteps(bs)
                noisy = sd.noise_scheduler.add_noise(latents, noise, timesteps)

            # Debug: inspect prompt embeddings to ensure captions are being used by the model
            try:
                if debug_captions:
                    # gather caption strings
                    captions_list = [getattr(fi, 'raw_caption', '') or '' for fi in batch.file_items]
                    te = conditional_embeds
                    emb_info = None
                    try:
                        if hasattr(te, 'text_embeds'):
                            te_obj = te.text_embeds
                            if isinstance(te_obj, torch.Tensor):
                                shapes = [tuple(te_obj.shape)]
                                try:
                                    norms = te_obj.view(te_obj.shape[0], -1).norm(dim=1).detach().cpu().tolist()
                                except Exception:
                                    norms = None
                            elif isinstance(te_obj, (list, tuple)):
                                shapes = [tuple(t.shape) if hasattr(t, 'shape') else None for t in te_obj]
                                norms = []
                                for t in te_obj:
                                    try:
                                        norms.append(t.view(t.shape[0], -1).norm(dim=1).detach().cpu().tolist())
                                    except Exception:
                                        norms.append(None)
                            else:
                                shapes = [None]
                                norms = None
                            # determine if classifier-free guidance will be used (heuristic)
                            guidance_possible = False
                            if isinstance(te_obj, torch.Tensor):
                                if te_obj.shape[0] == latents.shape[0] * 2:
                                    guidance_possible = True
                            elif isinstance(te_obj, (list, tuple)):
                                # assume lists are XL-style which typically include both
                                guidance_possible = True
                            emb_info = {'shapes': shapes, 'norms': norms, 'guidance_possible': guidance_possible}
                        else:
                            emb_info = None
                    except Exception as e:
                        emb_info = {'error': str(e)}
                    msg = f"[EVAL-CAPTION-DEBUG] captions={captions_list}, embeds={emb_info}"
                    print(msg)
                    # Also persist debug info into a file in dataset folder so it is
                    # available even if worker logs are not accessible.
                    try:
                        ds_path = None
                        if len(batch.file_items) > 0 and getattr(batch.file_items[0], 'dataset_config', None) is not None:
                            ds_path = getattr(batch.file_items[0].dataset_config, 'dataset_path', None) or getattr(batch.file_items[0].dataset_config, 'folder_path', None)
                        if ds_path:
                            import time
                            safe_name = f".eval_caption_debug_{int(time.time())}.log"
                            p = os.path.join(ds_path, safe_name)
                            try:
                                with open(p, 'a', encoding='utf-8') as _f:
                                    _f.write(msg + '\n')
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass

            # obtain model prediction (instrumented)
            # Be defensive: different SD implementations may accept different kwargs and
            # may return either a single prediction or a (prediction, conditional) tuple.
            noise_pred = None
            conditional_pred = None
            try:
                out = sd.predict_noise(noisy, text_embeddings=conditional_embeds, timestep=timesteps, batch=batch, return_conditional_pred=True)
                if isinstance(out, tuple) and len(out) == 2:
                    noise_pred, conditional_pred = out
                else:
                    noise_pred = out
                    conditional_pred = None
            except TypeError:
                try:
                    out = sd.predict_noise(noisy, prompt_embeds=conditional_embeds, timestep=timesteps, batch=batch, return_conditional_pred=True)
                    if isinstance(out, tuple) and len(out) == 2:
                        noise_pred, conditional_pred = out
                    else:
                        noise_pred = out
                        conditional_pred = None
                except TypeError:
                    # older signatures may not accept return_conditional_pred; fallback to single-return calls
                    try:
                        noise_pred = sd.predict_noise(noisy, text_embeddings=conditional_embeds, timestep=timesteps, batch=batch)
                    except TypeError:
                        noise_pred = sd.predict_noise(noisy, prompt_embeds=conditional_embeds, timestep=timesteps, batch=batch)
                    conditional_pred = None

            # optional: perform caption ablation and/or conditioning stats
            _mse_all = None
            _per_sample_mse = None
            if caption_ablation != 'none' or log_conditioning:
                try:
                    # helper to build ablated PromptEmbeds matching the original
                    def _make_ablated(pe, mode='zero'):
                        pe2 = pe.clone()
                        if isinstance(pe2.text_embeds, (list, tuple)):
                            new = []
                            for t in pe2.text_embeds:
                                if mode == 'zero':
                                    new.append(torch.zeros_like(t))
                                else:
                                    new.append(torch.randn_like(t))
                            pe2.text_embeds = new
                        else:
                            if mode == 'zero':
                                pe2.text_embeds = torch.zeros_like(pe2.text_embeds)
                            else:
                                pe2.text_embeds = torch.randn_like(pe2.text_embeds)
                        return pe2

                    if caption_ablation != 'none':
                        ablated = _make_ablated(conditional_embeds, caption_ablation)
                        noise_pred_abl = None
                        cond_pred_abl = None
                        try:
                            out = sd.predict_noise(noisy, text_embeddings=ablated, timestep=timesteps, batch=batch, return_conditional_pred=True)
                            if isinstance(out, tuple) and len(out) == 2:
                                noise_pred_abl, cond_pred_abl = out
                            else:
                                noise_pred_abl = out
                                cond_pred_abl = None
                        except TypeError:
                            try:
                                out = sd.predict_noise(noisy, prompt_embeds=ablated, timestep=timesteps, batch=batch, return_conditional_pred=True)
                                if isinstance(out, tuple) and len(out) == 2:
                                    noise_pred_abl, cond_pred_abl = out
                                else:
                                    noise_pred_abl = out
                                    cond_pred_abl = None
                            except TypeError:
                                try:
                                    noise_pred_abl = sd.predict_noise(noisy, text_embeddings=ablated, timestep=timesteps, batch=batch)
                                except TypeError:
                                    noise_pred_abl = sd.predict_noise(noisy, prompt_embeds=ablated, timestep=timesteps, batch=batch)
                                cond_pred_abl = None

                        try:
                            _mse_all = float(((noise_pred.float() - noise_pred_abl.float()) ** 2).mean().item())
                            _per_sample_mse = ((noise_pred.float() - noise_pred_abl.float()).view(noise_pred.shape[0], -1).pow(2).mean(dim=1).detach().cpu().tolist())
                        except Exception:
                            _mse_all = None
                            _per_sample_mse = None

                    # conditioning stats (means/stds)
                    if log_conditioning:
                        try:
                            te_obj = conditional_embeds.text_embeds
                            if isinstance(te_obj, torch.Tensor):
                                flat = te_obj.view(te_obj.shape[0], -1)
                                means = flat.mean(dim=1).detach().cpu().tolist()
                                stds = flat.std(dim=1).detach().cpu().tolist()
                            else:
                                means = []
                                stds = []
                                for t in te_obj:
                                    f = t.view(t.shape[0], -1)
                                    means.append(float(f.mean().detach().cpu().tolist()[0]) if hasattr(f.mean(),'tolist') else float(f.mean().detach().cpu()))
                                    stds.append(float(f.std().detach().cpu().tolist()[0]))
                        except Exception:
                            means = None
                            stds = None

                        try:
                            caps = captions_list if 'captions_list' in locals() else None
                        except Exception:
                            caps = None

                        print(f"[EVAL-CONDITIONING] captions={caps}, means={means}, stds={stds}, ablation_mse={_mse_all}, per_sample_mse={_per_sample_mse}")
                        # also persist to dataset folder
                        try:
                            ds_path = None
                            if len(batch.file_items) > 0 and getattr(batch.file_items[0], 'dataset_config', None) is not None:
                                ds_path = getattr(batch.file_items[0].dataset_config, 'dataset_path', None) or getattr(batch.file_items[0].dataset_config, 'folder_path', None)
                            if ds_path:
                                import time
                                safe_name = f".eval_caption_cond_{int(time.time())}.log"
                                p = os.path.join(ds_path, safe_name)
                                try:
                                    with open(p, 'a', encoding='utf-8') as _f:
                                        _f.write(json.dumps({'captions': caps, 'means': means, 'stds': stds, 'ablation_mse': _mse_all, 'per_sample_mse': _per_sample_mse}) + '\n')
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass

            # determine target for loss
            try:
                target = sd.get_loss_target(noise=noise, batch=batch, timesteps=timesteps)
            except Exception:
                target = noise

            # Optionally run multiple stochastic samples per image and average.
            # If caption_ablation_compare is enabled we compute losses for both original
            # and ablated conditionings and record the delta (blank - caption) as the 'loss'.
            samples = max(1, int(samples_per_image))
            sum_per_sample = None
            sum_per_sample_abl = None
            # prepare per-image lists to capture raw sample losses
            samples_list_local = [ [] for _ in range(bs) ]
            samples_list_local_abl = [ [] for _ in range(bs) ]
            samples_list_local_delta = [ [] for _ in range(bs) ]

            # prepare ablated embeddings if requested (zero by default)
            ablated_embeds = None
            if caption_ablation != 'none' or caption_ablation_compare:
                def _make_ablated(pe, mode='zero'):
                    # Prefer the model's empty prompt encoding when available (in-distribution baseline)
                    try:
                        empty_pe = sd.encode_prompt([""])
                        # expand to batch size if supported
                        try:
                            empty_pe = empty_pe.expand_to_batch(bs)
                        except Exception:
                            # fall back to manual expansion for simple tensor shapes
                            if hasattr(empty_pe, 'text_embeds') and isinstance(empty_pe.text_embeds, torch.Tensor):
                                empty_pe.text_embeds = empty_pe.text_embeds.expand(bs, -1)
                        return empty_pe
                    except Exception:
                        # fallback to previous behavior (zeros or random)
                        pe2 = pe.clone()
                        if isinstance(pe2.text_embeds, (list, tuple)):
                            new = []
                            for t in pe2.text_embeds:
                                if mode == 'zero':
                                    new.append(torch.zeros_like(t))
                                else:
                                    new.append(torch.randn_like(t))
                            pe2.text_embeds = new
                        else:
                            if mode == 'zero':
                                pe2.text_embeds = torch.zeros_like(pe2.text_embeds)
                            else:
                                pe2.text_embeds = torch.randn_like(pe2.text_embeds)
                        return pe2

                ablated_embeds = _make_ablated(conditional_embeds, 'zero')

            for _rep in range(samples):
                # compute per-element MSE (no reduction) for original conditioning
                loss_per_element = (noise_pred.float() - target.float()) ** 2
                per_sample = per_sample_from_loss_tensor(loss_per_element)
                if sum_per_sample is None:
                    sum_per_sample = per_sample
                else:
                    sum_per_sample = sum_per_sample + per_sample

                # capture scalar per-sample loss for each image in the batch
                per_sample_cpu = per_sample.detach().cpu()
                for ii in range(per_sample_cpu.shape[0]):
                    try:
                        samples_list_local[ii].append(float(per_sample_cpu[ii].item()))
                    except Exception:
                        # be defensive; ignore failures and continue
                        pass

                # if ablation-compare is requested, compute ablated prediction and losses
                if caption_ablation_compare and ablated_embeds is not None:
                    try:
                        try:
                            out = sd.predict_noise(noisy, text_embeddings=ablated_embeds, timestep=timesteps, batch=batch, return_conditional_pred=True)
                            if isinstance(out, tuple) and len(out) == 2:
                                noise_pred_abl, _ = out
                            else:
                                noise_pred_abl = out
                        except TypeError:
                            out = sd.predict_noise(noisy, prompt_embeds=ablated_embeds, timestep=timesteps, batch=batch, return_conditional_pred=True)
                            if isinstance(out, tuple) and len(out) == 2:
                                noise_pred_abl, _ = out
                            else:
                                noise_pred_abl = out
                    except TypeError:
                        try:
                            noise_pred_abl = sd.predict_noise(noisy, text_embeddings=ablated_embeds, timestep=timesteps, batch=batch)
                        except TypeError:
                            noise_pred_abl = sd.predict_noise(noisy, prompt_embeds=ablated_embeds, timestep=timesteps, batch=batch)

                    try:
                        loss_per_element_abl = (noise_pred_abl.float() - target.float()) ** 2
                        per_sample_abl = per_sample_from_loss_tensor(loss_per_element_abl)
                        if sum_per_sample_abl is None:
                            sum_per_sample_abl = per_sample_abl
                        else:
                            sum_per_sample_abl = sum_per_sample_abl + per_sample_abl

                        per_sample_abl_cpu = per_sample_abl.detach().cpu()
                        # compute per-sample delta (blank - caption)
                        for ii in range(per_sample_abl_cpu.shape[0]):
                            try:
                                a = float(per_sample_abl_cpu[ii].item())
                                m = float(per_sample_cpu[ii].item())
                                samples_list_local_abl[ii].append(a)
                                samples_list_local_delta[ii].append(a - m)
                            except Exception:
                                pass
                    except Exception:
                        # swallow ablation failures
                        pass

                # Debug: report noise magnitude and scheduler-derived std for this repetition
                try:
                    if debug_noise:
                        # per-sample raw noise std (computed on the current noise tensor)
                        try:
                            noise_std_per_sample = noise.view(noise.shape[0], -1).std(dim=1).detach().cpu().numpy().tolist()
                        except Exception:
                            noise_std_per_sample = None

                        # scheduler-derived std (if alphas_cumprod available), fallback to fixed std s when used
                        sched = getattr(sd, 'noise_scheduler', None)
                        sched_stds = None
                        try:
                            if fixed_noise_std is not None:
                                sched_stds = [float(fixed_noise_std)] * latents.shape[0]
                            elif sched is not None and hasattr(sched, 'alphas_cumprod'):
                                ac = sched.alphas_cumprod
                                import numpy as _np
                                if hasattr(ac, 'detach'):
                                    ac_cpu = ac.detach().cpu().numpy()
                                else:
                                    ac_cpu = _np.array(ac)
                                timesteps_cpu = timesteps.detach().cpu().numpy() if isinstance(timesteps, torch.Tensor) else _np.array(timesteps)
                                alpha_vals = ac_cpu[timesteps_cpu]
                                sched_stds = (_np.sqrt(1.0 - alpha_vals)).tolist()
                        except Exception:
                            sched_stds = None

                        # show file paths if available for context
                        paths = [getattr(fi, 'path', None) for fi in batch.file_items]
                        try:
                            tlist = None if timesteps is None else (timesteps.detach().cpu().tolist() if isinstance(timesteps, torch.Tensor) else list(timesteps))
                        except Exception:
                            tlist = None
                        print(f"[EVAL-DEBUG] rep={_rep}, timestep_type={debug_noise_timestep_type}, paths={paths}, timesteps={tlist}, noise_std={noise_std_per_sample}, sched_std={sched_stds}")
                except Exception:
                    # swallow debug failures so evaluation doesn't stop
                    pass

                # resample noise and timesteps for next repetition if any
                if _rep != samples - 1:
                    noise = torch.randn_like(latents).to(sd.device_torch)
                    if fixed_noise_std is not None:
                        # Use the same fixed-noise mixing for subsequent repetitions
                        noisy = latents * lat_coeff + noise * s
                        # keep timesteps as the previously chosen constant tensor
                    else:
                        timesteps = sample_timesteps(bs)
                        noisy = sd.noise_scheduler.add_noise(latents, noise, timesteps)
                    try:
                        noise_pred = sd.predict_noise(noisy, text_embeddings=conditional_embeds, timestep=timesteps, batch=batch)
                    except TypeError:
                        noise_pred = sd.predict_noise(noisy, prompt_embeds=conditional_embeds, timestep=timesteps, batch=batch)
            # average over samples
            avg_per_sample = (sum_per_sample / float(samples)).detach().cpu()

            # if ablation-compare was used, compute averages for ablated predictions and
            # set the recorded loss to the mean delta (blank - caption). Also include
            # supporting fields for diagnostics.
            if caption_ablation_compare and sum_per_sample_abl is not None:
                avg_per_sample_abl = (sum_per_sample_abl / float(samples)).detach().cpu()
                # compute delta per-sample (abl - orig)
                try:
                    avg_delta = (avg_per_sample_abl - avg_per_sample).detach().cpu()
                except Exception:
                    # fallback: convert to lists
                    try:
                        orig_list = avg_per_sample.tolist() if hasattr(avg_per_sample, 'tolist') else list(avg_per_sample)
                        abl_list = avg_per_sample_abl.tolist() if hasattr(avg_per_sample_abl, 'tolist') else list(avg_per_sample_abl)
                        avg_delta = [ (abl_list[i] - orig_list[i]) for i in range(min(len(orig_list), len(abl_list))) ]
                    except Exception:
                        avg_delta = None
            else:
                avg_per_sample_abl = None
                avg_delta = None

            # NOTE: per-batch normalization was removed here to ensure raw per-sample
            # losses are preserved into the post-collection `raw_report`. Global
            # normalization (across all examples) is handled by `evaluate_dataset`.

            # build records (include per-image sample min/max from the repeated stochastic samples)
            losses = avg_per_sample.tolist() if hasattr(avg_per_sample, 'tolist') else list(avg_per_sample)
            losses_abl = avg_per_sample_abl.tolist() if (avg_per_sample_abl is not None and hasattr(avg_per_sample_abl, 'tolist')) else (list(avg_per_sample_abl) if avg_per_sample_abl is not None else None)
            deltas = avg_delta.tolist() if (avg_delta is not None and hasattr(avg_delta, 'tolist')) else (list(avg_delta) if avg_delta is not None else None)

            for i, fi in enumerate(batch.file_items):
                sl = samples_list_local[i] if i < len(samples_list_local) else None
                if sl and isinstance(sl, list) and len(sl) > 0:
                    min_s = float(min(sl))
                    max_s = float(max(sl))
                else:
                    min_s = float(losses[i]) if i < len(losses) else None
                    max_s = float(losses[i]) if i < len(losses) else None

                # ablation-specific min/max (from deltas) if present
                if deltas is not None and i < len(deltas):
                    min_delta = float(min(samples_list_local_delta[i])) if len(samples_list_local_delta[i]) > 0 else float(deltas[i])
                    max_delta = float(max(samples_list_local_delta[i])) if len(samples_list_local_delta[i]) > 0 else float(deltas[i])
                    loss_value = float(deltas[i])
                    loss_with_caption = float(losses[i]) if i < len(losses) else None
                    loss_with_blank = float(losses_abl[i]) if (losses_abl is not None and i < len(losses_abl)) else None
                else:
                    min_delta = None
                    max_delta = None
                    loss_value = float(losses[i]) if i < len(losses) else None
                    # even when ablation is not used, record the non-ablated/original loss so
                    # consumers can choose which metric to display in the UI
                    loss_with_caption = float(losses[i]) if i < len(losses) else None
                    loss_with_blank = None

                out_item = {
                    'path': getattr(fi, 'path', None),
                    'dataset': getattr(fi.dataset_config, 'dataset_path', None) or getattr(fi.dataset_config, 'folder_path', None) if getattr(fi, 'dataset_config', None) is not None else None,
                    'caption': getattr(fi, 'raw_caption', '') or '',
                    'loss': loss_value,
                    'loss_with_caption': loss_with_caption,
                    'min_loss': min_delta if min_delta is not None else min_s,
                    'max_loss': max_delta if max_delta is not None else max_s,
                }
                # attach additional diagnostics when ablation compare was used
                if caption_ablation_compare:
                    out_item['loss_with_blank'] = loss_with_blank
                    out_item['ablation_delta'] = loss_value

                results.append(out_item)
        return results

    # Also enforce dropout=0.0 eagerly at compute_fn creation time so callers observe the change immediately
    try:
        import torch.nn as _nn
        _changed = False
        for root in (getattr(sd, 'unet', None), getattr(sd, 'text_encoder', None), getattr(sd, 'vae', None)):
            if root is None:
                continue
            for m in root.modules():
                if isinstance(m, _nn.Dropout):
                    try:
                        if getattr(m, 'p', None) is not None and float(m.p) != 0.0:
                            m.p = 0.0
                            _changed = True
                    except Exception:
                        pass
        if _changed:
            print('Enforced dropout=0.0 for evaluation (creation-time)')
    except Exception:
        pass

    return compute_fn


def build_dataloader_for_folder(folder_path: str, batch_size: int, sd: StableDiffusion, eval_resolution: int | None = None):
    """Create a dataloader for a single dataset folder using DatasetConfig.

    If `eval_resolution` is provided it overrides the dataset's configured `resolution`
    for evaluation runs (useful to reduce pixel resolution while preserving shape information).
    """
    if eval_resolution is not None:
        ds = DatasetConfig(folder_path=folder_path, resolution=eval_resolution)
    else:
        ds = DatasetConfig(folder_path=folder_path)
    return get_dataloader_from_datasets([ds], batch_size=batch_size, sd=sd)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a dataset and write JSON report(s)")
    parser.add_argument('--dataset-path', required=True, help='Folder or dataset config to evaluate')
    parser.add_argument('--model', required=False, help='Model name_or_path to load (ModelConfig.name_or_path)')
    parser.add_argument('--model-config-file', required=False, help='Path to JSON file with ModelConfig kwargs')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='bf16')
    parser.add_argument('--sample-fraction', type=float, default=1.0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--out-dir', default=None, help='Directory to write evaluation JSON; defaults to dataset folder')
    parser.add_argument('--job-name', default='dataset_eval')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--apply-inference-lora', action='store_true', help='If set, apply assistant/inference LoRA from model_config in-memory for evaluation')

    # Loss reporting options
    # Normalization is enabled by default; pass --no-normalize-loss to disable.
    parser.set_defaults(normalize_loss=True)
    parser.add_argument('--no-normalize-loss', dest='normalize_loss', action='store_false', help='Disable normalization of per-sample losses (default: enabled)')

    # Number of stochastic samples (noise/timesteps) to average per image
    parser.add_argument('--samples-per-image', type=int, default=8, help='Number of independent stochastic forward passes per image to average (default: 8)')
    parser.add_argument('--debug-noise', action='store_true', help='If set, print debug info about noise std and scheduler std per repetition')
    parser.add_argument('--timestep-type', choices=['uniform','one_step','cubic_early','cubic_late','sigmoid'], default='sigmoid', help='Timestep sampling strategy to mirror training. "uniform" samples uniformly, "one_step" uses timestep 0, "cubic_early" biases toward early timesteps, "cubic_late" biases toward late timesteps, "sigmoid" matches trainer default sigmoid schedule.')
    parser.add_argument('--fixed-noise-std', type=float, default=0.6, help='If set (0.0-1.0), use this fixed noise std instead of sampling timesteps. Defaults to 0.6 (60% noise magnitude) to reduce evaluation variance when comparing captions.')
    parser.add_argument('--debug-captions', action='store_true', help='If set, print debug info about prompt embeddings (shapes, norms) and whether classifier-free guidance is active for each batch')
    parser.add_argument('--caption-ablation', choices=['none','zero','random'], default='none', help='If set, replace prompt embeddings per-batch with zeros or random noise to test whether captions affect model predictions')
    parser.add_argument('--caption-ablation-compare', action='store_true', help='If set, run each evaluation twice (real vs blank) and store the difference as the recorded loss')
    parser.add_argument('--log-conditioning', action='store_true', help='If set, emit per-batch embedding statistics (mean/std) and, if caption-ablation used, the MSE between original and ablated noise predictions')

    # Evaluation dataset resolution (optional): override dataset resolution for evaluation (default: 256)
    parser.add_argument('--eval-resolution', type=int, default=256, help='If set, override dataset resolution for evaluation crops and resizing (default: 256)')

    args = parser.parse_args(argv)

    # normalize device string: accept 'gpu' from UI and map to 'cuda' when available
    device = args.device
    dev_lower = str(device).lower()
    if dev_lower == 'gpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif dev_lower.startswith('gpu:'):
        # map 'gpu:0' -> 'cuda:0'
        device = 'cuda' + dev_lower[4:]
    elif dev_lower == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # report resolved device
    print(f"Resolved device: {device}")

    # load model config from file if provided (sanitize / restrict options)
    model_cfg = None
    if args.model_config_file:
        try:
            with open(args.model_config_file, 'r', encoding='utf-8') as f:
                model_cfg = json.load(f)
        except Exception as e:
            print(f"Failed to read model config file {args.model_config_file}: {e}")
            return 2

    if model_cfg is not None:
        # For evaluation we do not want to create or attach LoRA/training adapters unless
        # explicitly requested via --apply-inference-lora. To avoid accidental side-effects
        # strip known LoRA fields when apply_inference_lora is False.
        if not args.apply_inference_lora:
            for k in ['assistant_lora_path', 'lora_path', 'inference_lora_path']:
                if k in model_cfg:
                    print(f"Stripping model_config field {k} for eval (apply_inference_lora is False)")
                    model_cfg.pop(k, None)

        print(f"Loading model config (inference-safe) from {args.model_config_file} on {device}... apply_lora={args.apply_inference_lora}")
        try:
            # model_utils will sanitize and load in inference mode
            sd = load_model_for_inference(model_cfg, device=device, dtype=args.dtype, apply_lora=args.apply_inference_lora)
        except Exception as e:
            print(f"Failed to load model from model_config file: {e}")
            print("This likely means the model assets are not available or require authentication on Hugging Face.")
            print("If the model is hosted privately, set the HF_TOKEN environment variable with an access token or provide a reachable local model path in the model config.")
            return 3
    else:
        if not args.model:
            print("Error: --model or --model-config-file must be provided")
            return 2
        print(f"Loading model {args.model} on {device} (inference-safe)... apply_lora={args.apply_inference_lora}")
        try:
            sd = load_model_for_inference(args.model, device=device, dtype=args.dtype, apply_lora=args.apply_inference_lora)
        except Exception as e:
            print(f"Failed to load model {args.model}: {e}")
            print("This likely means the model id was not found on Hugging Face or is private. Confirm the model name is correct or set HF_TOKEN to access private models.")
            return 3

    print(f"Building dataloader for {args.dataset_path} (batch_size={args.batch_size}, eval_resolution={args.eval_resolution})...")
    dataloader = build_dataloader_for_folder(args.dataset_path, args.batch_size, sd, eval_resolution=args.eval_resolution)

    compute_fn = make_compute_fn_for_sd(
        sd,
        device=device,
        normalize_loss=args.normalize_loss,
        samples_per_image=args.samples_per_image,
        debug_noise=args.debug_noise,
        debug_noise_timestep_type=args.timestep_type,
        fixed_noise_std=args.fixed_noise_std,
        debug_captions=args.debug_captions,
        caption_ablation=args.caption_ablation,
        caption_ablation_compare=args.caption_ablation_compare,
        log_conditioning=args.log_conditioning,
    )

    out_dir = args.out_dir or args.dataset_path
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            print(f"Could not create out_dir {out_dir}: {e}")
            return 2

    # Prefer a model-based filename so repeated evaluations of the same model overwrite
    # the previous report. If a model_config file was provided prefer its name_or_path.
    model_name_for_file = None
    try:
        if args.model_config_file:
            with open(args.model_config_file, 'r', encoding='utf-8') as f:
                mc = json.load(f)
                model_name_for_file = mc.get('name_or_path')
        if not model_name_for_file and args.model:
            model_name_for_file = args.model
    except Exception:
        model_name_for_file = model_name_for_file or args.model

    def _sanitize_name(n: str) -> str:
        if not n:
            return 'model'
        base = n.split('/')[-1].split('\\')[-1]
        import re
        s = re.sub(r"[^A-Za-z0-9_.-]", '_', base)
        return s

    out_json = None
    if model_name_for_file:
        filename = f"{_sanitize_name(model_name_for_file)}.json"
        out_json = os.path.join(out_dir, filename)
        print(f"Writing evaluation JSON to {out_json} (model-based name, will overwrite if exists)")
    else:
        print("No model name available for file naming; using job-based filename")

    print("Running evaluation. This may take a while...")

    # Build eval_config so the resulting JSON includes the actual model used.
    eval_conf = {'model': model_name_for_file or args.model}
    if model_cfg is not None:
        # include minimal, non-sensitive model_config metadata for the report
        # (avoid embedding secrets or large fields; name_or_path and arch are useful)
        try:
            eval_conf['model_config'] = {
                'name_or_path': model_cfg.get('name_or_path'),
                'arch': model_cfg.get('arch') if isinstance(model_cfg, dict) else None,
            }
        except Exception:
            pass

    # Include noise and sampling configuration so evaluation artifacts are self-describing.
    try:
        num_train_steps = None
        try:
            num_train_steps = int(sd.noise_scheduler.config.num_train_timesteps)
        except Exception:
            num_train_steps = None
        eval_conf['noise'] = {
            'fixed_noise_std': args.fixed_noise_std,
            'timestep_type': args.timestep_type,
            'samples_per_image': args.samples_per_image,
            'normalize_loss': bool(args.normalize_loss),
            'num_train_timesteps': num_train_steps,
        }
    except Exception:
        # best-effort: do not fail evaluation if metadata collection fails
        pass

    res = run_dataset_evaluation(
        compute_fn,
        dataloader,
        sample_fraction=args.sample_fraction,
        max_samples=args.max_samples,
        out_dir=out_dir,
        job_name=args.job_name,
        step=args.step,
        eval_config=eval_conf,
        normalize_loss=args.normalize_loss,
        **({'out_json': out_json} if out_json is not None else {}),
    )

    # print a concise summary
    aggregates = res.get('aggregates', {})
    global_summary = aggregates.get('global', {})
    print(f"Total examples: {global_summary.get('count', 0)}, mean loss: {global_summary.get('mean', 0.0):.6f}")
    print(f"Flagged captions: {len(res.get('flagged', {}).get('flagged', []))}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
