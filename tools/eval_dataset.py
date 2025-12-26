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


def make_compute_fn_for_sd(sd: StableDiffusion, device: str = "cpu", normalize_loss: bool = True, samples_per_image: int = 4, debug_noise: bool = False, debug_noise_timestep_type: str = 'sigmoid') -> Callable[[Any], List[Dict[str, Any]]]:
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

            timesteps = sample_timesteps(bs)

            noisy = sd.noise_scheduler.add_noise(latents, noise, timesteps)

            # obtain model prediction
            try:
                noise_pred = sd.predict_noise(noisy, text_embeddings=conditional_embeds, timestep=timesteps, batch=batch)
            except TypeError:
                # older signatures may expect 'prompt_embeds' naming
                noise_pred = sd.predict_noise(noisy, prompt_embeds=conditional_embeds, timestep=timesteps, batch=batch)

            # determine target for loss
            try:
                target = sd.get_loss_target(noise=noise, batch=batch, timesteps=timesteps)
            except Exception:
                target = noise

            # Optionally run multiple stochastic samples per image and average
            samples = max(1, int(samples_per_image))
            sum_per_sample = None
            # prepare per-image lists to capture raw sample losses
            samples_list_local = [ [] for _ in range(bs) ]
            for _rep in range(samples):
                # compute per-element MSE (no reduction)
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

                # Debug: report noise magnitude and scheduler-derived std for this repetition
                try:
                    if debug_noise:
                        # per-sample raw noise std (computed on the current noise tensor)
                        try:
                            noise_std_per_sample = noise.view(noise.shape[0], -1).std(dim=1).detach().cpu().numpy().tolist()
                        except Exception:
                            noise_std_per_sample = None

                        # scheduler-derived std (if alphas_cumprod available)
                        sched = getattr(sd, 'noise_scheduler', None)
                        sched_stds = None
                        try:
                            if sched is not None and hasattr(sched, 'alphas_cumprod'):
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
                            tlist = timesteps.detach().cpu().tolist() if isinstance(timesteps, torch.Tensor) else list(timesteps)
                        except Exception:
                            tlist = None
                        print(f"[EVAL-DEBUG] rep={_rep}, timestep_type={debug_noise_timestep_type}, paths={paths}, timesteps={tlist}, noise_std={noise_std_per_sample}, sched_std={sched_stds}")
                except Exception:
                    # swallow debug failures so evaluation doesn't stop
                    pass

                # resample noise and timesteps for next repetition if any
                if _rep != samples - 1:
                    noise = torch.randn_like(latents).to(sd.device_torch)
                    timesteps = sample_timesteps(bs)
                    noisy = sd.noise_scheduler.add_noise(latents, noise, timesteps)
                    try:
                        noise_pred = sd.predict_noise(noisy, text_embeddings=conditional_embeds, timestep=timesteps, batch=batch)
                    except TypeError:
                        noise_pred = sd.predict_noise(noisy, prompt_embeds=conditional_embeds, timestep=timesteps, batch=batch)
            # average over samples
            avg_per_sample = (sum_per_sample / float(samples)).detach().cpu()

            # NOTE: per-batch normalization was removed here to ensure raw per-sample
            # losses are preserved into the post-collection `raw_report`. Global
            # normalization (across all examples) is handled by `evaluate_dataset`.
            # (Left intentionally empty)

            # build records (include per-image sample min/max from the repeated stochastic samples)
            losses = avg_per_sample.tolist() if hasattr(avg_per_sample, 'tolist') else list(avg_per_sample)
            for i, fi in enumerate(batch.file_items):
                sl = samples_list_local[i] if i < len(samples_list_local) else None
                if sl and isinstance(sl, list) and len(sl) > 0:
                    min_s = float(min(sl))
                    max_s = float(max(sl))
                else:
                    min_s = float(losses[i]) if i < len(losses) else None
                    max_s = float(losses[i]) if i < len(losses) else None

                results.append({
                    'path': getattr(fi, 'path', None),
                    'dataset': getattr(fi.dataset_config, 'dataset_path', None) or getattr(fi.dataset_config, 'folder_path', None) if getattr(fi, 'dataset_config', None) is not None else None,
                    'caption': getattr(fi, 'raw_caption', '') or '',
                    'loss': float(losses[i]) if i < len(losses) else None,
                    'min_loss': min_s,
                    'max_loss': max_s,
                })
        return results

    return compute_fn


def build_dataloader_for_folder(folder_path: str, batch_size: int, sd: StableDiffusion):
    """Create a dataloader for a single dataset folder using DatasetConfig."""
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
    parser.add_argument('--samples-per-image', type=int, default=4, help='Number of independent stochastic forward passes per image to average (default: 4)')
    parser.add_argument('--debug-noise', action='store_true', help='If set, print debug info about noise std and scheduler std per repetition')
    parser.add_argument('--timestep-type', choices=['uniform','one_step','cubic_early','cubic_late','sigmoid'], default='sigmoid', help='Timestep sampling strategy to mirror training. "uniform" samples uniformly, "one_step" uses timestep 0, "cubic_early" biases toward early timesteps, "cubic_late" biases toward late timesteps, "sigmoid" matches trainer default sigmoid schedule.')

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

    print(f"Building dataloader for {args.dataset_path} (batch_size={args.batch_size})...")
    dataloader = build_dataloader_for_folder(args.dataset_path, args.batch_size, sd)

    compute_fn = make_compute_fn_for_sd(sd, device=device, normalize_loss=args.normalize_loss, samples_per_image=args.samples_per_image, debug_noise=args.debug_noise, debug_noise_timestep_type=args.timestep_type)

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
