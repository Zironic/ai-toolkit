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


def build_sd_model(name_or_path: str, device: str = "cpu", dtype: str = "float32") -> StableDiffusion:
    """Create and load a StableDiffusion wrapper given a model path/name."""
    model_config = ModelConfig(name_or_path=name_or_path)
    sd = StableDiffusion(device=device, model_config=model_config, dtype=dtype)
    sd.load_model()
    return sd


def make_compute_fn_for_sd(sd: StableDiffusion, device: str = "cpu") -> Callable[[Any], List[Dict[str, Any]]]:
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
                latents = sd.encode_images(batch.tensor.to(sd.device_torch))
            else:
                # nothing to evaluate
                return results

            bs = latents.shape[0]
            # sample noise and timesteps
            noise = torch.randn_like(latents).to(sd.device_torch)
            # choose timesteps uniformly across scheduler steps
            try:
                max_steps = int(sd.noise_scheduler.config.num_train_timesteps)
            except Exception:
                # fallback to 1000
                max_steps = 1000
            timesteps = torch.randint(0, max_steps, (bs,), device=sd.device_torch, dtype=torch.long)

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

            # compute per-element MSE (no reduction)
            loss_per_element = (noise_pred.float() - target.float()) ** 2

            # reduce to per-sample scalars
            per_sample = per_sample_from_loss_tensor(loss_per_element)

            # build records
            losses = per_sample.tolist() if hasattr(per_sample, 'tolist') else list(per_sample)
            for i, fi in enumerate(batch.file_items):
                results.append({
                    'path': getattr(fi, 'path', None),
                    'dataset': getattr(fi.dataset_config, 'dataset_path', None) or getattr(fi.dataset_config, 'folder_path', None) if getattr(fi, 'dataset_config', None) is not None else None,
                    'caption': getattr(fi, 'raw_caption', '') or '',
                    'loss': float(losses[i]) if i < len(losses) else None,
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
    parser.add_argument('--model', required=True, help='Model name_or_path to load (ModelConfig.name_or_path)')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('--sample-fraction', type=float, default=1.0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--out-dir', default=None, help='Directory to write evaluation JSON; defaults to dataset folder')
    parser.add_argument('--job-name', default='dataset_eval')
    parser.add_argument('--step', type=int, default=0)

    args = parser.parse_args(argv)

    device = args.device

    print(f"Loading model {args.model} on {device}...")
    sd = build_sd_model(args.model, device=device, dtype=args.dtype)

    print(f"Building dataloader for {args.dataset_path} (batch_size={args.batch_size})...")
    dataloader = build_dataloader_for_folder(args.dataset_path, args.batch_size, sd)

    compute_fn = make_compute_fn_for_sd(sd, device=device)

    out_dir = args.out_dir or args.dataset_path
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            print(f"Could not create out_dir {out_dir}: {e}")
            return 2

    print("Running evaluation. This may take a while...")
    res = run_dataset_evaluation(
        compute_fn,
        dataloader,
        sample_fraction=args.sample_fraction,
        max_samples=args.max_samples,
        out_dir=out_dir,
        job_name=args.job_name,
        step=args.step,
        eval_config={'model': args.model},
    )

    # print a concise summary
    aggregates = res.get('aggregates', {})
    global_summary = aggregates.get('global', {})
    print(f"Total examples: {global_summary.get('count', 0)}, mean loss: {global_summary.get('mean', 0.0):.6f}")
    print(f"Flagged captions: {len(res.get('flagged', {}).get('flagged', []))}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
