"""Quick evaluation simulator for datasets using a Dummy StableDiffusion-like object.

This script uses the same `make_compute_fn_for_sd` and `run_dataset_evaluation` from
`tools.eval_dataset` to generate a report quickly without loading a real model.
It is useful for validating JSON format (raw/norm/min/max) and UI behavior.
"""
import os
import json
from typing import List

import torch

from toolkit.util.loss_utils import run_dataset_evaluation

# Local copy of make_compute_fn_for_sd to avoid importing tools.eval_dataset (which pulls heavier deps)
from typing import Callable, Any, List, Dict

def make_compute_fn_for_sd_local(sd, device: str = "cpu", normalize_loss: bool = True, samples_per_image: int = 4) -> Callable[[Any], List[Dict[str, Any]]]:
    def compute_fn(batch) -> List[Dict[str, Any]]:
        results = []
        with torch.no_grad():
            if getattr(batch, 'prompt_embeds', None) is not None:
                conditional_embeds = batch.prompt_embeds
            else:
                captions = [getattr(fi, 'raw_caption', '') or '' for fi in batch.file_items]
                try:
                    conditional_embeds = sd.encode_prompt(captions)
                except Exception:
                    conditional_embeds = sd.encode_prompt([""] * len(captions))

            if getattr(batch, 'latents', None) is not None:
                latents = batch.latents.to(sd.device_torch)
            else:
                return results

            bs = latents.shape[0]
            noise = torch.randn_like(latents).to(sd.device_torch)
            try:
                max_steps = int(sd.noise_scheduler.config.num_train_timesteps)
            except Exception:
                max_steps = 1000
            timesteps = torch.randint(0, max_steps, (bs,), device=sd.device_torch, dtype=torch.long)

            noisy = sd.noise_scheduler.add_noise(latents, noise, timesteps)

            try:
                noise_pred = sd.predict_noise(noisy, text_embeddings=conditional_embeds, timestep=timesteps, batch=batch)
            except TypeError:
                noise_pred = sd.predict_noise(noisy, prompt_embeds=conditional_embeds, timestep=timesteps, batch=batch)

            try:
                target = sd.get_loss_target(noise=noise, batch=batch, timesteps=timesteps)
            except Exception:
                target = noise

            samples = max(1, int(samples_per_image))
            sum_per_sample = None
            samples_list_local = [ [] for _ in range(bs) ]
            for _rep in range(samples):
                loss_per_element = (noise_pred.float() - target.float()) ** 2
                per_sample = ((loss_per_element.view(loss_per_element.shape[0], -1)).mean(dim=1))
                if sum_per_sample is None:
                    sum_per_sample = per_sample
                else:
                    sum_per_sample = sum_per_sample + per_sample

                per_sample_cpu = per_sample.detach().cpu()
                for ii in range(per_sample_cpu.shape[0]):
                    try:
                        samples_list_local[ii].append(float(per_sample_cpu[ii].item()))
                    except Exception:
                        pass

                if _rep != samples - 1:
                    noise = torch.randn_like(latents).to(sd.device_torch)
                    try:
                        max_steps = int(sd.noise_scheduler.config.num_train_timesteps)
                    except Exception:
                        max_steps = 1000
                    timesteps = torch.randint(0, max_steps, (bs,), device=sd.device_torch, dtype=torch.long)
                    noisy = sd.noise_scheduler.add_noise(latents, noise, timesteps)
                    try:
                        noise_pred = sd.predict_noise(noisy, text_embeddings=conditional_embeds, timestep=timesteps, batch=batch)
                    except TypeError:
                        noise_pred = sd.predict_noise(noisy, prompt_embeds=conditional_embeds, timestep=timesteps, batch=batch)

            avg_per_sample = (sum_per_sample / float(samples)).detach().cpu()
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

DATASET = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'jinx_references')
OUT_JSON = os.path.join(DATASET, 'quick_sim.json')

class DummyNoiseScheduler:
    def __init__(self, num_train_timesteps=1000):
        self.config = type('C', (), {'num_train_timesteps': num_train_timesteps})
    def add_noise(self, latents, noise, timesteps):
        return latents + noise

class DummySD:
    def __init__(self, device='cpu'):
        self.device_torch = torch.device(device)
        self.device = device
        self.noise_scheduler = DummyNoiseScheduler(num_train_timesteps=10)
        # VAE dtypes / devices referenced by the compute_fn
        self.vae_device_torch = torch.device(device)
        self.vae_torch_dtype = torch.float32

    def encode_prompt(self, prompts: List[str]):
        return {'prompts': prompts}

    def encode_images(self, images, device=None, dtype=None):
        b = len(images)
        return torch.randn(b, 4, 16, 16)

    def predict_noise(self, noisy, text_embeddings=None, timestep=None, batch=None, **kwargs):
        # Predict varying noise so losses are non-zero and vary per sample
        return torch.randn_like(noisy) * 0.5

    def get_loss_target(self, noise=None, batch=None, timesteps=None):
        return noise

# Build the compute_fn and a simple dataloader generator
sd = DummySD(device='cpu')
compute_fn = make_compute_fn_for_sd_local(sd, device='cpu', normalize_loss=True, samples_per_image=4)

# Build batches: we will synthesize latents rather than decoding images for speed
file_list = [f for f in os.listdir(DATASET) if os.path.isfile(os.path.join(DATASET, f)) and f.lower().endswith('.png')]

class FileItem:
    def __init__(self, path):
        self.path = os.path.join(DATASET, path)
        self.raw_caption = ''
        self.dataset_config = type('D', (), {'dataset_path': DATASET})

class Batch:
    def __init__(self, file_items, latents):
        self.file_items = file_items
        self.latents = latents
        self.tensor = None
        self.prompt_embeds = None

batches = []
for fname in file_list:
    # create a single-item batch with a random latent
    lat = torch.randn(1, 4, 16, 16)
    batches.append(Batch([FileItem(fname)], lat))

# Run the evaluation and write JSON
res = run_dataset_evaluation(compute_fn, batches, sample_fraction=1.0, max_samples=None, out_json=OUT_JSON, normalize_loss=True, eval_config={'model': 'quick_sim'})

print('Wrote simulated eval JSON to', OUT_JSON)
with open(OUT_JSON, 'r', encoding='utf-8') as f:
    j = json.load(f)
print(json.dumps(j['datasets'], indent=2)[:2000])
