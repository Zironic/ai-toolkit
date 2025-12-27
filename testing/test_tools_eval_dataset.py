import torch

from tools.eval_dataset import make_compute_fn_for_sd
from toolkit.util.loss_utils import evaluate_dataset


class DummyNoiseScheduler:
    def __init__(self, num_train_timesteps=1000):
        self.config = type('C', (), {'num_train_timesteps': num_train_timesteps})

    def add_noise(self, latents, noise, timesteps):
        # simple elementwise add (timesteps ignored)
        return latents + noise


class DummySD:
    def __init__(self, device='cpu'):
        self.device_torch = torch.device(device)
        self.device = device
        self.noise_scheduler = DummyNoiseScheduler(num_train_timesteps=10)

    def encode_prompt(self, prompts):
        # return a dummy token indicating number of prompts
        return {'prompts': prompts}

    def encode_images(self, images):
        # collapse an image tensor into a small latent
        # images: (B,C,H,W) -> return (B,4,16,16) latents
        b = images.shape[0]
        return torch.randn(b, 4, 16, 16)

    def predict_noise(self, noisy, text_embeddings=None, timestep=None, batch=None, **kwargs):
        # pretend the model predicts zeros
        return torch.zeros_like(noisy)

    def get_loss_target(self, noise=None, batch=None, timesteps=None):
        # target is the noise (so mse will be noise^2)
        return noise


# Module-scoped SDAblation used by multiple tests; placed here so it is
# accessible across test functions (previously it was defined inside a test
# which made it unavailable to other tests that depended on it).
class SDAblation(DummySD):
    def encode_prompt(self, prompts):
        # Return zero embeddings for empty prompt to emulate model's empty encoding
        if len(prompts) == 1 and prompts[0] == '':
            return type('P', (), {'text_embeds': torch.zeros((1, 4))})
        # otherwise return non-zero to indicate 'real' caption
        return type('P', (), {'text_embeds': torch.ones((len(prompts), 4))})

    def predict_noise(self, noisy, text_embeddings=None, timestep=None, batch=None, **kwargs):
        te = text_embeddings
        # crude check: if embeddings are zero tensor, return ones; else zeros
        if hasattr(te, 'text_embeds'):
            t = te.text_embeds
            if isinstance(t, torch.Tensor):
                if torch.all(t == 0):
                    return torch.ones_like(noisy)
                else:
                    return torch.zeros_like(noisy)
            elif isinstance(t, (list, tuple)):
                # if first tensor is zero
                if torch.all(t[0] == 0):
                    return torch.ones_like(noisy)
                else:
                    return torch.zeros_like(noisy)
        return torch.zeros_like(noisy)

    def get_loss_target(self, noise=None, batch=None, timesteps=None):
        # return zeros so loss equals prediction^2
        return torch.zeros_like(noise)


class DummyFile:
    def __init__(self, path, caption):
        self.path = path
        self.raw_caption = caption
        self.dataset_config = type('D', (), {'dataset_path': 'dummy_dataset'})


class DummyBatch:
    def __init__(self, file_items, latents=None):
        self.file_items = file_items
        if latents is not None:
            self.latents = latents
        else:
            self.latents = None
        self.tensor = None
        self.prompt_embeds = None


def test_compute_fn_dummy_sd():
    sd = DummySD()
    compute_fn = make_compute_fn_for_sd(sd)

    # construct batch with 2 items and latents
    lat = torch.randn(2, 4, 16, 16)
    files = [DummyFile('/tmp/a.png', 'a caption'), DummyFile('/tmp/b.png', 'b caption')]
    batch = DummyBatch(files, latents=lat)

    entries = compute_fn(batch)
    assert isinstance(entries, list)
    assert len(entries) == 2
    for e in entries:
        assert 'path' in e and 'caption' in e and 'loss' in e
        assert e['dataset'] == 'dummy_dataset'
        assert isinstance(e['loss'], float)


def test_compute_fn_records_sample_min_max():
    # Use a deterministic random seed for reproducibility in test environment
    torch.manual_seed(12345)

    sd = DummySD()
    # make predict_noise return zeros so loss is purely noise^2 and varies per sample
    compute_fn = make_compute_fn_for_sd(sd, normalize_loss=False, samples_per_image=4)

    lat = torch.randn(2, 4, 16, 16)
    files = [DummyFile('/tmp/a.png', 'a caption'), DummyFile('/tmp/b.png', 'b caption')]
    batch = DummyBatch(files, latents=lat)

    entries = compute_fn(batch)
    assert len(entries) == 2
    for e in entries:
        # ensure we recorded min_loss and max_loss from the multiple stochastic samples
        assert 'min_loss' in e and 'max_loss' in e
        assert isinstance(e['min_loss'], float) and isinstance(e['max_loss'], float)
        # with multiple stochastic samples these should often differ
        # if they are equal, warn but don't strictly fail; otherwise assert min <= loss <= max
        assert e['min_loss'] <= e['loss'] <= e['max_loss']


def test_compute_fn_does_not_normalize_per_batch():
    """Ensure that compute_fn leaves raw per-sample losses unchanged so the
    outer `evaluate_dataset` can perform global normalization and `average_loss_raw`
    remains a true pre-normalized value."""
    class SDWithTargets(DummySD):
        def get_loss_target(self, noise=None, batch=None, timesteps=None):
            # produce deterministic targets: sample0 -> zeros, sample1 -> ones
            bs = noise.shape[0]
            out = torch.zeros_like(noise)
            if bs >= 2:
                out[1] = torch.ones_like(noise[1])
            return out

    sd2 = SDWithTargets()
    compute_fn = make_compute_fn_for_sd(sd2, normalize_loss=True, samples_per_image=1)

    lat = torch.randn(2, 4, 16, 16)
    files = [DummyFile('/tmp/a.png', 'a caption'), DummyFile('/tmp/b.png', 'b caption')]
    batch = DummyBatch(files, latents=lat)

    entries = compute_fn(batch)
    # With our deterministic targets and predict_noise() == 0, the per-sample
    # raw MSE should be 0.0 for sample 0 and 1.0 for sample 1. If compute_fn
    # performed per-batch normalization these values would be altered.
    assert len(entries) == 2
    assert abs(entries[0]['loss'] - 0.0) < 1e-6
    assert abs(entries[1]['loss'] - 1.0) < 1e-6


def test_compute_fn_ablation_compare():
    # predict_noise returns 0 for normal embeddings, 1 for ablated (zeros) embeddings
    class SDAblation(DummySD):
        def encode_prompt(self, prompts):
            # Return zero embeddings for empty prompt to emulate model's empty encoding
            if len(prompts) == 1 and prompts[0] == '':
                return type('P', (), {'text_embeds': torch.zeros((1, 4))})
            # otherwise return non-zero to indicate 'real' caption
            return type('P', (), {'text_embeds': torch.ones((len(prompts), 4))})

        def predict_noise(self, noisy, text_embeddings=None, timestep=None, batch=None, **kwargs):
            te = text_embeddings
            # crude check: if embeddings are zero tensor, return ones; else zeros
            if hasattr(te, 'text_embeds'):
                t = te.text_embeds
                if isinstance(t, torch.Tensor):
                    if torch.all(t == 0):
                        return torch.ones_like(noisy)
                    else:
                        return torch.zeros_like(noisy)
                elif isinstance(t, (list, tuple)):
                    # if first tensor is zero
                    if torch.all(t[0] == 0):
                        return torch.ones_like(noisy)
                    else:
                        return torch.zeros_like(noisy)
            return torch.zeros_like(noisy)

        def get_loss_target(self, noise=None, batch=None, timesteps=None):
            # return zeros so loss equals prediction^2
            return torch.zeros_like(noise)

    sd = SDAblation()
    compute_fn = make_compute_fn_for_sd(sd, normalize_loss=False, samples_per_image=2, caption_ablation_compare=True)

    lat = torch.randn(1, 4, 16, 16)
    files = [DummyFile('/tmp/a.png', 'a caption')]
    batch = DummyBatch(files, latents=lat)

    entries = compute_fn(batch)
    assert len(entries) == 1
    e = entries[0]
    # ablated prediction is ones so loss_with_blank ~ 1.0, loss_with_caption ~ 0.0, delta ~ 1.0
    assert 'loss_with_caption' in e and 'loss_with_blank' in e and 'ablation_delta' in e
    assert abs(e['loss_with_caption'] - 0.0) < 1e-6
    assert abs(e['loss_with_blank'] - 1.0) < 1e-6
    assert abs(e['ablation_delta'] - 1.0) < 1e-6


def test_compute_fn_always_records_loss_with_caption():
    sd = DummySD()
    compute_fn = make_compute_fn_for_sd(sd, normalize_loss=False, samples_per_image=1, caption_ablation_compare=False)

    lat = torch.randn(2, 4, 16, 16)
    files = [DummyFile('/tmp/a.png', 'a caption'), DummyFile('/tmp/b.png', 'b caption')]
    batch = DummyBatch(files, latents=lat)

    entries = compute_fn(batch)
    assert len(entries) == 2
    for e in entries:
        assert 'loss_with_caption' in e


import tempfile

def test_evaluate_dataset_average_loss_raw_for_ablation():
    # Use the SDAblation from above to produce an ablation delta and then run evaluate_dataset
    class SDAblationLocal(SDAblation):
        pass

    sd = SDAblationLocal()
    compute_fn = make_compute_fn_for_sd(sd, normalize_loss=False, samples_per_image=1, caption_ablation_compare=True)

    lat = torch.randn(2, 4, 16, 16)
    files = [DummyFile('/tmp/a.png', 'a caption'), DummyFile('/tmp/b.png', 'b caption')]
    batch = DummyBatch(files, latents=lat)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        out_path = tmp.name

    res = evaluate_dataset(compute_fn, [batch], out_json=out_path, eval_config={'model': 'test'})
    jr = res.get('json_report')
    assert jr is not None
    ds_key = list(jr['datasets'].keys())[0]
    item_stats = jr['datasets'][ds_key]['item_stats']
    for path, stats in item_stats.items():
        # ensure we have both average_loss_raw (non-ablated) and average_ablation_delta present (may be None if no ablation recorded for that path)
        assert 'average_loss_raw' in stats
        assert 'average_ablation_delta' in stats


def test_eval_enforces_dropout_zero():
    import torch.nn as nn
    # create a DummySD with a dropout in its unet
    class SDWithDropout(DummySD):
        def __init__(self):
            super().__init__()
            class U(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.drop = nn.Dropout(0.05)
            self.unet = U()

    sd = SDWithDropout()
    # ensure dropout initially non-zero
    import torch.nn as _nn
    found = False
    for m in sd.unet.modules():
        if isinstance(m, _nn.Dropout):
            found = True
            assert m.p == 0.05
    assert found

    # creating the compute fn should enforce dropout=0
    compute_fn = make_compute_fn_for_sd(sd)
    found_after = False
    for m in sd.unet.modules():
        if isinstance(m, _nn.Dropout):
            found_after = True
            assert m.p == 0.0
    assert found_after
