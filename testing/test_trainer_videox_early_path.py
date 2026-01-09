import pytest
try:
    import torch
except Exception:
    pytest.skip("Skipping: PyTorch import failed in this environment", allow_module_level=True)

from types import SimpleNamespace
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import AdapterConfig

class DummyAccelerator:
    def __init__(self):
        self.device = torch.device('cpu')
        self.is_local_main_process = True
        self.is_main_process = True
    def prepare(self, x):
        return x
    def backward(self, loss):
        loss.backward()
    def clip_grad_norm_(self, *args, **kwargs):
        return


def make_job_and_cfg():
    job = SimpleNamespace()
    job.training_folder = "./tmp_training"
    job.raw_config = {}
    job.name = 'test_job'
    job.log_dir = None
    job.training_seed = None
    job.meta = {}
    cfg = {}
    cfg['model'] = {'name_or_path': 'dummy_model'}
    cfg['train'] = {'steps': 1}
    return job, cfg


def test_trainer_runs_videox_adapter_early(monkeypatch):
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # Mark adapter as zimage by monkeypatching is_zimage_adapter
    monkeypatch.setattr('toolkit.controlnet_utils.is_zimage_adapter', lambda a, cfg: True)

    called = {}

    class FakeVx:
        def __call__(self, latents, timesteps, control_context, conditioning_scale=1.0):
            called['args'] = (getattr(latents, 'shape', None), getattr(timesteps, 'shape', None) if hasattr(timesteps,'shape') else timesteps, getattr(control_context, 'shape', None))
            return torch.zeros((latents.shape[0], 16, 1, latents.shape[-2], latents.shape[-1]))

    trainer.adapter = FakeVx()
    trainer.adapter_config = AdapterConfig(type='control_net')
    trainer.adapter_config.controlnet_mode = 'zimage'

    # Minimal sd stub to capture predict_noise kwargs
    sd = SimpleNamespace()
    sd.predict_noise_called = {}
    def fake_predict_noise(*args, **kwargs):
        sd.predict_noise_called = kwargs
        return torch.zeros((1,4,16,16))
    sd.predict_noise = fake_predict_noise
    sd.vae = SimpleNamespace(dtype=torch.float32, to=lambda *a, **k: None)
    sd.text_encoder = sd.vae
    sd.is_xl = False
    sd.device_torch = torch.device('cpu')
    sd.torch_dtype = torch.float32
    sd.controlnet_guidance_scale = 0.7
    trainer.sd = sd

    # Minimal batch
    class FakeBatch:
        def __init__(self):
            self.control_tensor = torch.rand(1, 3, 64, 64)
            self.tensor = torch.rand(1, 3, 64, 64)
            self.latents = torch.zeros((1, 4, 16, 16))
            self.file_items = [SimpleNamespace(is_reg=False, prior_reg=False, crop_height=64, crop_width=64, network_weight=1.0)]
            self.clip_image_tensor = None
            self.clip_image_embeds = None
            self.clip_image_embeds_unconditional = None
            self.mask_tensor = None
            self.unconditional_latents = None
            self.control_tensor_list = None
            self.control_residuals = None
            self.inpaint_tensor = None
        def get_network_weight_list(self):
            return [fi.network_weight for fi in self.file_items]

    batch = FakeBatch()

    # Monkeypatch internal helpers used by trainer to keep it minimal
    trainer.process_general_training_batch = lambda b: (b.latents, torch.zeros_like(b.latents), torch.tensor([10]), ['a prompt'], None)
    trainer.calculate_loss = lambda **kwargs: torch.tensor(0.0, requires_grad=True)

    # Run single accumulation which should run the VideoX adapter early
    trainer.train_single_accumulation(batch)

    assert called, "Expected VideoX adapter to be invoked in early path"
    assert 'down_block_additional_residuals' in sd.predict_noise_called, "Trainer should have attached down residuals to pred_kwargs and passed them into predict_noise"