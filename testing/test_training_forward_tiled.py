import torch
from types import SimpleNamespace
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.config_modules import ModelConfig


def make_job_and_cfg():
    job = SimpleNamespace()
    job.training_folder = "./tmp_training"
    job.raw_config = {}
    job.name = 'test_job'
    job.log_dir = None
    job.training_seed = None
    job.meta = SimpleNamespace()
    # Provide a serializable config dict to avoid JSON serialization issues in BaseProcess
    cfg = {'model': {'name_or_path': 'dummy_model'}, 'train': {'steps': 1}}
    return job, cfg


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


def test_training_forward_handles_control_tensor_list(monkeypatch):
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # Minimal SD stub
    sd = SimpleNamespace()
    sd.encode_prompt = lambda *a, **k: SimpleNamespace(text_embeds=torch.zeros((1, 8, 16)), pooled_embeds=None)
    sd.encode_control_in_text_embeddings = False
    sd.encode_images = lambda x: torch.zeros((x.shape[0], 4, 16, 16))
    sd.vae = SimpleNamespace(to=lambda *a, **k: None, eval=lambda *a, **k: None, dtype=torch.float32)
    sd.text_encoder = SimpleNamespace(dtype=torch.float32, to=lambda *a, **k: None, eval=lambda *a, **k: None)
    sd.is_xl = False
    sd.device_torch = torch.device('cpu')
    sd.torch_dtype = torch.float32
    sd.condition_noisy_latents = lambda noisy_latents, batch: noisy_latents
    sd.predict_noise = lambda **k: torch.zeros_like(k.get('latents'))
    trainer.sd = sd
    trainer.sd.vae_torch_dtype = torch.float32
    trainer.sd.te_torch_dtype = torch.float32

    # replace preprocess_batch to supply a control_tensor as a list of tensors
    def fake_preprocess_batch(batch):
        # simulate encode_control_images returning a list of per-sample tensors
        batch.control_tensor = [torch.zeros((1, 4, 16, 16))]
        return batch
    trainer.preprocess_batch = fake_preprocess_batch

    # minimal process_general_training_batch
    noisy = torch.randn(1, 4, 16, 16)
    noise = torch.randn_like(noisy)
    timesteps = torch.tensor([10])
    trainer.process_general_training_batch = lambda batch: (noisy, noise, timesteps, ['a prompt'], None)

    # Fake ControlNetModel adapter that records controlnet_cond
    from diffusers import ControlNetModel
    class FakeCN(ControlNetModel):
        def __init__(self):
            import torch.nn as nn
            nn.Module.__init__(self)
        def forward(self, noisy_latents, timesteps_arg, encoder_hidden_states=None, controlnet_cond=None, **kwargs):
            return [torch.zeros((1,3,8,8))], torch.zeros((1,6,4,4))
    trainer.adapter = FakeCN()
    trainer.assistant_adapter = None

    # replace calculate_loss to avoid complex internals
    trainer.calculate_loss = lambda **k: torch.tensor(0.1, requires_grad=True)
    trainer.train_config.controlnet_offload_strategy = 'none'

    class FakeBatch:
        def __init__(self):
            self.control_tensor = None
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

    loss = trainer.train_single_accumulation(batch)
    assert isinstance(loss, torch.Tensor)


def test_training_forward_handles_nested_control_tensor_list(monkeypatch):
    # similar test where preprocess returns nested list per sample
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    sd = SimpleNamespace()
    sd.encode_prompt = lambda *a, **k: SimpleNamespace(text_embeds=torch.zeros((1, 8, 16)), pooled_embeds=None)
    sd.encode_control_in_text_embeddings = False
    sd.encode_images = lambda x: torch.zeros((x.shape[0], 4, 16, 16))
    sd.vae = SimpleNamespace(to=lambda *a, **k: None, eval=lambda *a, **k: None, dtype=torch.float32)
    sd.text_encoder = SimpleNamespace(dtype=torch.float32, to=lambda *a, **k: None, eval=lambda *a, **k: None)
    sd.is_xl = False
    sd.device_torch = torch.device('cpu')
    sd.torch_dtype = torch.float32
    sd.condition_noisy_latents = lambda noisy_latents, batch: noisy_latents
    sd.predict_noise = lambda **k: torch.zeros_like(k.get('latents'))
    trainer.sd = sd
    trainer.sd.vae_torch_dtype = torch.float32
    trainer.sd.te_torch_dtype = torch.float32

    def fake_preprocess_batch(batch):
        # nested list per-sample: [[tile1, tile2]]
        batch.control_tensor = [[torch.zeros((4, 16, 16)), torch.zeros((4, 16, 16))]]
        return batch
    trainer.preprocess_batch = fake_preprocess_batch

    noisy = torch.randn(1, 4, 16, 16)
    noise = torch.randn_like(noisy)
    timesteps = torch.tensor([10])
    trainer.process_general_training_batch = lambda batch: (noisy, noise, timesteps, ['a prompt'], None)

    from diffusers import ControlNetModel
    class FakeCN(ControlNetModel):
        def __init__(self):
            import torch.nn as nn
            nn.Module.__init__(self)
        def forward(self, noisy_latents, timesteps_arg, encoder_hidden_states=None, controlnet_cond=None, **kwargs):
            return [torch.zeros((1,3,8,8))], torch.zeros((1,6,4,4))
    trainer.adapter = FakeCN()
    trainer.assistant_adapter = None
    trainer.calculate_loss = lambda **k: torch.tensor(0.1, requires_grad=True)
    trainer.train_config.controlnet_offload_strategy = 'none'

    class FakeBatch:
        def __init__(self):
            self.control_tensor = None
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
    loss = trainer.train_single_accumulation(batch)
    assert isinstance(loss, torch.Tensor)
