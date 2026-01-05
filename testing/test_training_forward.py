import torch
import types
from types import SimpleNamespace
from collections import OrderedDict
import pytest

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import ModelConfig

# Monkeypatch accelerator to avoid heavy initialization
class DummyAccelerator:
    def __init__(self):
        self.device = torch.device('cpu')
        self.is_local_main_process = True
        self.is_main_process = True
    def prepare(self, x):
        return x
    def backward(self, loss):
        # simulate accelerator backward
        loss.backward()
    def clip_grad_norm_(self, *args, **kwargs):
        return


def make_job_and_config():
    job = SimpleNamespace()
    job.training_folder = "./tmp_training"
    job.raw_config = {}
    job.name = 'test_job'
    job.log_dir = None
    job.training_seed = None
    job.meta = OrderedDict()
    cfg = OrderedDict()
    # Minimal model config
    cfg['model'] = {'name_or_path': 'dummy_model'}
    cfg['train'] = {'steps': 1}
    return job, cfg


def test_hook_before_train_loop_fails_when_control_enabled_but_no_dataset(monkeypatch):
    # Ensure fail-fast preflight check raises when controlnet is enabled but no dataset has control_type
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
    job, cfg = make_job_and_config()
    # enable controlnet but provide no dataset configs
    cfg['model']['controlnet_enabled'] = True
    cfg['model']['controlnet_name_or_path'] = None
    cfg['model']['controlnet_file'] = None

    proc = BaseSDTrainProcess(0, job, cfg)
    # provide a minimal sd to avoid prepare_accelerator failures
    proc.sd = SimpleNamespace(vae=SimpleNamespace(dtype=torch.float32), unet=None, text_encoder=None, refiner_unet=None, network=None)

    with pytest.raises(RuntimeError):
        proc.hook_before_train_loop()


class FakePromptEmbeds:
    def __init__(self):
        self.text_embeds = torch.zeros((1, 8, 16))
        self.attention_mask = torch.zeros((1, 16), dtype=torch.long)
    def to(self, *args, **kwargs):
        return self
    def clone(self):
        return self
    def detach(self):
        return self


def test_training_forward_with_control_images_and_mocked_adapter(monkeypatch):
    # Validate that training forward encodes control images (via sd.encode_control_images) and passes them to the adapter
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
    job, cfg = make_job_and_config()

    trainer = SDTrainer(0, job, cfg)

    # Prepare a minimal sd with encode_prompt and encode_control_images
    sd = SimpleNamespace()
    sd.encode_prompt = lambda *args, **kwargs: FakePromptEmbeds()
    # encode_control_images returns a latent of shape (bs, C, H_lat, W_lat)
    recorded = {}
    def fake_encode_control_images(control_tensor, tile=False, tile_size=None, overlap=None):
        bs = control_tensor.shape[0]
        recorded['encode_called'] = True
        return torch.zeros((bs, 4, 16, 16))
    sd.encode_control_images = fake_encode_control_images

    # ensure preprocessing replaces batch.control_tensor with encoded latents so adapter receives latents
    def fake_preprocess_batch(batch):
        if getattr(batch, 'control_tensor', None) is not None:
            batch.control_tensor = sd.encode_control_images(batch.control_tensor)
        return batch
    trainer.preprocess_batch = fake_preprocess_batch
    sd.encode_images = lambda x: torch.zeros((x.shape[0], 4, 16, 16))
    sd.vae = SimpleNamespace(to=lambda *args, **kwargs: None, eval=lambda *args, **kwargs: None, dtype=torch.float32)
    # minimal text encoder stub
    sd.text_encoder = SimpleNamespace(dtype=torch.float32, to=lambda *args, **kwargs: None, eval=lambda *args, **kwargs: None)
    # controls whether control images are encoded into text embeddings; set False for this test
    sd.encode_control_in_text_embeddings = False
    # small SD flags used in trainer branches
    sd.is_xl = False
    sd.device_torch = torch.device('cpu')
    sd.torch_dtype = torch.float32
    # stub out condition_noisy_latents to be a pass-through for this test
    sd.condition_noisy_latents = lambda noisy_latents, batch: noisy_latents
    def fake_predict_noise(**kwargs):
        latents = kwargs.get('latents')
        return torch.zeros_like(latents)
    sd.predict_noise = fake_predict_noise
    trainer.sd = sd
    # make sure trainer has vae_torch_dtype and te_torch_dtype to satisfy the sanity checks
    trainer.sd.vae_torch_dtype = torch.float32
    trainer.sd.te_torch_dtype = torch.float32


    # Monkeypatch process_general_training_batch to return simple tensors
    noisy = torch.randn(1, 4, 16, 16)
    noise = torch.randn_like(noisy)
    timesteps = torch.tensor([10])
    def fake_process_general_training_batch(batch):
        return noisy, noise, timesteps, ['a prompt'], None
    trainer.process_general_training_batch = fake_process_general_training_batch

    # Provide a fake adapter that *looks like* a ControlNetModel by subclassing it dynamically
    from diffusers import ControlNetModel
    recorded = {}
    class FakeCN(ControlNetModel):
        def __init__(self):
            import torch.nn as nn
            nn.Module.__init__(self)
        def forward(self, noisy_latents, timesteps_arg, encoder_hidden_states=None, controlnet_cond=None, **kwargs):
            # record shapes
            recorded['controlnet_cond_shape'] = None if controlnet_cond is None else tuple(controlnet_cond.shape)
            # return (down_block_res_samples, mid_block_res_sample)
            return [torch.zeros((1,3,8,8))], torch.zeros((1,6,4,4))
    fake_cn = FakeCN()
    trainer.adapter = fake_cn

    # Create a minimal batch object that implements required API
    class FakeBatch:
        def __init__(self):
            self.control_tensor = torch.rand(1, 3, 64, 64)
            self.tensor = torch.rand(1, 3, 64, 64)
            self.latents = torch.zeros((1, 4, 16, 16))
            self.file_items = [SimpleNamespace(is_reg=False, prior_reg=False, crop_height=64, crop_width=64, network_weight=1.0)]
            # optional tensors used in training flow
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

    # Monkeypatch calculate_loss to a simple MSE so we avoid much internal complexity
    def fake_calculate_loss(**kwargs):
        return torch.tensor(0.123, requires_grad=True)
    trainer.calculate_loss = fake_calculate_loss
    trainer.assistant_adapter = None

    # Ensure offload strategy is none so we don't attempt to use MemoryManager
    trainer.train_config.controlnet_offload_strategy = 'none'

    # Call train_single_accumulation (should run through encode_adapter or controlnet code path)
    loss = trainer.train_single_accumulation(batch)

    # Verify loss is returned and adapter received control tensor (encoded latent) of expected shape
    assert isinstance(loss, torch.Tensor)
    assert recorded.get('controlnet_cond_shape') is not None
    assert recorded['controlnet_cond_shape'][0] == 1
    # control latent channel dim should be 4 per our fake encoder
    assert recorded['controlnet_cond_shape'][1] == 4

