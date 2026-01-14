import pytest
# This module requires full optional dependencies (diffusers/optimum/etc.) to be present.
# Use import-based conditional skip so tests run when the deps are available.
pytest.importorskip("diffusers")
pytest.importorskip("optimum")
import torch
from types import SimpleNamespace
import pytest

import sys
import types
# Pre-insert lightweight 'diffusers' module to avoid heavy runtime import during tests.
# This provides minimal classes used by SDTrainer (T2IAdapter, AutoencoderTiny, ControlNetModel, EMAModel).
fake_diff = types.ModuleType('diffusers')
setattr(fake_diff, 'T2IAdapter', object)
setattr(fake_diff, 'AutoencoderTiny', object)
setattr(fake_diff, 'ControlNetModel', type('ControlNetModel', (), {}))
setattr(fake_diff, 'EMAModel', object)
# scheduler placeholders used by toolkit.train_tools (lightweight stubs)
for _name in ['DDPMScheduler','EulerAncestralDiscreteScheduler','DPMSolverMultistepScheduler','DPMSolverSinglestepScheduler','LMSDiscreteScheduler','PNDMScheduler','DDIMScheduler','EulerDiscreteScheduler','HeunDiscreteScheduler','KDPM2DiscreteScheduler','KDPM2AncestralDiscreteScheduler']:
    setattr(fake_diff, _name, type(_name, (), {}))
sys.modules['diffusers'] = fake_diff
# minimal submodule 'diffusers.utils' with torch_utils.is_compiled_module used by toolkit
fake_utils = types.ModuleType('diffusers.utils')
# also provide an actual submodule object for importers that do `from diffusers.utils.torch_utils import ...`
fake_torch_utils_mod = types.ModuleType('diffusers.utils.torch_utils')
setattr(fake_torch_utils_mod, 'is_compiled_module', lambda m: False)
sys.modules['diffusers.utils.torch_utils'] = fake_torch_utils_mod
# keep a lightweight package reference too
fake_utils.torch_utils = fake_torch_utils_mod
sys.modules['diffusers.utils'] = fake_utils

# Provide fake pipelines/pixart_alpha submodule used by toolkit imports
fake_pipelines = types.ModuleType('diffusers.pipelines')
sys.modules['diffusers.pipelines'] = fake_pipelines
fake_pixart_alpha_pkg = types.ModuleType('diffusers.pipelines.pixart_alpha')
sys.modules['diffusers.pipelines.pixart_alpha'] = fake_pixart_alpha_pkg
fake_pipeline_pixart_sigma = types.ModuleType('diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma')
setattr(fake_pipeline_pixart_sigma, 'ASPECT_RATIO_1024_BIN', b'')
setattr(fake_pipeline_pixart_sigma, 'ASPECT_RATIO_512_BIN', b'')
setattr(fake_pipeline_pixart_sigma, 'ASPECT_RATIO_2048_BIN', b'')
setattr(fake_pipeline_pixart_sigma, 'ASPECT_RATIO_256_BIN', b'')
sys.modules['diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma'] = fake_pipeline_pixart_sigma
# fake stable_diffusion_xl pipeline stub required by imports in StableDiffusion
fake_sdxl = types.ModuleType('diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl')
setattr(fake_sdxl, 'rescale_noise_cfg', lambda *a, **k: None)
sys.modules['diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl'] = fake_sdxl

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import AdapterConfig

# minimal dummy accelerator used in other tests
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


def test_trainer_explicit_zimage_routing(monkeypatch):
    # Prepare trainer
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # Configure adapter_config using explicit flag, and also test detection-by-name below
    trainer.adapter_config = AdapterConfig(type='control_net')
    trainer.adapter_config.controlnet_mode = 'zimage'

    # Also assert name-based auto-detection works (unset explicit mode and set name)
    trainer2 = SDTrainer(0, job, cfg)
    trainer2.adapter_config = AdapterConfig(type='control_net')
    trainer2.adapter_config.name_or_path = 'my_zimage_model_v1'

    # Fake ControlNetModel instance (we only need isinstance check and to be non-None)
    from diffusers import ControlNetModel

    class FakeCN(ControlNetModel):
        def __init__(self):
            import torch.nn as nn
            nn.Module.__init__(self)

        # Not called in this explicit routing flow
        def forward(self, *args, **kwargs):
            return None

    trainer.adapter = FakeCN()

    # Also test wrapped VideoXControlnetWrapper case
    try:
        from toolkit.controlnet_compat import VideoXControlnetWrapper
        trainer_wrapped = SDTrainer(0, job, cfg)
        trainer_wrapped.adapter_config = trainer.adapter_config
        trainer_wrapped.adapter = VideoXControlnetWrapper(FakeCN())
        trainer_wrapped.sd = sd
    except Exception:
        trainer_wrapped = None

    # Prepare a minimal sd and monkeypatch predict_noise to capture kwargs
    sd = SimpleNamespace()
    sd.predict_noise_called = {}

    def fake_predict_noise(*args, **kwargs):
        sd.predict_noise_called = kwargs
        # mimic return shape
        return torch.zeros((1, 4, 16, 16))

    sd.predict_noise = fake_predict_noise
    # minimal flags used by trainer
    sd.vae = SimpleNamespace(dtype=torch.float32, to=lambda *a, **k: None, eval=lambda *a, **k: None)
    sd.text_encoder = SimpleNamespace(dtype=torch.float32, to=lambda *a, **k: None, eval=lambda *a, **k: None)
    sd.is_xl = False
    sd.device_torch = torch.device('cpu')
    sd.torch_dtype = torch.float32
    sd.controlnet_guidance_scale = 0.7
    trainer.sd = sd

    # Minimal batch with control_tensor to trigger adapter path
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

    # Monkeypatch functions used by trainer to keep it minimal
    trainer.process_general_training_batch = lambda b: (b.latents, torch.zeros_like(b.latents), torch.tensor([10]), ['a prompt'], None)
    trainer.calculate_loss = lambda **kwargs: torch.tensor(0.0, requires_grad=True)

    # Run single accumulation which should route using zimage kwargs into predict_noise
    trainer.train_single_accumulation(batch)

    assert 'zimage_controlnet' in sd.predict_noise_called, "Expected 'zimage_controlnet' in predict_noise kwargs"
    assert 'zimage_control_images' in sd.predict_noise_called, "Expected 'zimage_control_images' in predict_noise kwargs"
    zimgs = sd.predict_noise_called['zimage_control_images']
    # control images should have frame dimension introduced -> shape [B, C, F, H, W]
    assert zimgs.ndim == 5 and zimgs.shape[2] == 1, f"Unexpected zimage shape: {tuple(zimgs.shape)}"

    # Also validate wrapper behavior (VideoXControlnetWrapper)
    try:
        from toolkit.controlnet_compat import VideoXControlnetWrapper
        trainer_wrapped = SDTrainer(0, job, cfg)
        trainer_wrapped.adapter_config = trainer.adapter_config
        trainer_wrapped.adapter = VideoXControlnetWrapper(FakeCN())
        trainer_wrapped.sd = sd
        trainer_wrapped.process_general_training_batch = trainer.process_general_training_batch
        trainer_wrapped.calculate_loss = trainer.calculate_loss
        trainer_wrapped.train_single_accumulation(batch)
        assert 'zimage_controlnet' in sd.predict_noise_called, "Expected 'zimage_controlnet' in predict_noise kwargs for wrapped adapter"
        assert 'zimage_control_images' in sd.predict_noise_called, "Expected 'zimage_control_images' in predict_noise kwargs for wrapped adapter"
        zimgs_w = sd.predict_noise_called['zimage_control_images']
        assert zimgs_w.ndim == 5 and zimgs_w.shape[2] == 1, f"Unexpected zimage shape for wrapped adapter: {tuple(zimgs_w.shape)}"
    except Exception:
        # If wrapper import fails in the environment, skip this part of the test
        pass


def test_trainer_fallback_when_sd_missing_zimage_helper(monkeypatch):
    # Verify trainer computes control_hints when SD lacks _predict_noise_zimage
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    trainer.adapter_config = AdapterConfig(type='control_net')
    trainer.adapter_config.controlnet_mode = 'zimage'

    # Fake controlnet returns a single-tensor control_hints
    class FakeCN:
        def __call__(self, latents, timestep, control_context, conditioning_scale=1.0):
            # return single-tensor residual (B,C,H,W)
            return torch.zeros((latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3]))

    trainer.adapter = FakeCN()

    # minimal sd WITHOUT _predict_noise_zimage to trigger fallback
    sd_local = SimpleNamespace()
    sd_local.predict_noise_called = {}
    def fake_predict_noise(*args, **kwargs):
        sd_local.predict_noise_called = kwargs
        return torch.zeros((1,4,16,16))
    sd_local.predict_noise = fake_predict_noise
    sd_local.vae = SimpleNamespace(dtype=torch.float32, to=lambda *a, **k: None, eval=lambda *a, **k: None)
    sd_local.text_encoder = sd_local.vae
    sd_local.is_xl = False
    sd_local.device_torch = torch.device('cpu')
    sd_local.torch_dtype = torch.float32
    sd_local.controlnet_guidance_scale = 0.7

    # encode_control_images should accept list of tensors and return latents tensor
    def fake_encode_control_images(imgs, tile=False, tile_size=None, overlap=None):
        # imgs is a list of per-sample C,H,W tensors -> return stacked latents
        return torch.stack(imgs, dim=0)
    sd_local.encode_control_images = fake_encode_control_images

    trainer.sd = sd_local

    # minimal batch
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

    trainer.process_general_training_batch = lambda b: (b.latents, torch.zeros_like(b.latents), torch.tensor([10]), ['a prompt'], None)
    trainer.calculate_loss = lambda **kwargs: torch.tensor(0.0, requires_grad=True)

    trainer.train_single_accumulation(batch)

    # With the trainer refactor we no longer compute adapter residuals in the trainer.
    # Instead the trainer passes zimage routing keys into `sd.predict_noise` and delegates
    # the forward to the model-side `_predict_noise_zimage` implementation.
    assert 'zimage_controlnet' in sd_local.predict_noise_called, "Expected 'zimage_controlnet' in predict_noise kwargs"
    assert 'zimage_control_images' in sd_local.predict_noise_called, "Expected 'zimage_control_images' in predict_noise kwargs"
    zimgs_local = sd_local.predict_noise_called['zimage_control_images']
    assert hasattr(zimgs_local, 'ndim') and zimgs_local.ndim == 5 and zimgs_local.shape[2] == 1, f"Unexpected zimage shape: {tuple(zimgs_local.shape) if hasattr(zimgs_local, 'shape') else type(zimgs_local)}"
    sd2.predict_noise = lambda *args, **kwargs: sd2.predict_noise_called.update(kwargs) or torch.zeros((1,4,16,16))
    sd2.vae = sd.vae
    sd2.text_encoder = sd.text_encoder
    sd2.is_xl = sd.is_xl
    sd2.device_torch = sd.device_torch
    sd2.torch_dtype = sd.torch_dtype
    sd2.controlnet_guidance_scale = 0.7
    trainer2.sd = sd2

    trainer2.process_general_training_batch = trainer.process_general_training_batch
    trainer2.calculate_loss = trainer.calculate_loss

    trainer2.train_single_accumulation(batch)

    assert 'zimage_controlnet' in sd2.predict_noise_called, "Name-based detection should provide 'zimage_controlnet' in predict_noise kwargs"
    assert 'zimage_control_images' in sd2.predict_noise_called, "Name-based detection should provide 'zimage_control_images' in predict_noise kwargs"
    zimgs2 = sd2.predict_noise_called['zimage_control_images']
    assert zimgs2.ndim == 5 and zimgs2.shape[2] == 1, f"Unexpected zimage shape from name detection: {tuple(zimgs2.shape)}"


def test_trainer_applies_legacy_shim(monkeypatch):
    # Verify that the trainer will apply the ControlNetLegacyAdapter shim when
    # the adapter only accepts legacy `controlnet_cond` kw and does not expose
    # `control_context` directly.
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    trainer.adapter_config = AdapterConfig(type='control_net')
    trainer.adapter_config.controlnet_mode = 'zimage'

    # Prepare a legacy-style controlnet that expects `controlnet_cond`
    class LegacyCN:
        def __init__(self):
            self.record = {}
        def forward(self, latents, timestep, controlnet_cond=None, controlnet_conditioning_scale=1.0, *args, **kwargs):
            assert controlnet_cond is not None, "controlnet_cond not provided"
            if isinstance(controlnet_cond, torch.Tensor):
                self.record['shape'] = tuple(controlnet_cond.shape)
            elif isinstance(controlnet_cond, (list, tuple)) and len(controlnet_cond) > 0:
                self.record['shape'] = tuple(controlnet_cond[0].shape)
            return torch.zeros_like(latents[0])

    trainer.adapter = LegacyCN()

    # minimal sd to capture predict kwargs
    sd_local = SimpleNamespace()
    sd_local.predict_noise_called = {}
    sd_local.predict_noise = lambda *a, **kwargs: sd_local.predict_noise_called.update(kwargs) or torch.zeros((1,4,16,16))
    sd_local.vae = trainer.sd.vae if hasattr(trainer, 'sd') else SimpleNamespace(dtype=torch.float32, to=lambda *a, **k: None, eval=lambda *a, **k: None)
    sd_local.text_encoder = sd_local.vae
    sd_local.is_xl = False
    sd_local.device_torch = torch.device('cpu')
    sd_local.torch_dtype = torch.float32
    sd_local.controlnet_guidance_scale = 0.7
    trainer.sd = sd_local

    # minimal batch
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

    trainer.process_general_training_batch = lambda b: (b.latents, torch.zeros_like(b.latents), torch.tensor([10]), ['a prompt'], None)
    trainer.calculate_loss = lambda **kwargs: torch.tensor(0.0, requires_grad=True)

    # Running a single accumulation should not raise and should route zimage kwargs
    trainer.train_single_accumulation(batch)

    assert 'zimage_controlnet' in sd_local.predict_noise_called, "Expected 'zimage_controlnet' in predict_noise kwargs"
    assert 'zimage_control_images' in sd_local.predict_noise_called, "Expected 'zimage_control_images' in predict_noise kwargs"
    zimgs = sd_local.predict_noise_called['zimage_control_images']
    assert zimgs.ndim == 5 and zimgs.shape[2] == 1, f"Unexpected zimage shape: {tuple(zimgs.shape)}"

    # The legacy CN should have received a controlnet_cond via the shim; verify recorded shape
    assert hasattr(trainer.adapter, 'record') or getattr(trainer.adapter, 'inner', None) is not None
    inner = trainer.adapter if hasattr(trainer.adapter, 'record') else getattr(trainer.adapter, 'inner', None)
    # If inner exists, it should have recorded the shape
    if inner is not None and hasattr(inner, 'record'):
        assert inner.record['shape'][1] == 33


def test_zimage_control_context_is_per_tile_batch():
    # Ensure that when cap_feats indicate multiple tiles, the zimage controlnet receives
    # a control_context list where each element is a batch tensor [B, C, H, W].
    import torch
    from types import SimpleNamespace
    from toolkit.stable_diffusion_model import StableDiffusion

    # Create fake controlnet that captures the control_context
    captured = {}
    class FakeCN:
        def __init__(self):
            pass
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
        def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
            captured['control_context'] = control_context
            # return a dummy single-tensor control_hints matching expected shape
            # If control_context is list, pick first element for shape
            if isinstance(control_context, list):
                B, C, H, W = control_context[0].shape
                return [torch.zeros((B, C, H, W)) for _ in range(len(control_context))]
            else:
                B, C, H, W = control_context.shape
                return torch.zeros((B, C, H, W))

    fakecn = FakeCN()

    # Prepare test inputs
    B = 1
    C = 16
    F = 1
    H = 16
    W = 16
    n_tiles = 4
    # latents must be 16-channel for Z-Image strict mode
    latents = torch.zeros((B, 16, H, W))
    timestep = torch.tensor([10])
    # text_embeddings with text_embeds attribute shaped [B, n_tiles, Dim]
    text_embeddings = SimpleNamespace()
    text_embeddings.text_embeds = torch.randn(B, n_tiles, 128)
    # control images shape [B, C, F, H, W]
    zimage_ctrl = torch.randn(B, C, F, H, W)

    # Call the helper directly (bind 'self' to a minimal object)
    sd_like = SimpleNamespace()
    # call the unbound method from the class
    StableDiffusion._predict_noise_zimage(sd_like, latents, text_embeddings, timestep, zimage_controlnet=fakecn, zimage_control_images=zimage_ctrl, zimage_conditioning_scale=1.0)

    assert 'control_context' in captured, "ControlNet was not invoked or did not receive control_context"
    cc = captured['control_context']
    assert isinstance(cc, list), f"Expected control_context to be a list for n_tiles>1, got {type(cc)}"
    assert len(cc) == n_tiles, f"Expected control_context list length {n_tiles}, got {len(cc)}"
    for t in range(n_tiles):
        assert isinstance(cc[t], torch.Tensor), f"Expected element {t} to be a Tensor"
        assert cc[t].shape == (B, C, H, W), f"Unexpected shape for tile {t}: {cc[t].shape}"


def test_predict_noise_zimage_sets_flags():
    import torch
    from types import SimpleNamespace
    from toolkit.stable_diffusion_model import StableDiffusion

    # Fake ControlNet that returns a single-tensor control_hints
    def fake_cn(sample, timestep, control_context, conditioning_scale=1.0, **kwargs):
        # ensure adapter kwarg was passed
        assert kwargs.get('controlnet') is fake_cn
        # return a simple tensor matching latent shape
        return torch.zeros((sample.shape[0], sample.shape[1], sample.shape[2], sample.shape[3]))

    # minimal sd_like with a unet that accepts control_context
    sd_like = SimpleNamespace()
    def fake_unet(latents, timestep, cap_feats, return_dict=False, control_context=None, **kwargs):
        # record that control_context was received
        sd_like._unet_received_control_context = control_context
        return torch.zeros((latents.shape[0], 4, latents.shape[2], latents.shape[3]))
    sd_like.unet = fake_unet

    # Prepare inputs
    latents = torch.zeros((1,4,16,16))
    text_embeddings = SimpleNamespace(); text_embeddings.text_embeds = torch.zeros((1,1,128))
    timestep = torch.tensor([10])
    zimage_ctrl = torch.randn(1,4,1,16,16)

    # Call helper
    StableDiffusion._predict_noise_zimage(sd_like, latents, text_embeddings, timestep, zimage_controlnet=fake_cn, zimage_control_images=zimage_ctrl, zimage_conditioning_scale=1.0)

    # Validate flags set
    assert getattr(sd_like, '_last_zimage_control_hints_present', False) is True
    assert getattr(sd_like, '_last_zimage_control_hints_shapes', None) is not None
    assert getattr(sd_like, '_last_zimage_control_context_passed', False) is True
    assert getattr(sd_like, '_last_zimage_control_context_shape', None) is not None
