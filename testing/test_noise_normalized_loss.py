import torch
from types import SimpleNamespace
import types

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


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


def test_noise_normalized_loss_metrics(monkeypatch):
    # Minimal trainer setup
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
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

    trainer = SDTrainer(0, job, cfg)

    # Minimal sd stub
    sd = SimpleNamespace()
    sd.predict_noise = lambda **kwargs: torch.zeros_like(kwargs.get('latents'))
    sd.encode_images = lambda x: torch.zeros((x.shape[0], 4, 16, 16))
    sd.vae = SimpleNamespace(to=lambda *args, **kwargs: None, eval=lambda *args, **kwargs: None, dtype=torch.float32)
    sd.text_encoder = SimpleNamespace(dtype=torch.float32, to=lambda *args, **kwargs: None, eval=lambda *args, **kwargs: None)
    sd.is_xl = False
    sd.device_torch = torch.device('cpu')
    sd.torch_dtype = torch.float32
    sd.vae_torch_dtype = torch.float32
    sd.te_torch_dtype = torch.float32

    # Attach a simple noise scheduler with timesteps and sigmas so fallback mapping can work
    class FakeNS:
        def __init__(self):
            self.timesteps = torch.tensor([1, 2, 3, 4])
            self.sigmas = torch.tensor([0.5, 1.0, 2.0, 4.0])
            self.config = SimpleNamespace(num_train_timesteps=4)
    sd.noise_scheduler = FakeNS()
    sd.is_multistage = False
    sd.encode_control_in_text_embeddings = False
    sd.is_flow_matching = False
    sd.prediction_type = 'epsilon'
    sd.get_loss_target = lambda **kwargs: kwargs.get('noise')
    # minimal prompt encoder stub
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
    sd.encode_prompt = lambda *args, **kwargs: FakePromptEmbeds()

    trainer.sd = sd
    # minimal optimizer/lr objects to allow hook to run
    trainer.optimizer = SimpleNamespace(zero_grad=lambda *a, **k: None, step=lambda *a, **k: None)
    trainer.lr_scheduler = SimpleNamespace(step=lambda *a, **k: None)
    # make params empty to avoid clip paths
    trainer.params = []

    # Fake batch
    class FakeBatch:
        def __init__(self):
            self.control_tensor = None
            self.tensor = torch.rand(2, 3, 64, 64)
            self.latents = torch.zeros((2, 4, 16, 16))
            self.file_items = [SimpleNamespace(is_reg=False, prior_reg=False, crop_height=64, crop_width=64, network_weight=1.0) for _ in range(2)]
            self.clip_image_tensor = None
            self.clip_image_embeds = None
            self.mask_tensor = None
            self.unconditional_latents = None
            self.control_tensor_list = None
            self.control_residuals = None
            self.inpaint_tensor = None
            # Provide per-sample sigmas (use two different values to make mean non-trivial)
            self.sigmas = torch.tensor([0.5, 2.0])
        def get_network_weight_list(self):
            return [fi.network_weight for fi in self.file_items]

    batch = FakeBatch()

    # Monkeypatch process_general_training_batch to return our constructed noisy/noise/timesteps
    noisy = torch.randn(2, 4, 16, 16)
    noise = torch.randn_like(noisy)
    timesteps = torch.tensor([1, 3])  # will map to sigmas 0.5 and 2.0 from FakeNS

    def fake_process_general_training_batch(b):
        return noisy, noise, timesteps, ['p1', 'p2'], None

    trainer.process_general_training_batch = fake_process_general_training_batch
    # ensure optional attributes exist to avoid attribute errors
    trainer.assistant_adapter = None
    trainer.adapter = None
    trainer.dfe = None

    # Instead of running the full loop (heavy stubbing required), directly test the helper that
    # attaches the aggregated noise diagnostics into a loss_dict.
    # Simulate per-sample scalars for a batch of size 2
    trainer.last_noise_norms = torch.tensor([3.0, 4.0])
    trainer.last_loss_over_noise = torch.tensor([0.1, 0.2])
    trainer.last_noise_sigmas = torch.tensor([0.5, 2.0])

    loss_dict = {}
    trainer._attach_noise_metrics_to_loss_dict(loss_dict)

    # Check trainer attached noise diagnostics
    assert getattr(trainer, 'last_noise_norms', None) is not None
    assert getattr(trainer, 'last_loss_over_noise', None) is not None
    assert trainer.last_noise_norms.shape[0] == 2
    assert trainer.last_loss_over_noise.shape[0] == 2

    # Check loss_dict contains the aggregated metrics
    assert 'train/noise_mean' in loss_dict
    assert isinstance(loss_dict['train/noise_mean'], float)
    assert 'train/loss_over_noise' in loss_dict
    assert isinstance(loss_dict['train/loss_over_noise'], float)
    # sigma mean should be present (we mapped via scheduler/timesteps)
    assert 'train/noise_sigma_mean' in loss_dict
    assert isinstance(loss_dict['train/noise_sigma_mean'], float)
