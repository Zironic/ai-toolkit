import torch
from types import SimpleNamespace
import types

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
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


class FakeWriter:
    def __init__(self):
        self.scalars = []
        self.histograms = []
    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))
    def add_histogram(self, tag, vals, step):
        self.histograms.append((tag, vals, step))


def test_per_example_tensorboard_logging(monkeypatch):
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
    cfg['train'] = {'steps': 1, 'log_per_example': True, 'log_per_example_to_tensorboard': True, 'max_per_example_to_tb': 5}

    trainer = SDTrainer(0, job, cfg)

    # Replace SummaryWriter with fake writer
    trainer.writer = FakeWriter()

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
    trainer.sd = sd

    # Fake batch
    class FakeBatch:
        def __init__(self):
            self.control_tensor = None
            self.tensor = torch.rand(2, 3, 64, 64)
            self.latents = torch.zeros((2, 4, 16, 16))
            self.file_items = [SimpleNamespace(path=f"img_{i}.jpg", is_reg=False, prior_reg=False, crop_height=64, crop_width=64, network_weight=1.0, dataset_config=SimpleNamespace(dataset_path='ds'), raw_caption='c')) for i in range(2)]
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
    timesteps = torch.tensor([1, 3])

    def fake_process_general_training_batch(b):
        return noisy, noise, timesteps, ['p1', 'p2'], None

    trainer.process_general_training_batch = fake_process_general_training_batch

    # Run hook -> this will call tensorboard logging code path
    loss_dict = trainer.hook_train_loop(batch)

    # Ensure histogram logged
    hist_tags = [h[0] for h in trainer.writer.histograms]
    assert 'per_example/loss' in hist_tags

    # Ensure sample scalars logged (we set max_per_example_to_tb=5)
    scalar_tags = [s[0] for s in trainer.writer.scalars]
    assert any(t.startswith('per_example_samples') for t in scalar_tags)
