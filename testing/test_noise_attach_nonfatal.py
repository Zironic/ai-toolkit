import torch
from types import SimpleNamespace

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


def test_noise_attach_nonfatal(monkeypatch):
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

    # minimal attributes
    trainer.optimizer = SimpleNamespace(zero_grad=lambda *a, **k: None, step=lambda *a, **k: None, param_groups=[{'lr': 0.0}])
    trainer.lr_scheduler = SimpleNamespace(step=lambda *args, **kwargs: None)

    sd = SimpleNamespace()
    sd.predict_noise = lambda **kwargs: kwargs.get('latents')
    sd.encode_images = lambda x: torch.zeros((x.shape[0], 4, 16, 16))
    sd.encode_prompt = lambda *args, **kwargs: torch.zeros((2, 4, 16, 16))
    sd.vae = SimpleNamespace(to=lambda *args, **kwargs: None, eval=lambda *args, **kwargs: None, dtype=torch.float32)
    sd.text_encoder = SimpleNamespace(dtype=torch.float32, to=lambda *args, **kwargs: None, eval=lambda *args, **kwargs: None)
    sd.is_xl = False
    sd.device_torch = torch.device('cpu')
    sd.torch_dtype = torch.float32
    sd.vae_torch_dtype = torch.float32
    sd.te_torch_dtype = torch.float32
    sd.is_multistage = False
    sd.condition_noisy_latents = lambda latents, batch: latents
    trainer.sd = sd

    class FakeBatch:
        def __init__(self):
            self.control_tensor = None
            self.tensor = torch.rand(2, 3, 64, 64)
            self.latents = torch.zeros((2, 4, 16, 16))
            self.file_items = [SimpleNamespace(path=f"img_{i}.jpg", is_reg=False, prior_reg=False, crop_height=64, crop_width=64, network_weight=1.0, dataset_config=SimpleNamespace(dataset_path='ds'), raw_caption='c') for i in range(2)]
            self.clip_image_tensor = None
            self.clip_image_embeds = None
            self.mask_tensor = None
            self.unconditional_latents = None
            self.control_tensor_list = None
            self.control_residuals = None
            self.inpaint_tensor = None
            self.sigmas = torch.tensor([0.5, 2.0])

        def get_network_weight_list(self):
            return [fi.network_weight for fi in self.file_items]

    batch = FakeBatch()

    noisy = torch.randn(2, 4, 16, 16, requires_grad=True)
    noise = torch.randn_like(noisy)
    timesteps = torch.tensor([1, 3])

    def fake_process_general_training_batch(b):
        return noisy, noise, timesteps, ['p1', 'p2'], None

    trainer.process_general_training_batch = fake_process_general_training_batch

    # ensure we have at least one param requiring grad so _fatal is not triggered
    trainer.params = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]

    # make attach_noise raise
    def failing_attach(self, loss_dict):
        raise RuntimeError("attach boom")

    monkeypatch.setattr(SDTrainer, '_attach_noise_metrics_to_loss_dict', failing_attach)

    loss_dict = trainer.hook_train_loop(batch)

    assert isinstance(loss_dict, dict)
    assert 'loss' in loss_dict
    # debug flags should still be present
    assert 'debug_flags' in loss_dict
