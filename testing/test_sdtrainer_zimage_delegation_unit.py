import os, sys
import types
from types import SimpleNamespace
import pytest

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Prevent heavy torch import at collection time by stubbing a minimal module when needed
import types as _types
if 'torch' not in sys.modules:
    fake_torch = _types.ModuleType('torch')
    fake_torch.Tensor = object
    fake_torch.is_tensor = lambda x: hasattr(x, '__class__')
    fake_torch.float32 = 'float32'
    fake_torch.bfloat16 = 'bfloat16'
    fake_torch.device = lambda *a, **k: None
    fake_torch.zeros = lambda *a, **k: None
    fake_torch.nn = _types.ModuleType('torch.nn')
    fake_torch.nn.functional = _types.ModuleType('torch.nn.functional')
    fake_torch.nn.functional.interpolate = lambda *a, **k: None
    fake_torch.set_grad_enabled = lambda v: None
    fake_torch.is_grad_enabled = lambda: True
    sys.modules['torch'] = fake_torch

try:
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from toolkit.config_modules import AdapterConfig
except Exception as e:
    import pytest as _pytest
    _pytest.skip(f"integration-only: cannot import SDTrainer in this environment ({e})", allow_module_level=True)

# Minimal dummy accelerator to satisfy trainer init
class DummyAccelerator:
    def __init__(self):
        self.device = None
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


def test_trainer_delegates_zimage_helpers(monkeypatch):
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    called = {}

    def fake_collect(batch):
        called['collect'] = True
        return 'SENTINEL'

    monkeypatch.setattr('extensions_built_in.diffusion_models.z_image.z_image.collect_preencoded_zimage_context_for_batch', fake_collect)

    batch = SimpleNamespace()
    batch.file_items = []

    assert trainer._collect_preencoded_zimage_context_for_batch(batch) == 'SENTINEL'
    assert called.get('collect', False) is True

    def fake_precompute(sd, dl):
        called['precompute'] = True

    monkeypatch.setattr('extensions_built_in.diffusion_models.z_image.z_image.precompute_zimage_control_contexts', fake_precompute)
    trainer._precompute_zimage_control_contexts()
    assert called.get('precompute', False) is True
