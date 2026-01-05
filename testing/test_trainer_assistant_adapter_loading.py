import torch
from types import SimpleNamespace
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import ModelConfig
from collections import OrderedDict


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


def test_assistant_controlnet_is_loaded_and_set(monkeypatch):
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())

    job = SimpleNamespace()
    job.training_folder = './tmp_training'
    job.raw_config = {}
    job.name = 'test_job'
    job.log_dir = None
    job.training_seed = None
    job.meta = OrderedDict()

    cfg = OrderedDict()
    cfg['model'] = {'name_or_path': 'dummy_model'}
    cfg['train'] = {'steps': 1}

    trainer = SDTrainer(0, job, cfg)

    # set train config fields to request an assistant controlnet
    trainer.train_config.adapter_assist_name_or_path = 'some_adapter'
    trainer.train_config.adapter_assist_type = 'control_net'

    class DummyAdapter:
        def __init__(self):
            self.to_called = False
            self.eval_called = False
            self.last_requires_grad = None
        def to(self, *args, **kwargs):
            self.to_called = True
            return self
        def eval(self):
            self.eval_called = True
            return self
        def requires_grad_(self, val):
            self.last_requires_grad = val
            return self

    # monkeypatch loading to return DummyAdapter
    monkeypatch.setattr('extensions_built_in.sd_trainer.SDTrainer.ControlNetModel.from_pretrained', lambda *args, **kwargs: DummyAdapter())

    trainer.before_dataset_load()

    assert trainer.assistant_adapter is not None
    assert getattr(trainer.assistant_adapter, 'to_called', False) or getattr(trainer.assistant_adapter, 'eval_called', False) or trainer.assistant_adapter.last_requires_grad == False
