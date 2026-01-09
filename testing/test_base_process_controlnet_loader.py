from types import SimpleNamespace
import pytest

from jobs.process import BaseSDTrainProcess


class DummyControlNet:
    def __init__(self):
        self.name_or_path = 'repo/controlnet'
    def parameters(self):
        return []


def make_minimal_process():
    # Create an instance without running full initialization
    p = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
    # minimal attributes required by setup_controlnet_training
    p.sd = SimpleNamespace()
    p.sd.is_controlnet_enabled = False
    p.sd.controlnet = None
    p.sd.torch_dtype = None

    p.model_config = SimpleNamespace()
    p.model_config.controlnet_name_or_path = 'repo/controlnet'
    p.model_config.controlnet_file = None
    p.model_config.controlnet_offload_strategy = 'none'

    # train_config stub
    p.train_config = SimpleNamespace()

    # simple print hook
    p.print_and_status_update = lambda msg: None
    return p


def test_lazy_load_uses_prepare_controlnet_adapter(monkeypatch):
    proc = make_minimal_process()

    # monkeypatch the deterministic loader to return a DummyControlNet
    monkeypatch.setattr('toolkit.control_util.prepare_controlnet_adapter', lambda *args, **kwargs: DummyControlNet())

    # run setup
    proc.setup_controlnet_training()

    assert proc.sd.controlnet is not None
    assert getattr(proc.sd.controlnet, 'name_or_path', None) == 'repo/controlnet'
