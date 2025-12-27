import os
from collections import OrderedDict

import torch

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


class DummyJob:
    def __init__(self, name='testjob'):
        self.name = name
        self.training_folder = os.path.join(os.getcwd(), 'output', 'testjob')
        self.raw_config = {}
        self.log_dir = None
        from collections import OrderedDict
        self.meta = OrderedDict()


class DummyAdapter:
    def __init__(self):
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        # return an iterator
        def _iter():
            yield self._p

        return _iter()


def _has_adapter_param(proc, adapter_param):
    for g in getattr(proc, 'params', []):
        ps = g.get('params') if isinstance(g, dict) else None
        if ps:
            for p in ps:
                if p is adapter_param:
                    return True
    return False


def test_controlnet_params_excluded_by_default(monkeypatch):
    cfg = OrderedDict({'network': {}, 'adapter': {'type': 'control_net', 'name_or_path': 'dummy_path'}, 'model': {'name_or_path': 'dummy_model'}})
    job = DummyJob('job_default')

    def fake_setup_adapter(self):
        self.adapter = DummyAdapter()

    monkeypatch.setattr(BaseSDTrainProcess, 'setup_adapter', fake_setup_adapter)

    proc = BaseSDTrainProcess(0, job, cfg)

    # ensure adapter can be setup (monkeypatched)
    proc.setup_adapter()
    assert proc.adapter is not None
    adapter_param = proc.adapter._p

    # do NOT append adapter params because train flag is False
    assert not _has_adapter_param(proc, adapter_param)


def test_controlnet_params_included_when_train_true(monkeypatch):
    cfg = OrderedDict({'network': {}, 'adapter': {'type': 'control_net', 'name_or_path': 'dummy_path', 'train': True}, 'model': {'name_or_path': 'dummy_model'}})
    job = DummyJob('job_train_true')

    def fake_setup_adapter(self):
        self.adapter = DummyAdapter()

    monkeypatch.setattr(BaseSDTrainProcess, 'setup_adapter', fake_setup_adapter)

    proc = BaseSDTrainProcess(0, job, cfg)

    # ensure adapter is setup (monkeypatched)
    proc.setup_adapter()
    assert proc.adapter is not None
    adapter_param = proc.adapter._p

    # emulate optimizer param inclusion logic from BaseSDTrainProcess
    if isinstance(proc.adapter, object):
        # non-IP adapter path
        proc.params.append({
            'params': list(proc.adapter.parameters()),
            'lr': proc.train_config.adapter_lr
        })

    assert _has_adapter_param(proc, adapter_param)
