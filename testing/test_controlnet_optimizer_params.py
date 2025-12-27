import os
from collections import OrderedDict

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.config_modules import AdapterConfig


class DummyJob:
    def __init__(self, name='testjob'):
        self.name = name
        self.training_folder = os.path.join(os.getcwd(), 'output', 'testjob')
        self.raw_config = {}
        self.log_dir = None
        # minimal attributes used by BaseTrainProcess
        from collections import OrderedDict
        self.meta = OrderedDict()


def test_adapter_train_flag_default_false():
    cfg = OrderedDict({'adapter': {'type': 'control_net', 'name_or_path': 'dummy/path'}, 'model': {'name_or_path': 'dummy_model'}})
    job = DummyJob('job_default')
    proc = BaseSDTrainProcess(0, job, cfg)
    assert proc.adapter_config is not None
    assert proc.adapter_config.train is False


def test_adapter_train_flag_explicit_true():
    cfg = OrderedDict({'adapter': {'type': 'control_net', 'name_or_path': 'dummy/path', 'train': True}, 'model': {'name_or_path': 'dummy_model'}})
    job = DummyJob('job_true')
    proc = BaseSDTrainProcess(0, job, cfg)
    assert proc.adapter_config is not None
    assert proc.adapter_config.train is True
