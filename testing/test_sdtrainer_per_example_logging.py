import torch
from types import SimpleNamespace

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.util.loss_utils import StreamingAggregator


class FakeFileItem:
    def __init__(self, path, dataset_config=None, raw_caption=''):
        self.path = path
        self.dataset_config = dataset_config or SimpleNamespace(dataset_path='ds1', folder_path=None, default_caption=None)
        self.raw_caption = raw_caption


class FakeBatch:
    def __init__(self, file_items):
        self.file_items = file_items


def make_trainer_stub():
    t = SDTrainer.__new__(SDTrainer)
    t.train_config = SimpleNamespace(gradient_accumulation=1, gradient_accumulation_steps=1)
    return t


def test_logging_only_on_single_batch():
    trainer = make_trainer_stub()
    batch = FakeBatch([FakeFileItem('a.jpg'), FakeFileItem('b.jpg')])
    # simulate last_example_losses for two samples
    trainer.last_example_losses = torch.tensor([1.0, 2.0])

    # call the helper with batch_list_len = 1 -> should log
    entries = trainer._maybe_log_per_example(batch, batch_list_len=1)
    assert len(entries) == 2
    assert hasattr(trainer, '_dataset_aggregator')
    # aggregator should have ds1
    report = trainer._dataset_aggregator.build_report()
    assert 'ds1' in report['datasets']

    # reset aggregator
    trainer._dataset_aggregator = None

    # call with batch_list_len = 2 -> should not log
    entries2 = trainer._maybe_log_per_example(batch, batch_list_len=2)
    assert entries2 == []
    assert trainer._dataset_aggregator is None


def test_logging_suppressed_when_accumulation_set():
    trainer = make_trainer_stub()
    trainer.train_config = SimpleNamespace(gradient_accumulation=2, gradient_accumulation_steps=1)
    batch = FakeBatch([FakeFileItem('a.jpg')])
    trainer.last_example_losses = torch.tensor([1.0])

    entries = trainer._maybe_log_per_example(batch, batch_list_len=1)
    assert entries == []
    assert not hasattr(trainer, '_dataset_aggregator')
