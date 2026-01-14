import pytest
from types import SimpleNamespace
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


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


def test_is_dop_scheduled_behavior():
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # enable DOP and set start-after=2
    trainer.train_config.diff_output_preservation = True
    trainer.train_config.diff_output_preservation_after_steps = 2

    # starting with total count 0: encoding decision should look at (0 + 1) >= 2 -> False
    trainer._total_batch_count = 0
    assert trainer._is_dop_scheduled(for_encoding=True) is False
    # runtime decision (after increment) should be 0 >= 2 -> False
    assert trainer._is_dop_scheduled(for_encoding=False) is False

    # when total_count == 1: encoding (1+1) >= 2 -> True
    trainer._total_batch_count = 1
    assert trainer._is_dop_scheduled(for_encoding=True) is True
    assert trainer._is_dop_scheduled(for_encoding=False) is False

    # when total_count == 2: both encoding and runtime are True
    trainer._total_batch_count = 2
    assert trainer._is_dop_scheduled(for_encoding=True) is True
    assert trainer._is_dop_scheduled(for_encoding=False) is True

    # general sanity checks for start-after=0 (run immediately)
    trainer.train_config.diff_output_preservation_after_steps = 0
    trainer._total_batch_count = 0
    assert trainer._is_dop_scheduled(for_encoding=True) is True
    assert trainer._is_dop_scheduled(for_encoding=False) is True


def test_invalid_every_raises():
    job, cfg = make_job_and_cfg()
    # configure invalid value before constructing trainer to exercise TrainConfig validation
    cfg['train']['diff_output_preservation'] = True
    cfg['train']['diff_output_preservation_every'] = 0
    with pytest.raises(ValueError):
        SDTrainer(0, job, cfg)


def test_is_dop_scheduled_disabled():
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)
    trainer.train_config.diff_output_preservation = False
    trainer.train_config.diff_output_preservation_every = 2
    trainer._total_batch_count = 0
    assert trainer._is_dop_scheduled(for_encoding=True) is False
    assert trainer._is_dop_scheduled(for_encoding=False) is False
