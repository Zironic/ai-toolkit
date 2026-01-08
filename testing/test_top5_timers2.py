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


def test_after_unet_predict_timer_recorded():
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)
    trainer.after_unet_predict()
    assert 'after_unet_predict' in trainer.timer.timers


def test_process_general_training_batch_timer_recorded():
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)
    try:
        trainer.process_general_training_batch(None)
    except Exception:
        pass
    assert 'process_general_training_batch' in trainer.timer.timers
