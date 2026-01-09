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


def test_dop_debug_prints(capsys):
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)
    trainer.train_config.diff_output_preservation_debug = True
    trainer._dop_replacements = [("Jinx", ""), ("Zapper", "Gun")]

    out = trainer._map_triggers_to_classes_in_text("Jinx with a Zapper")
    captured = capsys.readouterr()
    assert "[DOP DEBUG]" in captured.out
    assert "Jinx with a Zapper" in captured.out
    assert "with a Gun" in captured.out
    assert out == "with a Gun"