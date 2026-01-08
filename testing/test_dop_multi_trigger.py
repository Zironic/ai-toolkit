import pytest
from types import SimpleNamespace
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.prompt_utils import parse_csv_list


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


def test_parse_csv_list_basics():
    assert parse_csv_list(None) == []
    assert parse_csv_list("") == []
    assert parse_csv_list("Jinx") == ["Jinx"]
    assert parse_csv_list("Jinx, Zapper") == ["Jinx", "Zapper"]
    assert parse_csv_list("a, , c") == ["a", "", "c"]


def test_single_trigger_behaviour():
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    trainer.train_config.diff_output_preservation = True
    trainer.trigger_word = "Jinx"
    trainer.train_config.diff_output_preservation_class = "Woman"
    # rebuild mapping (normally in __init__); mirror __init__ logic
    trainer._dop_replacements = [("Jinx", "Woman")]

    out = trainer._map_triggers_to_classes_in_text("A Jinx portrait with Jinx")
    assert "Woman" in out
    assert "Jinx" not in out


def test_multiple_triggers_pairwise_mapping():
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    trainer.train_config.diff_output_preservation = True
    trainer.trigger_word = "Jinx, Zapper"
    trainer.train_config.diff_output_preservation_class = "Woman, Gun"
    trainer._dop_replacements = [("Zapper", "Gun"), ("Jinx", "Woman")]

    text = "Jinx with a Zapper"
    out = trainer._map_triggers_to_classes_in_text(text)
    assert "Woman" in out and "Gun" in out
    assert "Jinx" not in out and "Zapper" not in out


def test_more_triggers_than_classes():
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    trainer.trigger_word = "Jinx, Zapper, Vest"
    trainer.train_config.diff_output_preservation_class = "Woman, Gun"
    trainer._dop_replacements = [("Jinx", "Woman"), ("Zapper", "Gun"), ("Vest", "")]

    out = trainer._map_triggers_to_classes_in_text("Jinx Zapper Vest")
    assert "Woman" in out and "Gun" in out
    assert "Vest" not in out


def test_overlapping_triggers_longest_first():
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    trainer.trigger_word = "Jinx, Jinx Master"
    trainer.train_config.diff_output_preservation_class = "Woman, Veteran"
    # ensure longest-first ordering
    trainer._dop_replacements = [("Jinx Master", "Veteran"), ("Jinx", "Woman")]

    out = trainer._map_triggers_to_classes_in_text("Jinx Master and Jinx")
    assert out == "Veteran and Woman"


def test_case_sensitivity_default():
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    trainer.trigger_word = "Jinx"
    trainer.train_config.diff_output_preservation_class = "Woman"
    trainer._dop_replacements = [("Jinx", "Woman")]

    out = trainer._map_triggers_to_classes_in_text("jinx lowercase")
    # default is case-sensitive, so lowercase should remain unchanged
    assert "jinx" in out and "Woman" not in out
