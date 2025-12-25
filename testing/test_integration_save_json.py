import json
import os
from types import SimpleNamespace

import pytest

from toolkit.util import loss_utils
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


def make_minimal_process(tmp_path):
    """Construct a minimal BaseSDTrainProcess-like object (without running __init__) and set
    attributes required for the save() code path we want to test.
    """
    proc = BaseSDTrainProcess.__new__(BaseSDTrainProcess)

    # minimal accelerator stub
    proc.accelerator = SimpleNamespace(is_main_process=True)

    proc.ema = None
    proc.save_root = str(tmp_path)

    # job.name used to build filename
    proc.job = SimpleNamespace(name='testjob')

    # meta used as eval_config when building report
    proc.meta = {'training_info': {'step': 0}}

    # avoid doing heavy work in update/cleanup hooks
    proc.update_training_metadata = lambda: None
    proc.clean_up_saves = lambda: None
    proc.post_save_hook = lambda x: None

    # mark as not fine_tuning so saving branches do not try to save network components
    proc.is_fine_tuning = False

    # no optimizer to save
    proc.optimizer = None

    # ensure attributes referenced by save() exist and are harmless
    proc.adapter = None
    proc.decorator = None
    proc.embedding = None
    proc.network = None
    proc.network_config = None
    proc.snr_gos = None
    proc.save_config = SimpleNamespace(dtype='float16')

    # ensure we will attempt to write JSON
    proc.train_config = SimpleNamespace(save_loss_json=True)

    return proc


def test_save_step_writes_json(tmp_path):
    proc = make_minimal_process(tmp_path)

    # build a streaming aggregator and populate with entries
    agg = loss_utils.StreamingAggregator()
    entries = [
        {'path': 'a.jpg', 'dataset': 'ds1', 'caption': 'a cat', 'loss': 1.0},
        {'path': 'b.jpg', 'dataset': 'ds1', 'caption': 'a dog', 'loss': 3.0},
        {'path': 'a.jpg', 'dataset': 'ds1', 'caption': 'a cat', 'loss': 2.0},
        {'path': 'c.jpg', 'dataset': 'ds2', 'caption': 'foo', 'loss': 4.0},
    ]
    for e in entries:
        agg.add_entry(e)

    proc._dataset_aggregator = agg

    # call save with a specific step
    proc.save(step=123)

    # filename s.b. testjob_000000123.json
    expected = os.path.join(str(tmp_path), f"{proc.job.name}_{str(123).zfill(9)}.json")
    assert os.path.exists(expected), f"Expected JSON report at {expected}"

    with open(expected, 'r', encoding='utf-8') as f:
        j = json.load(f)

    assert 'datasets' in j
    ds1 = j['datasets'].get('ds1')
    assert ds1 is not None
    a_stats = ds1['item_stats'].get('a.jpg')
    assert a_stats is not None
    assert a_stats['average_loss'] == 1.5
    assert a_stats['min_loss'] == 1.0
    assert a_stats['max_loss'] == 2.0

    # dataset ds2 must exist and have c.jpg stats
    ds2 = j['datasets'].get('ds2')
    assert ds2 is not None
    c_stats = ds2['item_stats'].get('c.jpg')
    assert c_stats['average_loss'] == 4.0
    assert c_stats['min_loss'] == 4.0
    assert c_stats['max_loss'] == 4.0
