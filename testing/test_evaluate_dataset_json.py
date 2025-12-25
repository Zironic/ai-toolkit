import json
import os
from pathlib import Path

from toolkit.util import loss_utils


def test_evaluate_dataset_writes_json(tmp_path):
    # build a fake dataloader (list of dummy batches)
    batches = [1, 2]

    def compute_fn(batch):
        # return per-sample entries; batch value determines content
        if batch == 1:
            return [
                {'path': 'a.jpg', 'dataset': 'ds1', 'caption': 'a cat', 'loss': 1.0},
                {'path': 'b.jpg', 'dataset': 'ds1', 'caption': 'a dog', 'loss': 3.0},
            ]
        else:
            return [
                {'path': 'a.jpg', 'dataset': 'ds1', 'caption': 'a cat', 'loss': 2.0},
                {'path': 'c.jpg', 'dataset': 'ds2', 'caption': 'foo', 'loss': 4.0},
            ]

    out_json = str(tmp_path / 'report.json')
    cfg = {'model': 'test-model', 'resolution': 512}
    res = loss_utils.evaluate_dataset(compute_fn, batches, out_json=out_json, eval_config=cfg)

    # check returned json_report is not None and file exists
    assert res['json_report'] is not None
    assert os.path.exists(out_json)

    with open(out_json, 'r', encoding='utf-8') as f:
        j = json.load(f)

    assert 'config' in j and j['config']['model'] == 'test-model'
    assert 'datasets' in j
    ds1 = j['datasets'].get('ds1')
    assert ds1 is not None
    # a.jpg should have avg = (1+2)/2 = 1.5, min=1.0, max=2.0
    a_stats = ds1['item_stats'].get('a.jpg')
    assert a_stats is not None
    assert a_stats['average_loss'] == 1.5
    assert a_stats['min_loss'] == 1.0
    assert a_stats['max_loss'] == 2.0
