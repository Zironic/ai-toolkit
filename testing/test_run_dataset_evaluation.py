import os
import json

from toolkit.util import loss_utils


def test_run_dataset_evaluation_writes_json(tmp_path):
    batches = [1, 2]

    def compute_fn(batch):
        if batch == 1:
            return [
                {'path': 'a.jpg', 'dataset': 'ds1', 'caption': 'a cat', 'loss': 1.0},
            ]
        else:
            return [
                {'path': 'a.jpg', 'dataset': 'ds1', 'caption': 'a cat', 'loss': 2.0},
                {'path': 'b.jpg', 'dataset': 'ds2', 'caption': 'foo', 'loss': 4.0},
            ]

    res = loss_utils.run_dataset_evaluation(compute_fn, batches, job_name='jobx', step=7, out_dir=str(tmp_path))

    expected = os.path.join(str(tmp_path), 'jobx_000000007.json')
    assert os.path.exists(expected)

    with open(expected, 'r', encoding='utf-8') as f:
        j = json.load(f)

    assert 'datasets' in j
    ds1 = j['datasets'].get('ds1')
    assert ds1 is not None
    a_stats = ds1['item_stats'].get('a.jpg')
    assert a_stats['average_loss'] == 1.5
