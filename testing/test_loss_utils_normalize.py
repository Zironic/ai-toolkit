import json

from toolkit.util import loss_utils


def test_normalize_loss_changes_values_and_mean():
    # Two batches: first returns one item with loss 1.0, second returns two with loss 3.0 and 5.0
    batches = [1, 2]

    def compute_fn(batch):
        if batch == 1:
            return [
                {'path': 'a.jpg', 'dataset': 'ds1', 'caption': 'a', 'loss': 1.0},
            ]
        else:
            return [
                {'path': 'b.jpg', 'dataset': 'ds1', 'caption': 'b', 'loss': 3.0},
                {'path': 'c.jpg', 'dataset': 'ds1', 'caption': 'c', 'loss': 5.0},
            ]

    # Without normalization, mean should be (1+3+5)/3 = 3.0
    res_no_norm = loss_utils.evaluate_dataset(compute_fn, batches, out_json=None, normalize_loss=False)
    per_no_norm = res_no_norm['per_example']
    mean_no_norm = sum([e['loss'] for e in per_no_norm]) / len(per_no_norm)
    assert abs(mean_no_norm - 3.0) < 1e-6

    # With normalization, new mean should be 1.0 and values should be scaled
    res_norm = loss_utils.evaluate_dataset(compute_fn, batches, out_json=None, normalize_loss=True)
    per_norm = res_norm['per_example']
    mean_norm = sum([e['loss'] for e in per_norm]) / len(per_norm)
    assert abs(mean_norm - 1.0) < 1e-6
    # check scaled values: original [1,3,5] scaled by 1/3 => [0.333..., 1.0, 1.666...]
    scaled = [round(e['loss'], 6) for e in per_norm]
    assert scaled[0] == round(1.0 / 3.0, 6)
    assert scaled[1] == round(3.0 / 3.0, 6)
    assert scaled[2] == round(5.0 / 3.0, 6)
