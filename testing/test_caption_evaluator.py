import pytest
from toolkit.util import loss_utils


def make_item(path, dataset, caption, loss):
    return {
        'path': path,
        'dataset': dataset,
        'caption': caption,
        'loss': loss,
    }


def test_aggregate_by_dataset_basic():
    items = [
        make_item('a', 'ds1', 'a cat', 1.0),
        make_item('b', 'ds1', 'a dog', 3.0),
        make_item('c', 'ds2', 'foo', 2.0),
    ]

    res = loss_utils.aggregate_by_dataset(items, top_k=2)
    assert 'dataset_summary' in res
    ds1 = res['dataset_summary']['ds1']
    assert pytest.approx(ds1['mean']) == 2.0
    assert ds1['count'] == 2
    # top_k must have first item with highest loss
    assert ds1['top_k'][0]['path'] == 'b'

    g = res['global']
    assert pytest.approx(g['mean']) == pytest.approx((1.0 + 3.0 + 2.0) / 3.0)


def test_flag_bad_captions_heuristics():
    items = [
        make_item('p1', 'd1', '', 10.0),
        make_item('p2', 'd1', 'hi', 2.0),
        make_item('p3', 'd2', 'good caption here', 0.1),
        make_item('p4', 'd2', 'bad ### caption', 5.0),
        make_item('p5', 'd2', 'wow wow wow wow wow', 4.0),
        make_item('p6', 'd3', 'token ' + ('x ' * 300), 6.0),
    ]

    flagged = loss_utils.flag_bad_captions(items, top_n=10)
    assert 'flagged' in flagged
    # expect at least the empty caption to be flagged
    empty_flagged = [f for f in flagged['flagged'] if f['path'] == 'p1']
    assert len(empty_flagged) == 1
    # suspicious token should be flagged
    susp = [f for f in flagged['flagged'] if f['path'] == 'p4']
    assert len(susp) == 1
    # repeated tokens should be flagged (p5)
    rep = [f for f in flagged['flagged'] if f['path'] == 'p5']
    assert len(rep) == 1
    # very long should be flagged (p6)
    long = [f for f in flagged['flagged'] if f['path'] == 'p6']
    assert len(long) == 1


def test_per_sample_from_loss_tensor():
    import torch
    # 4D example
    loss = torch.tensor([[[[1.0, 1.0],[1.0,1.0]]]], dtype=torch.float32)  # shape (1,1,2,2)
    per = loss_utils.per_sample_from_loss_tensor(loss)
    assert per.shape == (1,)
    assert float(per[0]) == 1.0

    # batch of two with varying values
    loss = torch.tensor([[[[1.0, 2.0],[3.0,4.0]]], [[[2.0,2.0],[2.0,2.0]]]], dtype=torch.float32)  # (2,1,2,2)
    per = loss_utils.per_sample_from_loss_tensor(loss)
    assert per.shape == (2,)
    assert pytest.approx(per[0].item(), rel=1e-6) == 2.5
    assert pytest.approx(per[1].item(), rel=1e-6) == 2.0


def test_per_example_from_batch():
    class Dummy:
        def __init__(self, path, dataset_config, raw_caption):
            self.path = path
            self.dataset_config = dataset_config
            self.raw_caption = raw_caption

    class DummyDatasetConfig:
        def __init__(self, dataset_path=None, folder_path=None, default_caption=None):
            self.dataset_path = dataset_path
            self.folder_path = folder_path
            self.default_caption = default_caption

    class Batch:
        def __init__(self, file_items):
            self.file_items = file_items

    import torch
    fi1 = Dummy('p1', DummyDatasetConfig(dataset_path='ds1'), 'cat')
    fi2 = Dummy('p2', DummyDatasetConfig(dataset_path='ds2'), 'dog')
    batch = Batch([fi1, fi2])

    losses = torch.tensor([1.5, 2.5])
    recs = loss_utils.per_example_from_batch(batch, losses)
    assert len(recs) == 2
    assert recs[0]['path'] == 'p1'
    assert recs[0]['loss'] == pytest.approx(1.5)

    # doubled
    losses = torch.tensor([1.0, 1.1, 2.0, 2.1])
    recs = loss_utils.per_example_from_batch(batch, losses)
    assert len(recs) == 4
    assert recs[2]['path'] == 'p1'


if __name__ == '__main__':
    pytest.main([__file__])
