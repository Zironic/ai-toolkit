import json
import os
from toolkit.util import loss_utils


def test_streaming_aggregator_basic(tmp_path):
    agg = loss_utils.StreamingAggregator()
    items = [
        {'path': 'a.jpg', 'dataset': 'ds1', 'caption': 'a cat', 'loss': 1.0},
        {'path': 'b.jpg', 'dataset': 'ds1', 'caption': 'a dog', 'loss': 3.0},
        {'path': 'a.jpg', 'dataset': 'ds1', 'caption': 'a cat', 'loss': 2.0},
        {'path': 'c.jpg', 'dataset': 'ds2', 'caption': 'foo', 'loss': 4.0},
    ]
    for e in items:
        agg.add_entry(e)

    report = agg.build_report(eval_config={'model': 'x'})
    assert 'datasets' in report
    ds1 = report['datasets'].get('ds1')
    assert ds1 is not None
    a_stats = ds1['item_stats'].get('a.jpg')
    assert a_stats['average_loss'] == 1.5
    assert a_stats['min_loss'] == 1.0
    assert a_stats['max_loss'] == 2.0

    # write to file
    out_json = str(tmp_path / 'agg.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    assert os.path.exists(out_json)
