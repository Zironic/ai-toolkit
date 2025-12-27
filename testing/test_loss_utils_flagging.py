from toolkit.util.loss_utils import flag_bad_captions
import math


def test_flag_bad_captions_handles_negative_losses():
    per_example = [
        {'path': 'img1.png', 'dataset': 'ds', 'caption': 'a sample', 'loss': -0.75},
        {'path': 'img2.png', 'dataset': 'ds', 'caption': '', 'loss': 2.0},
    ]

    res = flag_bad_captions(per_example)

    # Should not raise and should return a dict with flagged list and summary
    assert isinstance(res, dict)
    assert 'flagged' in res and 'summary' in res

    # check that scores are finite numbers and no math domain errors occurred
    for f in res['flagged']:
        assert math.isfinite(f.get('score', 0.0))
