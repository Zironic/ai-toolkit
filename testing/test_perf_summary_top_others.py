import json
from jobs.process.BaseProcess import BaseProcess


class DummyJob:
    name = 'dummy'
    meta = {}


def test_print_top_other_timers(monkeypatch, capsys):
    # ensure print_acc prints in this test by monkeypatching it to print
    import toolkit.print as tprint

    monkeypatch.setattr(tprint, 'print_acc', print)

    bp = BaseProcess(0, DummyJob(), {'name': 'p', 'performance_log_every': 0})

    timing_dict = {
        'train_loop': 22.7371,
        'prior predict': 0.4237,
        'encode_images': 1.2345,
        'to_device': 0.5000,
        'get_batch': 0.0162,
        'preprocess_batch': 0.1000,
        'predict_unet': 0.2000,
    }

    bp._print_timer_composites(timing_dict)

    captured = capsys.readouterr().out
    assert 'PERF SUMMARY' in captured
    assert '(top:' in captured
    # ensure train_loop was excluded from the top list
    top_block = captured.split('(top:')[1].split(')')[0]
    assert 'train_loop' not in top_block
    # ensure we show up to 5 top timers and encode_images is now part of Model (so not in Other)
    assert 'encode_images' not in top_block
    assert len([s for s in top_block.split(',') if s.strip()]) <= 5

