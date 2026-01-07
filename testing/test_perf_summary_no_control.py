import json
from jobs.process.BaseProcess import BaseProcess


class DummyJob:
    name = 'dummy'
    meta = {}


def test_no_controlnet_print(monkeypatch, capsys):
    import toolkit.print as tprint
    monkeypatch.setattr(tprint, 'print_acc', print)

    bp = BaseProcess(0, DummyJob(), {'name': 'p', 'performance_log_every': 0})

    timing_dict = {
        'prior predict': 0.2510,
        'get_batch': 0.0289,
        'preprocess_batch': 0.0075,
        'batch_cleanup': 0.0042,
        'encode_prompt': 0.0005,
        # no control keys present
        'train_loop': 16.5600
    }

    bp._print_timer_composites(timing_dict)

    captured = capsys.readouterr().out
    assert 'PERF SUMMARY' in captured
    assert 'ControlNet' not in captured  # should not show ControlNet when no control timers
    assert 'Other' in captured
    assert 'total/train_loop' in captured
