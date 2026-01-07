import time
from toolkit.timer import Timer


def test_nested_timers_do_not_error():
    t = Timer('test', max_buffer=5)
    # Use nested contexts; should not raise and should record both timers
    with t('outer'):
        with t('inner'):
            pass

    # both timers should have recorded entries
    assert 'outer' in t.timers
    assert 'inner' in t.timers


def test_timer_context_records_duration():
    t = Timer('test', max_buffer=5)
    with t('a'):
        time.sleep(0.001)
    with t('b'):
        time.sleep(0.001)

    # averages should be non-zero (small but > 0)
    assert sum(t.timers['a']) > 0
    assert sum(t.timers['b']) > 0
