import math
from collections import deque

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.timer import Timer


def test_append_python_gap_using_step_total():
    p = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
    p.timer = Timer('p')

    # Simulate step_total_python and two measured timers
    p.timer.timers['step_total_python'] = deque([1.0], maxlen=p.timer.max_buffer)
    p.timer.timers['predict_unet'] = deque([0.3], maxlen=p.timer.max_buffer)
    p.timer.timers['preservation_backward'] = deque([0.2], maxlen=p.timer.max_buffer)

    p._append_python_gap()

    assert 'python_gap' in p.timer.timers
    assert math.isclose(p.timer.timers['python_gap'][-1], 1.0 - 0.3 - 0.2, rel_tol=1e-6)


def test_append_python_gap_uses_train_loop_if_no_step_total():
    p = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
    p.timer = Timer('p')

    # No step_total_python, fallback to train_loop
    p.timer.timers['train_loop'] = deque([2.0], maxlen=p.timer.max_buffer)
    p.timer.timers['get_batch'] = deque([0.1], maxlen=p.timer.max_buffer)
    p.timer.timers['predict_unet'] = deque([0.4], maxlen=p.timer.max_buffer)

    p._append_python_gap()

    assert 'python_gap' in p.timer.timers
    assert math.isclose(p.timer.timers['python_gap'][-1], 2.0 - 0.1 - 0.4, rel_tol=1e-6)
