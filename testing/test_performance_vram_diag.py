import pytest
from jobs.process.BaseProcess import BaseProcess


class DummyJob:
    name = 'dummy'
    meta = {}


def test_performance_vram_default_disabled():
    bp = BaseProcess(0, DummyJob(), {'name': 'p', 'performance_log_every': 0})
    # default precise_gpu_timing should be False
    assert bp.timer.gpu_timing_enabled is False


def test_performance_vram_diag_hook_enabled():
    cfg = {'name': 'p', 'performance_log_every': 0, 'performance': {'precise_gpu_timing': False, 'vram_diagnostics': {'enabled': True, 'once_per_run': True}}}
    bp = BaseProcess(0, DummyJob(), cfg)
    # ensure hook does not raise when printing (best-effort, no GPU required)
    # call the hook directly with an empty timing dict
    bp._print_vram_diagnostics({})
    # after printing once, subsequent calls should be no-op if once_per_run is True
    bp._print_vram_diagnostics({})
