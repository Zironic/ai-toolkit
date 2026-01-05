import sys
import pytest
from types import SimpleNamespace

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


def test_setup_controlnet_print_failure_raises(monkeypatch):
    # Create a minimal instance without running __init__ to avoid heavy setup
    proc = BaseSDTrainProcess.__new__(BaseSDTrainProcess)

    # Minimal sd object to trigger early branch and print path
    proc.sd = SimpleNamespace(is_controlnet_enabled=True, controlnet=object())
    # Ensure no print_and_status_update attribute so the fallback print_acc path is used
    if hasattr(proc, 'print_and_status_update'):
        delattr(proc, 'print_and_status_update')

    # Monkeypatch toolkit.print.print_acc to raise an error
    import toolkit.print as tprint

    def boom(msg):
        raise RuntimeError("boom")

    monkeypatch.setattr(tprint, 'print_acc', boom)

    with pytest.raises(RuntimeError) as excinfo:
        proc.setup_controlnet_training()

    assert 'Failed to emit ControlNet setup status' in str(excinfo.value) or 'Failed to print' in str(excinfo.value)
