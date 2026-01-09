import pytest
try:
    import torch
except Exception:
    pytest.skip("torch import failed in this environment", allow_module_level=True)

from toolkit.device_utils import _cast_and_move, _maybe_log_cast


def test_cast_and_move_tensor_dtype_and_device():
    t = torch.tensor([1.0], dtype=torch.float32)
    t2 = _cast_and_move(t, torch.device('cpu'), dtype=torch.float64)
    assert isinstance(t2, torch.Tensor)
    assert t2.dtype == torch.float64


def test_maybe_log_cast_logs_when_dtype_differs(monkeypatch):
    recorded = {}
    def fake_print_acc(msg):
        recorded['msg'] = msg

    # monkeypatch the module-level print_acc used by device_utils
    monkeypatch.setattr('toolkit.device_utils.print_acc', fake_print_acc)

    t = torch.tensor([1.0], dtype=torch.float32)
    _maybe_log_cast(t, torch.float64, 'test_tensor')
    assert 'casting test_tensor' in recorded.get('msg', ''), f"Unexpected log: {recorded.get('msg')!r}"

    # When dtype is None or matches, nothing should be logged
    recorded.clear()
    _maybe_log_cast(t, None, 'test_tensor')
    assert recorded == {}

    recorded.clear()
    _maybe_log_cast(t, torch.float32, 'test_tensor')
    assert recorded == {}
