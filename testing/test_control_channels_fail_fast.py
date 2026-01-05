import torch
import pytest
import types

import toolkit.control_channels as cc


def test_record_meta_failure_raises(monkeypatch):
    # Force _tensor_meta __setitem__ to raise
    class BrokenDict(dict):
        def __setitem__(self, k, v):
            raise ValueError("boom")

    monkeypatch.setattr(cc, '_tensor_meta', BrokenDict())
    t = torch.zeros((1,3,4,4))
    with pytest.raises(RuntimeError) as excinfo:
        cc._record_meta(t, {'op': 'test'})
    assert 'Failed to record tensor metadata' in str(excinfo.value)


def test_tag_tensor_failure_raises(monkeypatch):
    # Monkeypatch _record_meta to raise
    def bad_record(t, m):
        raise RuntimeError("broken-record")
    monkeypatch.setattr(cc, '_record_meta', bad_record)
    t = torch.zeros((1,3,4,4))
    with pytest.raises(RuntimeError) as excinfo:
        cc.tag_tensor(t, 'op')
    assert 'Failed to tag tensor metadata' in str(excinfo.value)
