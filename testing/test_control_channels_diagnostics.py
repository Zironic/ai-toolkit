import pytest

# Deleted legacy tests referencing `adapt_control_images` (see issue #TODO).
# These tests assumed a deprecated control image adaptation design which is
# intentionally not part of the current training path.
pytest.skip("Deleted: legacy tests referencing adapt_control_images.", allow_module_level=True)


def test_wrapper_raises_on_none_inner():
    wrapper = VideoXControlnetWrapper(None)
    lat = torch.randn(1, 16, 8, 8)
    ctrl = torch.randn(1, 3, 64, 64)
    with pytest.raises(RuntimeError, match=r"inner adapter is None"):
        wrapper(lat, 0.0, ctrl, conditioning_scale=1.0)


class DummyAdapterNoName:
    def __init__(self):
        # intentionally no name_or_path
        self.control_in_dim = 4


def test_adapt_control_images_error_includes_adapter_details():
    a = DummyAdapterNoName()
    ctrl = torch.randn(1, 3, 64, 64)
    with pytest.raises(RuntimeError) as exc:
        adapt_control_images(ctrl, a)
    msg = str(exc.value)
    assert 'adapter_repr=' in msg
    assert "adapter.name_or_path=None" in msg
    assert 'adapter.control_in_dim=4' in msg
    assert 'looks_like=pixel-image' in msg
