import types
from toolkit.control_util import infer_expected_in_ch


def make_adapter(name):
    a = types.SimpleNamespace()
    a.name_or_path = name
    return a


def test_infer_expected_in_ch_only_uses_control_in_dim():
    a = make_adapter('alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1')
    # By default we do not perform heuristic scanning. Only an explicit control_in_dim should be honored.
    expected = infer_expected_in_ch(a)
    assert expected is None, f"Expected None when no explicit control_in_dim provided, got {expected}"


def test_infer_expected_in_ch_respects_control_in_dim():
    a = make_adapter('alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1')
    a.control_in_dim = 33
    expected = infer_expected_in_ch(a)
    assert expected == 33, f"Expected control_in_dim to be returned when present, got {expected}"


def test_ensure_control_in_dim_from_config_sets_attribute():
    from toolkit.control_util import ensure_control_in_dim
    a = types.SimpleNamespace()
    a.config = {'control_in_dim': 33}
    ok = ensure_control_in_dim(a)
    assert ok is True
    assert getattr(a, 'control_in_dim', None) == 33


def test_ensure_control_in_dim_from_in_channels_sets_attribute():
    from toolkit.control_util import ensure_control_in_dim
    a = types.SimpleNamespace()
    a.in_channels = 4
    # We no longer infer from `in_channels`; only an explicit `control_in_dim`
    # or a provided `fallback` should set the attribute.
    ok = ensure_control_in_dim(a)
    assert ok is False
    assert getattr(a, 'control_in_dim', None) is None

    # With an explicit fallback we still set the default
    ok2 = ensure_control_in_dim(a, fallback=33)
    assert ok2 is True
    assert getattr(a, 'control_in_dim', None) == 33


def test_ensure_control_in_dim_fallback_sets_default():
    from toolkit.control_util import ensure_control_in_dim
    a = types.SimpleNamespace()
    ok = ensure_control_in_dim(a, fallback=33)
    assert ok is True
    assert getattr(a, 'control_in_dim', None) == 33


def test_set_adapter_name_if_missing():
    from toolkit.control_util import set_adapter_name_if_missing
    class A:
        pass

    a = A()
    assert getattr(a, 'name_or_path', None) is None
    ok = set_adapter_name_if_missing(a, 'repo/identifier')
    assert ok is True
    assert getattr(a, 'name_or_path') == 'repo/identifier'
    assert getattr(a, '_name_forced', False) is True

    # calling again should be a no-op
    ok2 = set_adapter_name_if_missing(a, 'other')
    assert ok2 is False
    assert getattr(a, 'name_or_path') == 'repo/identifier'
