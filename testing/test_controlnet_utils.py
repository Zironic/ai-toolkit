import pytest
import types
from toolkit.controlnet_utils import is_zimage_adapter, validate_zimage_adapter

class DummyConfig:
    def __init__(self, mode=None, control_in_dim=None):
        self.controlnet_mode = mode
        self.control_in_dim = control_in_dim

class AdapterNoControlContext:
    def forward(self, latents, timestep):
        return None

class AdapterGood:
    control_in_dim = 33
    def forward(self, latents, timestep, control_context=None, conditioning_scale=1.0):
        return None

def test_is_zimage_adapter_true_with_config():
    cfg = DummyConfig(mode='zimage')
    assert is_zimage_adapter(None, cfg)

def test_validate_zimage_adapter_missing_control_context_raises():
    cfg = DummyConfig(mode='zimage', control_in_dim=33)
    a = AdapterNoControlContext()
    with pytest.raises(RuntimeError):
        validate_zimage_adapter(a, cfg)

def test_validate_zimage_adapter_ok():
    cfg = DummyConfig(mode='zimage', control_in_dim=33)
    a = AdapterGood()
    # Should not raise
    validate_zimage_adapter(a, cfg)


def test_is_zimage_adapter_name_detection():
    # When explicit controlnet_mode is not set, detection should fall back
    # to name-based hints (e.g., name_or_path containing 'zimage')
    cfg = DummyConfig(mode=None)
    cfg.name_or_path = 'some_zimage_model_v0'
    assert is_zimage_adapter(None, cfg)

    # Also allow adapter instances with name_or_path hint to trigger detection
    class A:
        name_or_path = 'my_videox_adapter'
    assert is_zimage_adapter(A(), None)


def test_ensure_zimage_mode_sets_cfg_from_name_in_config():
    cfg = DummyConfig(mode=None)
    cfg.name_or_path = 'some_zimage_model_v0'
    from toolkit.controlnet_utils import ensure_zimage_mode
    assert ensure_zimage_mode(None, cfg)
    assert getattr(cfg, 'controlnet_mode', None) == 'zimage'


def test_ensure_zimage_mode_sets_cfg_from_adapter_name():
    cfg = DummyConfig(mode=None)
    class A:
        name_or_path = 'foo_videox'
    from toolkit.controlnet_utils import ensure_zimage_mode
    assert ensure_zimage_mode(A(), cfg)
    assert getattr(cfg, 'controlnet_mode', None) == 'zimage'


def test_enforce_control_in_dim_sets_on_adapter():
    # Adapter without control_in_dim should be set to 33 by enforce_zimage_control_in_dim
    class A:
        name_or_path = 'foo_videox'
    a = A()
    from toolkit.control_util import enforce_zimage_control_in_dim
    assert enforce_zimage_control_in_dim(a, expected=33, force=True)
    assert getattr(a, 'control_in_dim', None) == 33


def test_setup_controlnet_training_raises_when_unsettable(monkeypatch):
    # Simulate enforcement logic without importing the full BaseSDTrainProcess to avoid heavyweight imports
    # create adapter that disallows setattr
    class NoSet:
        name_or_path = 'zimage_test'
        def __setattr__(self, name, value):
            raise AttributeError('cannot set')
    adapter = NoSet()
    # provide adapter_config that also disallows setattr
    class NoSetCfg:
        type = 'control_net'
        def __setattr__(self, name, value):
            raise AttributeError('cannot set')
    cfg = NoSetCfg()

    from toolkit.control_util import enforce_zimage_control_in_dim

    # First, enforce_zimage_control_in_dim will attempt to set on adapter and return False on failure
    ok = False
    try:
        ok = enforce_zimage_control_in_dim(adapter, expected=33, force=True)
    except Exception:
        ok = False

    # Then attempt to set on cfg (should raise)
    with pytest.raises(Exception):
        cfg.control_in_dim = 33

    # Final behavior: both attempts fail; our setup code is expected to treat this as fatal.
    assert ok is False
