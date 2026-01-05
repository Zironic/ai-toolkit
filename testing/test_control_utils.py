from types import SimpleNamespace
from toolkit.control_util import adapter_uses_zimage


def test_adapter_detects_mode_flag():
    cfg = SimpleNamespace()
    cfg.controlnet_mode = 'zimage'
    assert adapter_uses_zimage(None, cfg)


def test_adapter_detects_name_in_config():
    cfg = SimpleNamespace()
    cfg.controlnet_mode = None
    cfg.name_or_path = 'user/zimage-awesome'
    assert adapter_uses_zimage(None, cfg)


def test_adapter_detects_loaded_adapter_name():
    cfg = SimpleNamespace()
    cfg.controlnet_mode = None
    cfg.name_or_path = None
    class Dummy:
        pass
    a = Dummy()
    a.name_or_path = 'foo/pipeline_z_image_v2'
    assert adapter_uses_zimage(a, cfg)


def test_adapter_detects_class_name_hint():
    cfg = SimpleNamespace()
    cfg.controlnet_mode = None
    cfg.name_or_path = None
    class ZImageAdapter:
        pass
    a = ZImageAdapter()
    assert adapter_uses_zimage(a, cfg)


def test_adapter_negative_case():
    cfg = SimpleNamespace()
    cfg.controlnet_mode = None
    cfg.name_or_path = 'some_other_model'
    class Other:
        pass
    a = Other()
    assert not adapter_uses_zimage(a, cfg)