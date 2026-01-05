import pytest
from types import SimpleNamespace

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


def test_videox_loader_reports_import_failure(monkeypatch):
    proc = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
    proc.name = 'test_job'
    proc.network_config = None
    proc.get_latest_save_path = lambda name: None
    proc.device_torch = None
    proc.sd = SimpleNamespace()

    tcfg = SimpleNamespace()
    tcfg.dtype = 'float32'
    proc.train_config = tcfg

    # Make the vendored module import raise ImportError when imported within the loader
    import sys
    import types

    fake_mod = types.ModuleType('extensions_built_in.diffusion_models.z_image_transformer2d_control')
    def bad_import(*a, **k):
        raise ImportError('missing-dep')

    # Ensure import inside load_videox_control_adapter triggers ImportError
    monkeypatch.setitem(sys.modules, 'extensions_built_in.diffusion_models.z_image_transformer2d_control', fake_mod)
    monkeypatch.setattr(fake_mod, 'ZImageControlTransformer2DModel', None, raising=False)

    from extensions_built_in.diffusion_models import z_image_adapter as za

    with pytest.raises(RuntimeError) as excinfo:
        za.load_videox_control_adapter(name_or_path=None)
    msg = str(excinfo.value)
    assert ('Failed to import vendored VideoX adapter' in msg) or ("missing callable 'ZImageControlTransformer2DModel'" in msg)
