from types import SimpleNamespace
import pytest


def make_minimal_process_with_zimage(BaseSDTrainProcess):
    p = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
    p.sd = SimpleNamespace()
    p.sd.is_controlnet_enabled = False
    p.sd.controlnet = None
    p.sd.torch_dtype = None

    # make model_config with zimage hint
    p.model_config = SimpleNamespace()
    p.model_config.controlnet_name_or_path = 'owner/zimage-model'
    p.model_config.controlnet_file = None
    p.model_config.controlnet_offload_strategy = 'none'

    p.train_config = SimpleNamespace()
    p.print_and_status_update = lambda msg: None
    return p


def test_setup_fails_if_zimage_model_requires_predict_hook(monkeypatch):
    # Import BaseSDTrainProcess lazily to avoid importing heavy torch internals at collection time
    try:
        from jobs.process import BaseSDTrainProcess
    except Exception:
        pytest.skip("Skipping: environment cannot import BaseSDTrainProcess (torch import issue)")

    proc = make_minimal_process_with_zimage(BaseSDTrainProcess)

    def fake_prepare(*args, **kwargs):
        # Simulate deterministic loader raising on require_zimage_model
        raise RuntimeError('Z-Image model missing required _predict_noise_zimage')

    monkeypatch.setattr('toolkit.control_util.prepare_controlnet_adapter', fake_prepare)

    with pytest.raises(RuntimeError) as exc:
        proc.setup_controlnet_training()
    assert 'Z-Image' in str(exc.value) or '_predict_noise_zimage' in str(exc.value)
