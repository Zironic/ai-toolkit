import pytest
try:
    import torch
except Exception:
    pytest.skip("Skipping: PyTorch import failed in this environment", allow_module_level=True)

from toolkit.stable_diffusion_model import StableDiffusion


def _make_sd_with_dummy_transformer():
    sd = StableDiffusion.__new__(StableDiffusion)

    # Simple no-op timer context used by the method
    class DummyTimer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    sd.timer = DummyTimer()

    class DummyTransformer:
        def __call__(self, latent_list, tstep, te, **kwargs):
            # return a tuple whose first element is a list of per-sample tensors
            return ([torch.zeros_like(latent_list[0]) for _ in range(len(latent_list))],)

    sd.transformer = DummyTransformer()
    # minimal attributes used by predict_noise_zimage
    sd.torch_dtype = torch.float32
    sd.device_torch = torch.device('cpu')
    sd.model_config = type('C', (), {})()
    return sd


def test_local_debug_quick_path_records_no_residuals():
    sd = _make_sd_with_dummy_transformer()

    B, C, H, W = 1, 4, 32, 32
    latents = torch.zeros((B, C, H, W))
    text_embeds = torch.zeros((B, 77, 768))
    timestep = torch.tensor([10.0])

    # Calling model-side routing without control info is invalid; assert it raises
    with pytest.raises(RuntimeError):
        sd._predict_noise_zimage(latents=latents, text_embeddings=text_embeds, timestep=timestep)

    # Local debug info, if set, should indicate absence of residuals (non-mandatory)
    info = getattr(sd, '_last_zimage_local_debug', None)
    if info is not None:
        assert info.get('has_down') is False
        assert info.get('num_down', 0) == 0
        assert info.get('has_mid') is False


def test_local_debug_main_path_records_residuals():
    sd = _make_sd_with_dummy_transformer()

    # Provide a DummyTimer that records start/stop
    class DummyTimer:
        def __init__(self):
            self.started = {}
            self.stopped = {}
            self.timers = {}
            self._start_ts = {}

        def start(self, name):
            # mark started
            self.started[name] = True
            self._start_ts[name] = 0.0  # fake

        def stop(self, name):
            # mark stopped and record a fake duration
            self.stopped[name] = True
            self.timers.setdefault(name, []).append(0.0)

        def __call__(self, name):
            # context manager
            class Ctx:
                def __init__(self, timer, name):
                    self.timer = timer
                    self.name = name
                def __enter__(self):
                    self.timer.start(self.name)
                    return self.timer
                def __exit__(self, exc_type, exc, tb):
                    self.timer.stop(self.name)
                    return False
            return Ctx(self, name)

    sd.timer = DummyTimer()

    B, C, H, W = 1, 4, 32, 32
    latents = torch.zeros((B, C, H, W))
    text_embeds = torch.zeros((B, 77, 768))
    timestep = torch.tensor([10.0])

    # Provide a precomputed control_context (with frame dim)
    control_ctx = torch.zeros((B, 33, 1, H, W))

    # Dummy ControlNet that returns a single tensor (will be interpreted as down residuals)
    called = {}

    class DummyControlNet:
        def __call__(self, sample_for_adapter, timestep_in, control_context_in, conditioning_scale=None):
            called['args'] = (sample_for_adapter, timestep_in, control_context_in)
            # Return a tensor (interpreted as a single down residual)
            return torch.zeros((1, 16, 1, H, W))

    zcn = DummyControlNet()

    out = sd._predict_noise_zimage(latents=latents, text_embeddings=text_embeds, timestep=timestep, zimage_controlnet=zcn, zimage_control_context=control_ctx)

    # Ensure adapter was called and local debug indicates down residuals
    assert 'args' in called
    info = getattr(sd, '_last_zimage_local_debug', None)
    assert info is not None
    assert info.get('has_down') is True
    assert info.get('num_down', 0) == 1
    assert isinstance(info.get('down_shapes'), list)
    assert info.get('has_mid') is False

    # Timer assertions: controlnet_forward_io should have been started/stopped, and controlnet_zimage_forward also stopped
    assert sd.timer.started.get('controlnet_forward_io', False) is True
    assert sd.timer.stopped.get('controlnet_forward_io', False) is True
    assert sd.timer.stopped.get('controlnet_zimage_forward', False) is True


def test_dataset_debug_flag_enables_diagnostics():
    sd = _make_sd_with_dummy_transformer()

    class DummyTimer:
        def __init__(self):
            self.started = {}
            self.stopped = {}
            self.timers = {}
            self._start_ts = {}

        def start(self, name):
            # mark started
            self.started[name] = True
            self._start_ts[name] = 0.0  # fake

        def stop(self, name):
            # mark stopped and record a fake duration
            self.stopped[name] = True
            self.timers.setdefault(name, []).append(0.0)

        def __call__(self, name):
            # context manager
            class Ctx:
                def __init__(self, timer, name):
                    self.timer = timer
                    self.name = name
                def __enter__(self):
                    self.timer.start(self.name)
                    return self.timer
                def __exit__(self, exc_type, exc, tb):
                    self.timer.stop(self.name)
                    return False
            return Ctx(self, name)

    sd.timer = DummyTimer()

    B, C, H, W = 1, 4, 32, 32
    latents = torch.zeros((B, C, H, W))
    text_embeds = torch.zeros((B, 77, 768))
    timestep = torch.tensor([10.0])

    control_ctx = torch.zeros((B, 33, 1, H, W))

    called = {}

    class DummyControlNet:
        def __call__(self, sample_for_adapter, timestep_in, control_context_in, conditioning_scale=None):
            called['args'] = (sample_for_adapter, timestep_in, control_context_in)
            # Return a tensor (interpreted as a single down residual)
            return torch.zeros((1, 16, 1, H, W))

    zcn = DummyControlNet()

    # Pass dataset_controlnet_debug via kwargs as the trainer would
    out = sd._predict_noise_zimage(latents=latents, text_embeddings=text_embeds, timestep=timestep, zimage_controlnet=zcn, zimage_control_context=control_ctx, dataset_controlnet_debug=True)

    # Ensure adapter debug detail was recorded
    detail = getattr(sd, '_last_zimage_control_debug_detail', None)
    assert detail is not None
    assert 'adapter_name' in detail
    assert isinstance(detail.get('down_shapes'), list)
    assert detail.get('mid_shape') is None

    # Observability: entry timestamp and adapter call count recorded
    assert getattr(sd, '_last_zimage_entry_ts', None) is not None
    assert getattr(sd, '_last_zimage_adapter_called', False) is True
    assert getattr(sd, '_last_zimage_adapter_call_count', 0) >= 1


def test_timer_recorded_quick_path():
    sd = _make_sd_with_dummy_transformer()

    class DummyTimer2:
        def __init__(self):
            self.started = {}
            self.stopped = {}
            self.timers = {}
        def start(self, name):
            self.started[name] = True
        def stop(self, name):
            self.stopped[name] = True
            self.timers.setdefault(name, []).append(0.0)
        def __call__(self, name):
            class Ctx:
                def __init__(self, timer, name):
                    self.timer = timer
                    self.name = name
                def __enter__(self):
                    self.timer.start(self.name)
                    return self.timer
                def __exit__(self, exc_type, exc, tb):
                    self.timer.stop(self.name)
                    return False
            return Ctx(self, name)

    sd.timer = DummyTimer2()

    B, C, H, W = 1, 4, 32, 32
    latents = torch.zeros((B, C, H, W))
    text_embeds = torch.zeros((B, 77, 768))
    timestep = torch.tensor([10.0])

    # Calling model-side routing without control info should raise
    with pytest.raises(RuntimeError):
        sd._predict_noise_zimage(latents=latents, text_embeddings=text_embeds, timestep=timestep)

    # Ensure control I/O timer was started and stopped even on the error path
    assert sd.timer.started.get('controlnet_forward_io', False) is True
    assert sd.timer.stopped.get('controlnet_forward_io', False) is True


def test_adapter_without_control_images_raises():
    sd = _make_sd_with_dummy_transformer()

    class DummyTimer:
        def __init__(self):
            self.started = {}
            self.stopped = {}
            self.timers = {}
        def start(self, name):
            self.started[name] = True
        def stop(self, name):
            self.stopped[name] = True
            self.timers.setdefault(name, []).append(0.0)
        def __call__(self, name):
            class Ctx:
                def __init__(self, timer, name):
                    self.timer = timer
                    self.name = name
                def __enter__(self):
                    self.timer.start(self.name)
                    return self.timer
                def __exit__(self, exc_type, exc, tb):
                    self.timer.stop(self.name)
                    return False
            return Ctx(self, name)

    sd.timer = DummyTimer()

    B, C, H, W = 1, 4, 32, 32
    latents = torch.zeros((B, C, H, W))
    text_embeds = torch.zeros((B, 77, 768))
    timestep = torch.tensor([10.0])

    # Dummy ControlNet present but no control images/context
    class DummyControlNet:
        def __call__(self, *a, **k):
            return torch.zeros((1, 16, 1, H, W))

    zcn = DummyControlNet()

    with pytest.raises(RuntimeError):
        sd._predict_noise_zimage(latents=latents, text_embeddings=text_embeds, timestep=timestep, zimage_controlnet=zcn)

    assert sd.timer.started.get('controlnet_forward_io', False) is True
    assert sd.timer.stopped.get('controlnet_forward_io', False) is True

