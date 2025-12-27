import torch
from toolkit.controlnet_offload import compute_control_residuals


class DummyAdapter:
    def __init__(self, scales=3, out_channels=[64, 128, 256]):
        self.scales = scales
        self.out_channels = out_channels
        self._device = torch.device("cpu")
        self.training = False

    def parameters(self):
        # Emulate a tiny parameter on cpu
        # Return as an iterator (generator) so `next()` works
        def _iter():
            yield torch.nn.Parameter(torch.zeros(1))
        return _iter()

    def eval(self):
        pass

    def train(self, mode=True):
        # stub to satisfy compute_control_residuals toggling
        self._training = bool(mode)
    def __call__(self, control_tensor, noisy_latents, timesteps):
        # Return a tuple of tensors representing per-scale residuals
        b = noisy_latents.shape[0]
        residuals = []
        for c in self.out_channels:
            # produce a small per-scale tensor
            residuals.append(torch.ones((b, c, 8, 8)))
        return tuple(residuals)


def test_compute_control_residuals_multiscale_and_cpu_pinned():
    adapter = DummyAdapter()
    control = torch.zeros((1, 3, 128, 128))
    latents = torch.zeros((1, 4, 64, 64))
    timesteps = torch.tensor([10])

    # default storage (gpu assumed) -- we run on cpu so this should return tensors
    residuals = compute_control_residuals(adapter, control, latents, timesteps, device=torch.device("cpu"), residual_storage="gpu")
    assert isinstance(residuals, tuple)
    assert len(residuals) == 3
    for r in residuals:
        assert isinstance(r, torch.Tensor)
        assert r.requires_grad is False

    # test cpu_pinned storage
    residuals_pinned = compute_control_residuals(adapter, control, latents, timesteps, device=torch.device("cpu"), residual_storage="cpu_pinned")
    assert isinstance(residuals_pinned, tuple)
    for r in residuals_pinned:
        assert isinstance(r, torch.Tensor)
        # cpu_pinned tensors are on cpu and pinned
        assert r.device.type == 'cpu'
        assert r.is_pinned()
