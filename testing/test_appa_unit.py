import torch
from toolkit.appearance_pose_adapt import AppearancePoseAdapter
from toolkit.controlnet_offload import bring_adapter, offload_adapter


def test_transform_preserves_shape_dtype():
    # Create sample residuals
    down = [torch.randn(2, 64, 64, 64), torch.randn(2, 128, 32, 32)]
    mid = torch.randn(2, 256, 16, 16)

    appa = AppearancePoseAdapter()

    out_down, out_mid = appa.transform_residuals(down, mid)

    # shapes and dtypes preserved
    assert isinstance(out_down, list)
    assert len(out_down) == len(down)
    for inp, out in zip(down, out_down):
        assert out.shape == inp.shape
        assert out.dtype == inp.dtype

    assert out_mid is not None
    assert out_mid.shape == mid.shape
    assert out_mid.dtype == mid.dtype


def test_grad_flow_and_controlnet_frozen():
    # Simulate a frozen controlnet adapter by creating params that do not require grad
    adapter = torch.nn.Linear(10, 10)
    for p in adapter.parameters():
        p.requires_grad = False

    appa = AppearancePoseAdapter()

    # initialize via call
    down = [torch.randn(1, 16, 8, 8), torch.randn(1, 32, 4, 4)]
    mid = torch.randn(1, 64, 2, 2)

    out_down, out_mid = appa.transform_residuals(down, mid)

    # create a simple scalar loss from outputs
    loss = sum([o.sum() for o in out_down if o is not None]) + (out_mid.sum() if out_mid is not None else 0.0)
    loss.backward()

    # APPA params should have gradients
    grads = [p.grad for p in appa.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "APPA parameters did not receive gradients"

    # Adapter params must remain without grad (frozen)
    assert all(p.grad is None for p in adapter.parameters())


def test_offload_helpers_accept_appa():
    appa = AppearancePoseAdapter()
    # bring/offload with 'none' strategy should be callable and not raise
    bring_adapter(appa, device=torch.device('cpu'), strategy='none')
    offload_adapter(appa, strategy='none')
