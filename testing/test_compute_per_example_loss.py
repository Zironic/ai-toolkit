import torch
from toolkit.util import loss_utils


def test_compute_per_example_loss_image_shape():
    # B,C,H,W
    loss = torch.tensor([[[[1.0, 1.0],[1.0,1.0]]]], dtype=torch.float32)  # shape (1,1,2,2)
    per, comps = loss_utils.compute_per_example_loss(loss)
    assert per.shape == (1,)
    assert per[0].item() == 1.0
    assert 'main' in comps


def test_compute_per_example_loss_video_shape():
    # B,C,T,H,W
    loss = torch.zeros((2,3,4,2,2), dtype=torch.float32)
    loss[0] += 2.0
    loss[1] += 5.0
    per, comps = loss_utils.compute_per_example_loss(loss)
    assert per.shape == (2,)
    assert per[0].item() == 2.0
    assert per[1].item() == 5.0


def test_compute_per_example_with_prior():
    loss = torch.tensor([[[[1.0, 1.0],[1.0,1.0]]],[ [[2.0,2.0],[2.0,2.0]] ]], dtype=torch.float32) # (2,1,2,2)
    prior = torch.tensor([1.0, 0.5])
    per, comps = loss_utils.compute_per_example_loss(loss, prior_loss_tensor=prior)
    # main per-sample should be 1.0 and 2.0
    assert comps['main'].shape == (2,)
    assert comps['prior'].shape == (2,)
    assert per[0].item() == 1.0 + 1.0
    assert abs(per[1].item() - (2.0 + 0.5)) < 1e-6


def test_mean_parity_with_tensor_reduction():
    # ensure that mean(per_sample) equals loss.mean() after per-sample reduction
    loss = torch.randn((4,3,8,8), dtype=torch.float32).abs()
    per, comps = loss_utils.compute_per_example_loss(loss)
    # compute the same reduction manually
    manual = loss.mean(dim=[1,2,3])
    assert torch.allclose(per, manual.detach().cpu())
    assert abs(float(per.mean()) - float(manual.mean())) < 1e-6
