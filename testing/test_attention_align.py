import torch
from toolkit.attention_align import avg_attention_maps, attention_alignment_loss


def make_fake_attentions(num_layers=3, B=1, H=2, T=10, S=10):
    # return list of tensors [B, H, T, S]
    return [torch.rand(B, H, T, S) for _ in range(num_layers)]


def test_avg_attention_maps_shape():
    atts = make_fake_attentions()
    out = avg_attention_maps(atts, token_indices=[3, 5])
    assert out.shape[0] == 1
    assert out.shape[1] == 2
    assert out.shape[2] == atts[0].shape[2]


def test_attention_alignment_loss_mse():
    B, K, N = 2, 1, 16
    att = torch.rand(B, K, N)
    mask = torch.rand(B, K, N)
    loss = attention_alignment_loss(att, mask, mode='mse')
    assert loss.item() >= 0


def test_attention_alignment_loss_iou():
    B, K, N = 1, 1, 9
    att = torch.eye(N).unsqueeze(0).unsqueeze(0)
    mask = torch.eye(N).unsqueeze(0).unsqueeze(0)
    loss = attention_alignment_loss(att, mask, mode='iou')
    assert loss.item() <= 1.0
