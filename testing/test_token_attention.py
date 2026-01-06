import torch
from toolkit.token_attention import (
    token_zscore_saliency,
    token_sigmoid_weights,
    avg_token_attention_maps,
    token_attention_mask_from_weights,
    token_attention_mask_to_spatial,
)


def test_token_zscore_saliency_basic():
    emb = torch.tensor([[[1.0, 0.0], [5.0, 0.0], [1.0, 0.0]]])  # B=1, T=3, D=2
    sal = token_zscore_saliency(emb)
    assert sal.shape == (1, 3)
    # middle token has different value -> larger saliency
    assert sal[0, 1] > sal[0, 0]


def test_token_sigmoid_weights_basic():
    sal = torch.tensor([[0.1, 0.5, 0.9]])
    w = token_sigmoid_weights(sal, scale=10.0)
    assert w.shape == sal.shape
    assert w[0, 2] > w[0, 0]


def test_avg_token_attention_maps_and_mask():
    # create fake attentions: one layer [B,H,T,S]
    B, H, T, S = 1, 2, 3, 4
    a = torch.zeros((B, H, T, S))
    # make token 1 attend to source positions 2 and 3
    a[0, :, 1, 2] = 0.6
    a[0, :, 1, 3] = 0.4
    atts = [a]
    maps = avg_token_attention_maps(atts)  # [B,T,S]
    assert maps.shape == (B, T, S)
    # token attention mask: pick token 1
    token_weights = torch.tensor([[0.0, 1.0, 0.0]])
    mask_flat = token_attention_mask_from_weights(maps, token_weights, normalize=False)
    assert mask_flat.shape == (B, S)
    # values should be present at index 2 and 3
    assert mask_flat[0, 2] > 0
    assert mask_flat[0, 3] > 0

    # reshape to HxW if S matches
    H_lat, W_lat = 2, 2
    mask_spatial = token_attention_mask_to_spatial(mask_flat, H_lat, W_lat)
    assert mask_spatial.shape == (B, 1, H_lat, W_lat)