import torch
from extensions_built_in.diffusion_models import z_image_transformer2d as shim


def test_shim_imports_and_instantiates():
    model = shim.ZImageTransformer2DModel()
    assert hasattr(model, 'in_channels')
    x = torch.randn(1, model.in_channels, 4, 4)
    out = model(x)
    # passthrough shim returns same tensor (or shape-compatible); check no error
    assert out is not None


def test_shim_final_layer():
    f = shim.FinalLayer(8, 4)
    x = torch.randn(2, 8)
    y = f(x)
    assert y.shape[1] == 4
