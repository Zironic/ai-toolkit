import inspect

import pytest
import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_z_image import (
    ZSingleStreamAttnProcessor,
)

from toolkit.models.v2.z_image import (
    CacheableZSingleStreamAttnProcessor,
    ZImageTransformer2DModel,
)


def test_cacheable_zimage_processor_has_no_nested_autocast_context():
    source = inspect.getsource(CacheableZSingleStreamAttnProcessor.__call__)
    assert "torch.amp.autocast" not in source
    assert "_enter_autocast" not in source


def test_zimage_model_replaces_diffusers_processor(monkeypatch):
    def minimal_init(model, *args, **kwargs):
        torch.nn.Module.__init__(model)
        model.attention = Attention(
            query_dim=8,
            heads=2,
            dim_head=4,
            processor=ZSingleStreamAttnProcessor(),
        )

    monkeypatch.setattr(
        "toolkit.models.v2.z_image.DiffusersZImageTransformer2DModel.__init__",
        minimal_init,
    )
    model = ZImageTransformer2DModel()
    assert isinstance(model.attention.processor, CacheableZSingleStreamAttnProcessor)


def test_cacheable_zimage_processor_matches_diffusers_on_cpu():
    torch.manual_seed(11)
    attention = Attention(query_dim=8, heads=2, dim_head=4)
    hidden = torch.randn(2, 5, 8)
    angles = torch.randn(2, 5, 2)
    frequencies = torch.polar(torch.ones_like(angles), angles)

    original = ZSingleStreamAttnProcessor()
    cacheable = CacheableZSingleStreamAttnProcessor()
    expected = original(attention, hidden, freqs_cis=frequencies)
    actual = cacheable(attention, hidden, freqs_cis=frequencies)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cacheable_zimage_processor_compiles_fullgraph_under_outer_autocast():
    torch.manual_seed(17)
    device = torch.device("cuda")
    attention = Attention(query_dim=32, heads=4, dim_head=8).to(
        device=device, dtype=torch.bfloat16
    )
    processor = CacheableZSingleStreamAttnProcessor()
    hidden = torch.randn(
        2, 8, 32, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    angles = torch.randn(2, 8, 4, device=device)
    frequencies = torch.polar(torch.ones_like(angles), angles)

    def kernel(value, rotary):
        return processor(attention, value, freqs_cis=rotary)

    compiled = torch.compile(kernel, fullgraph=True, dynamic=False)
    graph_breaks_before = sum(
        torch._dynamo.utils.counters["graph_break"].values()
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = compiled(hidden, frequencies)
        output.float().square().mean().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert sum(torch._dynamo.utils.counters["graph_break"].values()) == (
        graph_breaks_before
    )
