import torch
import pytest


class DummyUnionPipeline:
    def __init__(self):
        self.num_inference_steps = 1

    def predict(self, prompt, image, controls, control_context_scale=1.0):
        # Basic validation according to contract
        assert isinstance(image, torch.FloatTensor)
        assert image.ndim == 4  # [B,C,H,W]
        B = image.shape[0]
        assert isinstance(controls, list)
        # every control must be [B, Cc, H, W]
        for c in controls:
            assert isinstance(c, torch.FloatTensor)
            assert c.shape[0] == B
            assert c.ndim == 4
            # values in [0,1]
            assert c.min() >= 0.0 and c.max() <= 1.0
        # control_context_scale either float or list of floats
        if isinstance(control_context_scale, list):
            assert len(control_context_scale) == len(controls)
        # produce a deterministic dummy "image" shaped output
        out = torch.zeros(B, 3, 512, 512, dtype=torch.float32)
        # change output slightly depending on steps to simulate difference
        out += float(self.num_inference_steps) * 1e-3
        return out


def test_shapes_and_basic_forward():
    pipe = DummyUnionPipeline()
    B, C, H, W = 2, 3, 64, 64
    image = torch.rand(B, C, H, W, dtype=torch.float32)
    c1 = torch.rand(B, 1, H, W, dtype=torch.float32)
    c2 = torch.rand(B, 2, H, W, dtype=torch.float32)
    out = pipe.predict('hi', image, controls=[c1, c2], control_context_scale=[0.8, 0.7])
    assert out.shape == (B, 3, 512, 512)


def test_8_step_inference_behavior():
    pipe = DummyUnionPipeline()
    B, C, H, W = 1, 3, 64, 64
    image = torch.rand(B, C, H, W, dtype=torch.float32)
    c = torch.rand(B, 1, H, W, dtype=torch.float32)
    # run with default steps
    pipe.num_inference_steps = 1
    out1 = pipe.predict('prompt', image, controls=[c], control_context_scale=0.75)
    # run with 8 steps (recommended distilled case)
    pipe.num_inference_steps = 8
    out8 = pipe.predict('prompt', image, controls=[c], control_context_scale=0.75)
    assert out1.shape == out8.shape
    # outputs should differ because steps differ (dummy behavior)
    assert not torch.allclose(out1, out8)
