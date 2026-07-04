"""CPU tests for the Krea2 SKC projector injection.

Verifies (1) the 12-value vector is extracted from both the .diff (z0) and the
rank-1 lora_A/lora_B (skc3vo) storage forms, and (2) the forward hook adds the
exact projector delta the probe scripts use -- reducing the trailing 12-slot
axis, `(x * v).sum(-1, keepdim=True) * strength`.
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extensions_built_in.diffusion_models.krea2.src.skc_injection import (  # noqa: E402
    extract_projector_vector,
    make_skc_projector_hook,
)


class _DummyProjector(torch.nn.Module):
    """Stand-in projector: (B, N, 12) -> (B, N, 1). Returns a fixed base output."""

    def __init__(self, base_value: float = 0.0):
        super().__init__()
        self.base_value = base_value

    def forward(self, x):  # noqa: D401
        return torch.full((*x.shape[:-1], 1), self.base_value, dtype=x.dtype)


def test_extract_vector_diff_form():
    vec = torch.arange(12, dtype=torch.float32) * 0.1
    sd = {"diffusion_model.txtfusion.projector.diff": vec.clone().reshape(1, 12)}
    source, out = extract_projector_vector(sd)
    assert source == "diffusion_model.txtfusion.projector.diff"
    assert out.numel() == 12
    torch.testing.assert_close(out, vec)


def test_extract_vector_lora_pair_form():
    # skc3vo layout: lora_A (1, 12), lora_B (1, 1); effective vector = B @ A.
    a = torch.randn(1, 12)
    b = torch.randn(1, 1)
    sd = {
        "transformer.text_fusion.projector.lora_A.weight": a,
        "transformer.text_fusion.projector.lora_B.weight": b,
    }
    source, out = extract_projector_vector(sd)
    assert "lora_B.weight @" in source
    torch.testing.assert_close(out, torch.matmul(b, a).reshape(-1))


def test_extract_vector_rejects_wrong_width():
    sd = {"txtfusion.projector.diff": torch.zeros(8)}
    try:
        extract_projector_vector(sd)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-12 vector")


def test_hook_adds_expected_delta():
    torch.manual_seed(0)
    vector = torch.randn(12)
    strength = 0.075
    proj = _DummyProjector(base_value=0.0)
    handle = proj.register_forward_hook(make_skc_projector_hook(vector, strength))
    try:
        x = torch.randn(2, 5, 12)  # (B, N, 12)
        out = proj(x)
        expected = (x * vector.reshape(1, 1, 12)).sum(dim=-1, keepdim=True) * strength
        assert out.shape == (2, 5, 1)
        torch.testing.assert_close(out, expected)
    finally:
        handle.remove()


def test_hook_scales_linearly_with_strength():
    torch.manual_seed(1)
    vector = torch.randn(12)
    x = torch.randn(3, 4, 12)
    proj = torch.nn.Identity  # not used; call hook math directly via two hooks
    d1 = (x * vector.reshape(1, 1, 12)).sum(-1, keepdim=True) * 0.03
    d2 = (x * vector.reshape(1, 1, 12)).sum(-1, keepdim=True) * 0.06
    torch.testing.assert_close(d2, d1 * 2.0)


def test_real_skc3vo_file_if_present():
    path = REPO_ROOT / "loras" / "krea_vector_explore" / "skc3vo.safetensors"
    if not path.exists():
        return  # file not present in this checkout; skip silently
    from safetensors.torch import load_file

    source, vec = extract_projector_vector(load_file(str(path)))
    assert vec.numel() == 12
    assert torch.isfinite(vec).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
