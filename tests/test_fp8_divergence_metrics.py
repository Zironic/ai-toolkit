"""CPU unit tests for the fp8-divergence analyzer math (ticket fce0b45).

Synthetic tensors with known injected error -- no GPU, no model, no dumps.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_fp8_backward_divergence import (  # noqa: E402
    block_index,
    checkpoint_rounding_metrics,
    classify_growth,
    dw_metrics,
    effective_dw,
    grad_cosines,
    optim_state_rel_error,
    pair_lora_modules,
    tensor_metrics,
)

RANK = 4
ALPHA = 8.0


def _lora_state(seed=0, modules=("lora_transformer_blocks_0_attn", "lora_transformer_tproj")):
    gen = torch.Generator().manual_seed(seed)
    state = {}
    for key in modules:
        state[f"{key}.lora_down.weight"] = torch.randn(RANK, 32, generator=gen)
        state[f"{key}.lora_up.weight"] = torch.randn(16, RANK, generator=gen)
    return state


def test_pair_lora_modules_pairs_and_skips_orphans():
    state = _lora_state()
    state["orphan.lora_down.weight"] = torch.randn(RANK, 8)
    pairs = pair_lora_modules(state)
    assert set(pairs) == {"lora_transformer_blocks_0_attn", "lora_transformer_tproj"}
    down, up = pairs["lora_transformer_tproj"]
    assert down.shape == (RANK, 32) and up.shape == (16, RANK)


def test_effective_dw_scale():
    down = torch.eye(RANK, 8)
    up = torch.eye(8, RANK)
    dw = effective_dw(down, up, ALPHA, RANK)
    assert torch.allclose(dw, (ALPHA / RANK) * (up @ down))


def test_tensor_metrics_identity_and_known_error():
    ref = torch.randn(64)
    same = tensor_metrics(ref, ref)
    assert same["rel_fro"] == 0.0
    assert same["cos"] == pytest.approx(1.0, abs=1e-6)
    assert same["norm_ratio"] == pytest.approx(1.0)

    eps = 1e-3
    perturbed = ref * (1.0 + eps)  # pure scaling: rel_fro == eps, cos == 1
    m = tensor_metrics(perturbed, ref)
    assert m["rel_fro"] == pytest.approx(eps, rel=1e-4)
    assert m["cos"] == pytest.approx(1.0, abs=1e-6)
    assert m["norm_ratio"] == pytest.approx(1.0 + eps, rel=1e-5)
    # (ref*(1+eps) - ref) vs ref*eps differ by fp32 rounding of the subtraction
    assert m["max_abs"] == pytest.approx((ref * eps).abs().max().item(), rel=1e-2)


def test_dw_metrics_injected_error_and_block_resolution():
    ref = _lora_state()
    test = {k: v.clone() for k, v in ref.items()}
    # Perturb ONLY the block-0 module's up weight.
    key = "lora_transformer_blocks_0_attn.lora_up.weight"
    test[key] = test[key] * 1.01
    out = dw_metrics(test, ref, ALPHA, RANK)
    per_block = out["per_block_rel_fro"]
    assert per_block[0] == pytest.approx(0.01, rel=1e-4)
    assert per_block["tproj/other"] == 0.0
    assert 0.0 < out["aggregate"]["rel_fro"] < 0.01
    assert out["per_module"]["lora_transformer_blocks_0_attn"]["block"] == 0
    assert out["per_module"]["lora_transformer_tproj"]["block"] is None


def test_block_index_parsing():
    assert block_index("lora_transformer_blocks_17_mlp") == 17
    assert block_index("blocks.3.attn.qkv") == 3
    assert block_index("transformer$$blocks$$0$$attn$$wq") == 0
    assert block_index("transformer$$blocks$$27$$mlp$$down") == 27
    assert block_index("transformer$$txtfusion$$refiner_blocks$$1$$mlp$$up") is None
    assert block_index("lora_transformer_tproj") is None


def test_checkpoint_rounding_floor_order_of_magnitude():
    # bf16 has 8 mantissa bits: elementwise relative rounding ~2^-9; the
    # aggregate dW rel_fro must land well inside (1e-4, 1e-2).
    ref = _lora_state(seed=3)
    floor = checkpoint_rounding_metrics(ref, ALPHA, RANK)
    assert 1e-4 < floor["rel_fro"] < 1e-2
    assert floor["cos"] > 0.9999


def test_grad_cosines():
    ref = {"a.lora_down.weight": torch.randn(4, 8), "b.lora_down.weight": torch.randn(4, 8)}
    test = {k: v.clone() for k, v in ref.items()}
    test["a.lora_down.weight"] = -test["a.lora_down.weight"]  # anti-parallel
    out = grad_cosines(test, ref)
    assert out["per_param"]["a.lora_down.weight"] == pytest.approx(-1.0, abs=1e-6)
    assert out["per_param"]["b.lora_down.weight"] == pytest.approx(1.0, abs=1e-6)
    assert -1.0 < out["aggregate"] < 1.0


def test_optim_state_rel_error():
    ref = {"state": {0: {"exp_avg": torch.ones(10), "exp_avg_sq": torch.ones(10)}}}
    test = {
        "state": {
            0: {"exp_avg": torch.full((10,), 1.02), "exp_avg_sq": torch.ones(10)}
        }
    }
    out = optim_state_rel_error(test, ref)
    assert out["exp_avg"] == pytest.approx(0.02, rel=1e-5)
    assert out["exp_avg_sq"] == 0.0


def test_classify_growth_modes():
    steps = [1, 5, 10, 25, 50, 100]
    flat = {s: 1e-4 for s in steps}
    diffusive = {s: 1e-4 * math.sqrt(s) for s in steps}
    linear = {s: 1e-4 * s for s in steps}
    unstable = {s: 1e-4 * s ** 2 for s in steps}
    assert classify_growth(flat).startswith("fixed-offset")
    assert classify_growth(diffusive).startswith("sqrt-diffusive")
    assert classify_growth(linear).startswith("linear")
    assert classify_growth(unstable).startswith("superlinear")
    assert classify_growth({1: 1e-4}) == "insufficient-data"
