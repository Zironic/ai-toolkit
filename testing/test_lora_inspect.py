import torch
from toolkit.lora_inspect import inspect_lora_state_dict


def test_detects_control_and_residual_keys():
    sd = {
        "transformer.encoder.lora_A.weight": torch.zeros(1),
        "transformer.encoder.lora_B.weight": torch.zeros(1),
        "controlnet.down_block_additional_residuals.0.weight": torch.randn(2, 2),
        "control_projection.weight": torch.randn(3, 3),
        "x_embedder.lora_A.weight": torch.randn(1),
    }

    rep = inspect_lora_state_dict(sd)
    assert rep["lora_keys_count"] >= 1
    assert rep["control_like_count"] >= 2
    assert rep["residual_like_count"] >= 1
    assert rep["input_embedder_like_count"] >= 1


def test_no_false_positive_on_generic_lora():
    sd = {
        "transformer.blocks.0.attention.to_k.lora_A.weight": torch.zeros(1),
        "transformer.blocks.0.attention.to_k.lora_B.weight": torch.zeros(1),
    }
    rep = inspect_lora_state_dict(sd)
    assert rep["lora_keys_count"] == 2
    assert rep["control_like_count"] == 0
    assert rep["residual_like_count"] == 0
    assert rep["input_embedder_like_count"] == 0
