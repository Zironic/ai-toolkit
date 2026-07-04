import pytest
import torch

from toolkit.network_mixins import apply_rank_gates


def test_apply_rank_gates_linear_last_dim():
    tensor = torch.ones(2, 3, 4)
    gates = torch.tensor([1.0, 0.5, 0.0, -1.0])

    gated = apply_rank_gates(tensor, gates, "linear")

    assert gated.shape == tensor.shape
    assert torch.allclose(gated[0, 0], gates)
    assert torch.allclose(gated[1, 2], gates)


def test_apply_rank_gates_conv_channel_dim():
    tensor = torch.ones(2, 4, 3, 3)
    gates = torch.tensor([1.0, 0.5, 0.0, -1.0])

    gated = apply_rank_gates(tensor, gates, "conv")

    assert gated.shape == tensor.shape
    assert torch.allclose(gated[0, :, 0, 0], gates)
    assert torch.allclose(gated[1, :, 2, 2], gates)


def test_apply_rank_gates_rejects_unmatched_rank():
    with pytest.raises(ValueError, match="Cannot apply rank gates"):
        apply_rank_gates(torch.ones(2, 3, 5), torch.ones(4), "bad")

def test_extract_projector_vectors_from_diff():
    from extensions_built_in.lora_vector_explorer.LoraVectorExploreProcess import extract_projector_vectors

    vectors = extract_projector_vectors({
        "diffusion_model.txtfusion.projector.diff": torch.arange(12, dtype=torch.float32).reshape(1, 12),
    })

    assert len(vectors) == 1
    assert vectors[0]["module"] == "txtfusion.projector"
    assert vectors[0]["vector"] == [float(x) for x in range(12)]


def test_extract_projector_vectors_from_rank_one_lora():
    from extensions_built_in.lora_vector_explorer.LoraVectorExploreProcess import extract_projector_vectors

    vectors = extract_projector_vectors({
        "transformer.text_fusion.projector.lora_A.weight": torch.arange(12, dtype=torch.float32).reshape(1, 12),
        "transformer.text_fusion.projector.lora_B.weight": torch.tensor([[2.0]]),
    })

    assert len(vectors) == 1
    assert vectors[0]["module"] == "txtfusion.projector"
    assert vectors[0]["vector"] == [float(x * 2) for x in range(12)]


def test_projector_diff_controller_adds_weight_delta():
    from extensions_built_in.lora_vector_explorer.LoraVectorExploreProcess import ProjectorDiffController

    projector = torch.nn.Linear(12, 1, bias=False)
    projector.weight.data.zero_()
    controller = ProjectorDiffController(projector)
    controller.set_vector([1.0] * 12, 0.5)

    try:
        out = projector(torch.ones(2, 12))
    finally:
        controller.restore()

    assert torch.allclose(out, torch.full((2, 1), 6.0))

def test_vector_stats_reports_largest_slots():
    from extensions_built_in.lora_vector_explorer.LoraVectorExploreProcess import _vector_stats

    stats = _vector_stats([0.0, -2.0, 0.5, 3.0])

    assert stats["numel"] == 4
    assert stats["top_slots"][0]["name"] == "V4"
    assert stats["top_slots"][1]["name"] == "V2"


def test_cosine_handles_zero_vectors():
    from extensions_built_in.lora_vector_explorer.LoraVectorExploreProcess import _cosine

    assert _cosine([0.0, 0.0], [1.0, 0.0]) is None
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0

