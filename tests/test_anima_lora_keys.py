import torch

from extensions_built_in.diffusion_models.anima import AnimaModel


def test_anima_lora_keys_save_and_load_round_trip():
    model = AnimaModel.__new__(AnimaModel)
    source = {
        "transformer.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.ones(1),
        "transformer.patch_embed.proj.lora_B.weight": torch.ones(1),
    }

    saved = model.convert_lora_weights_before_save(source)

    assert set(saved) == {
        "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight",
        "diffusion_model.x_embedder.proj.1.lora_up.weight",
    }
    loaded = model.convert_lora_weights_before_load(saved)
    assert set(loaded) == set(source)
