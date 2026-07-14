from pathlib import Path

from toolkit.config_modules import ModelConfig
from toolkit.util.get_model import LEGACY_MODEL_ARCHES, get_model_class


def test_anima_resolves_to_base_model_extension():
    model_class = get_model_class(
        ModelConfig(arch="anima", name_or_path="dummy/anima")
    )

    assert model_class.__name__ == "AnimaModel"
    assert model_class.arch == "anima"
    assert "anima" not in LEGACY_MODEL_ARCHES


def test_legacy_core_has_no_anima_dispatch():
    root = Path(__file__).parents[1]
    forbidden = {
        "toolkit/stable_diffusion_model.py": ("is_anima", "encode_prompts_anima"),
        "toolkit/train_tools.py": ("encode_prompts_anima",),
        "toolkit/lora_special.py": ("is_anima", "CosmosTransformer3DModel"),
        "jobs/process/BaseSDTrainProcess.py": ("is_anima",),
        "toolkit/config_modules.py": ("is_anima",),
    }
    for relative_path, needles in forbidden.items():
        source = (root / relative_path).read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in source

    assert not (root / "toolkit/models/anima.py").exists()
