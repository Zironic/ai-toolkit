from types import SimpleNamespace
from importlib import import_module

import torch

import toolkit.util.get_model as model_registry
from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds


def test_embedding_space_resolver_uses_class_level_model_hook(monkeypatch):
    class FutureModel:
        @classmethod
        def get_text_embedding_space_version(cls, model_config):
            return f"{model_config.arch}-{model_config.model_kwargs['format']}"

    config = SimpleNamespace(arch="future", model_kwargs={"format": "v7"})
    monkeypatch.setattr(model_registry, "get_model_class", lambda _: FutureModel)

    assert model_registry.resolve_text_embedding_space_version(config) == "future-v7"


def test_embedding_space_resolver_keeps_string_attribute_compatibility(monkeypatch):
    class ThirdPartyModel:
        text_embedding_space_version = "third_party_v2"

    config = SimpleNamespace(arch="third_party")
    monkeypatch.setattr(model_registry, "get_model_class", lambda _: ThirdPartyModel)

    assert (
        model_registry.resolve_text_embedding_space_version(config)
        == "third_party_v2"
    )


def test_config_dependent_model_version_is_available_before_model_load():
    unlimited = SimpleNamespace(model_kwargs={})
    strict = SimpleNamespace(
        model_kwargs={
            "prompt_overflow_policy": "error",
            "max_text_length": 768,
        }
    )

    assert (
        Krea2Model.get_text_embedding_space_version(unlimited)
        == "krea2-v2-unlimited"
    )
    assert (
        Krea2Model.get_text_embedding_space_version(strict)
        == "krea2-v2-error-768"
    )


def test_worker_orchestration_accepts_any_model_declaring_capability(
    monkeypatch,
):
    class FutureModel:
        supports_te_cache_worker = True

    process = SimpleNamespace(
        is_caching_text_embeddings=True,
        model_config=SimpleNamespace(arch="future"),
        cache_text_encoder_outputs_to_disk=lambda: None,
        aux_cache_is_ready=lambda: True,
        _te_caption_manifest_is_current=lambda: True,
    )
    monkeypatch.delenv("AITK_IS_TE_WORKER", raising=False)
    process_module = import_module("jobs.process.BaseSDTrainProcess")
    monkeypatch.setattr(process_module, "get_model_class", lambda _: FutureModel)

    assert BaseSDTrainProcess.maybe_run_te_cache_worker(process) is True


def test_dataset_and_aux_caches_use_the_same_model_version_resolver(monkeypatch):
    trainer = object.__new__(SDTrainer)
    trainer.model_config = SimpleNamespace(
        arch="future",
        name_or_path="owner/future",
        encode_control_in_text_embeddings=False,
    )
    trainer.sample_config = None
    trainer.train_config = SimpleNamespace(
        disable_sampling=True,
        unconditional_prompt=None,
        diff_output_preservation=False,
        diff_output_preservation_class=None,
    )
    trainer.trigger_word = None

    monkeypatch.setattr(
        model_registry,
        "resolve_text_embedding_space_version",
        lambda _: "future_te_v4",
    )

    params = trainer._aux_config_params()
    file_item = SimpleNamespace(
        refresh_caption_for_text_embedding_cache=lambda: None,
    )
    trainer._prepare_file_item_text_cache_signature(None, file_item)

    assert params["text_embedding_space_version"] == "future_te_v4"
    assert file_item.text_embedding_space_version == "future_te_v4"


def test_zimage_owns_its_variable_length_cache_representation():
    class FakeTextEncoder:
        device = torch.device("cpu")

        def to(self, device):
            self.device = device
            return self

    class FakePipeline:
        text_encoder = FakeTextEncoder()

        def encode_prompt(self, prompt, do_classifier_free_guidance, device):
            return [torch.ones(3, 4), torch.ones(5, 4)], None

    model = object.__new__(ZImageModel)
    model.device_torch = torch.device("cpu")
    model.pipeline = FakePipeline()

    embeds = model.get_prompt_embeds(["short", "long"])

    assert isinstance(embeds, AdvancedPromptEmbeds)
    assert [tuple(item.shape) for item in embeds.text_embeds] == [(3, 4), (5, 4)]
    assert (
        ZImageModel.get_text_embedding_space_version(
            SimpleNamespace(arch="zimage")
        )
        == "zimage_te_v2"
    )


def test_advanced_prompt_embeds_round_trip_and_collate_as_opaque_data(tmp_path):
    cache_path = tmp_path / "prompt.safetensors"
    original = AdvancedPromptEmbeds(
        text_embeds=[torch.ones(3, 4)],
        token_weights=[torch.arange(3)],
    )
    original.frozen_dtype_keys = ["token_weights"]
    original.save(str(cache_path))

    loaded = PromptEmbeds.load(str(cache_path))
    batch = concat_prompt_embeds([loaded, loaded])
    moved = batch.to(dtype=torch.float16)

    assert isinstance(batch, AdvancedPromptEmbeds)
    assert [tuple(item.shape) for item in batch.text_embeds] == [(3, 4), (3, 4)]
    assert [tuple(item.shape) for item in batch.token_weights] == [(3,), (3,)]
    assert batch.frozen_dtype_keys == ["token_weights"]
    assert all(item.dtype == torch.float16 for item in moved.text_embeds)
    assert all(item.dtype == torch.int64 for item in moved.token_weights)
