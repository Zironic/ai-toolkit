from contextlib import nullcontext
from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models.anima import AnimaModel
from extensions_built_in.diffusion_models.anima.anima import _load_anima_tokenizers
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds


class _LatentDistribution:
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value


class _FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = SimpleNamespace(
            latents_mean=[1.0, 2.0],
            latents_std=[2.0, 4.0],
        )
        self.encoded_shape = None
        self.decoded_shapes = []

    @property
    def device(self):
        return self.anchor.device

    def encode(self, images):
        self.encoded_shape = tuple(images.shape)
        raw = torch.empty(images.shape[0], 2, 1, images.shape[-2] // 8, images.shape[-1] // 8)
        raw[:, 0].fill_(5.0)
        raw[:, 1].fill_(10.0)
        return SimpleNamespace(latent_dist=_LatentDistribution(raw))

    def decode(self, latents, return_dict=False):
        self.decoded_shapes.append(tuple(latents.shape))
        decoded = torch.zeros(latents.shape[0], 3, 1, latents.shape[-2] * 8, latents.shape[-1] * 8)
        return (decoded,)


class _FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = SimpleNamespace(in_channels=2, concat_padding_mask=True)
        self.patch_embed = SimpleNamespace(
            proj=SimpleNamespace(in_channels=self.config.in_channels + 1)
        )
        self.hidden_states = None
        self.padding_mask = None
        self.encoder_hidden_states = None
        self.encoder_sequence_lengths = []

    @property
    def device(self):
        return self.anchor.device

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        padding_mask,
        return_dict=False,
    ):
        self.hidden_states = hidden_states
        self.padding_mask = padding_mask
        self.encoder_hidden_states = encoder_hidden_states
        self.encoder_sequence_lengths.append(encoder_hidden_states.shape[1])
        return (hidden_states,)


class _FakeScheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.timesteps = torch.tensor([1000.0])

    def set_timesteps(self, sigmas, device):
        self.timesteps = self.timesteps.to(device)

    def set_begin_index(self, index):
        self.begin_index = index

    def step(self, prediction, timestep, latents, return_dict=False):
        return (latents,)


class _FakeTokenizer:
    def __call__(self, prompts, **kwargs):
        batch = len(prompts)
        ids = torch.arange(3).expand(batch, 3)
        mask = torch.tensor([[1, 1, 1], [1, 0, 0]])[:batch]
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


class _FakeTextEncoder(torch.nn.Module):
    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        hidden = torch.ones(input_ids.shape[0], input_ids.shape[1], 4)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeConditioner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(
        self,
        source_hidden_states,
        target_input_ids,
        target_attention_mask,
        source_attention_mask,
    ):
        batch = target_input_ids.shape[0]
        return torch.ones(batch, 5, 6)


def _bare_model():
    model = AnimaModel.__new__(AnimaModel)
    model.device_torch = torch.device("cpu")
    model.vae_device_torch = torch.device("cpu")
    model.te_device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.vae_torch_dtype = torch.float32
    model.vae_scale_factor = 8
    model.accelerator = SimpleNamespace(autocast=nullcontext)
    return model


def test_anima_tokenizers_use_complete_cached_snapshot(monkeypatch, tmp_path):
    repo_id = "owner/anima"
    snapshot = str(tmp_path)
    qwen_calls = []
    t5_calls = []
    qwen_tokenizer = object()
    t5_tokenizer = object()

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo, local_files_only: snapshot,
    )
    monkeypatch.setattr(
        "extensions_built_in.diffusion_models.anima.anima.Qwen2Tokenizer.from_pretrained",
        lambda source, subfolder: qwen_calls.append((source, subfolder))
        or qwen_tokenizer,
    )
    monkeypatch.setattr(
        "extensions_built_in.diffusion_models.anima.anima.T5TokenizerFast.from_pretrained",
        lambda source, subfolder: t5_calls.append((source, subfolder))
        or t5_tokenizer,
    )

    result = _load_anima_tokenizers(repo_id)

    assert result == (qwen_tokenizer, t5_tokenizer)
    assert qwen_calls == [(snapshot, "tokenizer")]
    assert t5_calls == [(snapshot, "t5_tokenizer")]


def test_anima_tokenizers_fall_back_when_cached_snapshot_is_partial(
    monkeypatch, tmp_path
):
    repo_id = "owner/anima"
    snapshot = str(tmp_path)
    qwen_calls = []
    t5_calls = []
    qwen_tokenizer = object()
    t5_tokenizer = object()

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo, local_files_only: snapshot,
    )

    def load_qwen(source, subfolder):
        qwen_calls.append((source, subfolder))
        return qwen_tokenizer

    def load_t5(source, subfolder):
        t5_calls.append((source, subfolder))
        if source == snapshot:
            raise OSError("t5 tokenizer is not cached")
        return t5_tokenizer

    monkeypatch.setattr(
        "extensions_built_in.diffusion_models.anima.anima.Qwen2Tokenizer.from_pretrained",
        load_qwen,
    )
    monkeypatch.setattr(
        "extensions_built_in.diffusion_models.anima.anima.T5TokenizerFast.from_pretrained",
        load_t5,
    )

    result = _load_anima_tokenizers(repo_id)

    assert result == (qwen_tokenizer, t5_tokenizer)
    assert qwen_calls == [
        (snapshot, "tokenizer"),
        (repo_id, "tokenizer"),
    ]
    assert t5_calls == [
        (snapshot, "t5_tokenizer"),
        (repo_id, "t5_tokenizer"),
    ]


def test_anima_encode_and_decode_keep_trainer_latents_4d():
    model = _bare_model()
    model.vae = _FakeVAE()

    latents = model.encode_images([torch.zeros(3, 16, 24)])

    assert model.vae.encoded_shape == (1, 3, 1, 16, 24)
    assert latents.shape == (1, 2, 2, 3)
    torch.testing.assert_close(latents[:, 0], torch.full((1, 2, 3), 2.0))
    torch.testing.assert_close(latents[:, 1], torch.full((1, 2, 3), 2.0))

    assert model.decode_latents(latents).shape == (1, 3, 16, 24)
    assert model.decode_latents(latents.unsqueeze(2)).shape == (1, 3, 16, 24)
    assert model.vae.decoded_shapes == [(1, 2, 1, 2, 3), (1, 2, 1, 2, 3)]


def test_anima_forward_expands_cosmos_shape_and_passes_mask_separately():
    model = _bare_model()
    model.model = _FakeTransformer()
    model.noise_scheduler = SimpleNamespace(
        config=SimpleNamespace(num_train_timesteps=1000)
    )
    embeds = AdvancedPromptEmbeds(
        text_embeds=[torch.ones(2, 6), torch.ones(3, 6)]
    )

    prediction = model.get_noise_prediction(
        torch.zeros(2, 2, 4, 5), torch.tensor([500.0, 250.0]), embeds
    )

    assert prediction.shape == (2, 2, 4, 5)
    assert model.model.hidden_states.shape == (2, 2, 1, 4, 5)
    assert model.model.padding_mask.shape == (1, 1, 32, 40)
    assert model.model.encoder_hidden_states.shape == (2, 3, 6)
    assert model.model.hidden_states.shape[1] == model.model.config.in_channels
    assert model.model.patch_embed.proj.in_channels == model.model.config.in_channels + 1


def test_anima_prompt_cache_stores_one_2d_tensor_per_prompt():
    model = _bare_model()
    model.max_sequence_length = 512
    model.tokenizer = _FakeTokenizer()
    model.t5_tokenizer = _FakeTokenizer()
    model.text_encoder = _FakeTextEncoder()
    model.text_conditioner = _FakeConditioner()

    embeds = model.get_prompt_embeds(["long", "short"])

    assert isinstance(embeds, AdvancedPromptEmbeds)
    assert [tuple(item.shape) for item in embeds.text_embeds] == [(5, 6), (5, 6)]
    assert all(item.ndim == 2 for item in embeds.text_embeds)
    assert (
        AnimaModel.get_text_embedding_space_version(
            SimpleNamespace(arch="anima")
        )
        == "anima_te_v3"
    )


def test_anima_keeps_five_dimensional_patch_projection_dense():
    model = AnimaModel.__new__(AnimaModel)

    assert model.get_quantization_exclude_modules() == ["patch_embed.proj"]


def test_anima_scheduler_uses_checkpoint_static_shift():
    scheduler = AnimaModel.get_train_scheduler()

    scheduler.set_timesteps(sigmas=[1.0, 0.5], device="cpu")

    assert scheduler.config.use_dynamic_shifting is False
    torch.testing.assert_close(scheduler.timesteps, torch.tensor([1000.0, 750.0]))


def test_anima_sampling_preserves_distinct_cfg_sequence_lengths():
    model = _bare_model()
    model.model = _FakeTransformer()
    model.get_train_scheduler = lambda: _FakeScheduler()
    model.decode_latents = lambda latents: torch.zeros(1, 3, 16, 16)
    conditional = AdvancedPromptEmbeds(text_embeds=[torch.ones(7, 6)])
    unconditional = AdvancedPromptEmbeds(text_embeds=[torch.ones(3, 6)])
    gen_config = SimpleNamespace(
        height=16,
        width=16,
        latents=None,
        num_inference_steps=1,
        guidance_scale=4.0,
    )

    model.generate_single_image(
        pipeline=None,
        gen_config=gen_config,
        conditional_embeds=conditional,
        unconditional_embeds=unconditional,
        generator=torch.Generator().manual_seed(42),
        extra={},
    )

    assert model.model.encoder_sequence_lengths == [7, 3]
