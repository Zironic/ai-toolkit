from contextlib import nullcontext
from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models.anima import AnimaModel
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
        return (hidden_states,)


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
        batch, length = target_input_ids.shape
        return torch.ones(batch, length, 6)


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
    assert [tuple(item.shape) for item in embeds.text_embeds] == [(3, 6), (1, 6)]
    assert all(item.ndim == 2 for item in embeds.text_embeds)


def test_anima_keeps_five_dimensional_patch_projection_dense():
    model = AnimaModel.__new__(AnimaModel)

    assert model.get_quantization_exclude_modules() == ["patch_embed.proj"]


def test_anima_scheduler_uses_checkpoint_static_shift():
    scheduler = AnimaModel.get_train_scheduler()

    scheduler.set_timesteps(sigmas=[1.0, 0.5], device="cpu")

    assert scheduler.config.use_dynamic_shifting is False
    torch.testing.assert_close(scheduler.timesteps, torch.tensor([1000.0, 750.0]))
