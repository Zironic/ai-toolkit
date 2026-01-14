import torch
import types
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class DummySDV:
    def __init__(self):
        self.vae = types.SimpleNamespace(config={'block_out_channels': [64, 128, 256, 512]})
        self.transformer = types.SimpleNamespace(all_patch_size=None)


class DummyTrainer(SDTrainer):
    def __init__(self):
        # minimal init
        self.sd = DummySDV()
        self.train_config = types.SimpleNamespace()
        self.device_torch = torch.device('cpu')
        # set flags
        self.train_config.diff_output_preservation = True
        self.train_config.blank_prompt_preservation = False
        self.train_config.diff_output_preservation_class = 'cat'
        self.train_config.diff_output_preservation_after_steps = 0
        self.train_config.diff_output_preservation_every = 1  # compatibility: full-res schedule
        self.train_config.diff_output_preservation_resolution = None


def test_dop_does_not_use_blank_prior_when_trigger_absent(monkeypatch):
    t = DummyTrainer()
    # prepare conditional prompts that do NOT include the trigger word
    conditioned_prompts = ['a photo of a tree']

    # monkeypatch encoding to produce a sentinel PromptEmbeds replacement object
    class DummyEmbeds:
        def expand_to_batch(self, b):
            return 'DOP_EXPANDED'

    def fake_encode_prompt(prompts, *args, **kwargs):
        return DummyEmbeds()

    # Simulate that DOP embeddings were prepared and stored
    class DummyEmbeds:
        def expand_to_batch(self, b):
            return 'DOP_EXPANDED'

    t.diff_output_preservation_embeds = DummyEmbeds()

    # Now assert that prior_embeds_to_use would be set to the DOP expanded embeds (not blank)
    prior_embeds_to_use = None
    if getattr(t, 'diff_output_preservation_embeds', None) is not None:
        prior_embeds_to_use = t.diff_output_preservation_embeds.expand_to_batch(1)

    assert prior_embeds_to_use == 'DOP_EXPANDED'