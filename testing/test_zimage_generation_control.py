import torch
from types import SimpleNamespace
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
from toolkit.config_modules import ModelConfig, GenerateImageConfig
from toolkit.prompt_utils import PromptEmbeds


def test_generate_single_image_passes_control_context_to_pipeline():
    cfg = ModelConfig(name_or_path='dummy_model')
    model = ZImageModel('cpu', cfg)
    # mark controlnet enabled and stub encoder
    model.is_controlnet_enabled = True

    gen_config = GenerateImageConfig(output_folder='.', output_ext='png')
    gen_config.width = 64
    gen_config.height = 64
    gen_config.num_inference_steps = 2
    gen_config.latents = None
    gen_config.control_conditioning_scale = 0.7

    conditional_embeds = PromptEmbeds(torch.zeros((1,8,16)))
    unconditional_embeds = PromptEmbeds(torch.zeros((1,8,16)))

    control_img = torch.rand((3, 64, 64))

    # Dummy pipeline captures kwargs passed
    class DummyPipeline:
        def __init__(self):
            self.last_call = None
        def __call__(self, *args, **kwargs):
            self.last_call = kwargs
            class R: pass
            R.images = [torch.zeros((64,64,3))]
            return R()

    pipeline = DummyPipeline()

    img = model.generate_single_image(pipeline, gen_config, conditional_embeds, unconditional_embeds, torch.Generator(), extra={'control_images': control_img})

    assert pipeline.last_call is not None
    assert 'control_context' in pipeline.last_call
    assert isinstance(pipeline.last_call['control_context'], list)
    assert pipeline.last_call['control_conditioning_scale'] == 0.7
    assert img is not None
