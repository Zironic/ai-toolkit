import torch
from types import SimpleNamespace
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
from toolkit.config_modules import ModelConfig, GenerateImageConfig
from toolkit.prompt_utils import PromptEmbeds


def test_generate_single_image_with_controlnet_routes_control_context():
    cfg = ModelConfig(name_or_path='dummy_model')
    model = ZImageModel('cpu', cfg)

    # Patch a controlnet into the model and mark as enabled
    class DummyControl(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = 3
            self.control_layers = []
            self.control_in_dim = 1
            self.control_all_x_embedder = True
        def forward(self, x, t, cap):
            # record call args for test via closure
            return torch.zeros(1)
    model.controlnet = DummyControl()
    model.is_controlnet_enabled = True

    # Build gen config and control input
    gen_config = GenerateImageConfig(output_folder='.', output_ext='png')
    gen_config.width = 64
    gen_config.height = 64
    gen_config.num_inference_steps = 2
    gen_config.latents = None
    gen_config.control_conditioning_scale = 0.9

    conditional_embeds = PromptEmbeds(torch.zeros((1,8,16)))
    unconditional_embeds = PromptEmbeds(torch.zeros((1,8,16)))

    control_img = torch.rand((3, 64, 64))

    # Create a dummy pipeline that ensures incoming 'control_context' is passed to the actual controlnet path
    class DummyPipeline:
        def __init__(self, model):
            self.model = model
            self.received_extra = None
        def __call__(self, *args, **kwargs):
            self.received_extra = kwargs
            class R: pass
            R.images = [torch.zeros((64,64,3))]
            return R()

    pipeline = DummyPipeline(model)

    img = model.generate_single_image(pipeline, gen_config, conditional_embeds, unconditional_embeds, torch.Generator(), extra={'control_images': control_img})

    assert pipeline.received_extra is not None
    assert 'control_context' in pipeline.received_extra
    assert isinstance(pipeline.received_extra['control_context'], list)
    assert pipeline.received_extra['control_conditioning_scale'] == 0.9
    assert img is not None
