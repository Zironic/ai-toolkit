import torch
import tempfile
import os
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel


def test_lora_attach_and_optimizer_params():
    # Create a dummy UNet-like module with a Linear child so LoRA can attach
    class UNet2DConditionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.some_block = torch.nn.Module()
            self.some_block.lin = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.some_block.lin(x)

    unet = UNet2DConditionModel()
    # Ensure base params are frozen as in training flow
    unet.requires_grad_(False)

    # Create a minimal LoRA network to use for constructing LoRA modules
    net = LoRASpecialNetwork(text_encoder=torch.nn.Module(), unet=torch.nn.Module(), lora_dim=2, alpha=2)

    # Create a LoRAModule directly bound to a base linear module
    from toolkit.lora_special import LoRAModule
    base_lin = torch.nn.Linear(4, 4)
    base_lin.requires_grad_(False)
    lmod = LoRAModule('test_lora', org_module=base_lin, multiplier=1.0, lora_dim=2, alpha=2, network=net)
    lmod.apply_to()

    # LoRA parameters should be present and trainable; base remains frozen
    lparams = list(lmod.parameters())
    assert len(lparams) > 0
    for p in lparams:
        assert p.requires_grad
    assert not base_lin.weight.requires_grad


def test_single_step_updates_only_lora_params():
    # Same dummy UNet module
    class UNet2DConditionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.some_block = torch.nn.Module()
            self.some_block.lin = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.some_block.lin(x)

    unet = UNet2DConditionModel()
    unet.requires_grad_(False)

    # Use LoRA module directly to verify update behavior
    net = LoRASpecialNetwork(text_encoder=torch.nn.Module(), unet=torch.nn.Module(), lora_dim=2, alpha=2)
    from toolkit.lora_special import LoRAModule

    base_lin = torch.nn.Linear(4, 4)
    base_lin.requires_grad_(False)

    lmod = LoRAModule('test_lora', org_module=base_lin, multiplier=1.0, lora_dim=2, alpha=2, network=net)
    lmod.apply_to()

    # lora_up is initialized to zeros by default; set to non-zero so grads flow
    if hasattr(lmod, 'lora_up') and hasattr(lmod.lora_up, 'weight'):
        lmod.lora_up.weight.data.uniform_(0.01, 0.1)

    params = list(lmod.parameters())
    assert len(params) > 0
    opt = torch.optim.SGD(params, lr=1e-2)

    base_w = base_lin.weight.detach().clone()
    l_snap = [p.detach().clone() for p in params]

    x = torch.randn(1, 4)
    # Directly use lora path to ensure output depends on lora params (lora_up initially non-zero)
    y = lmod.lora_up(lmod.lora_down(x))
    loss = y.sum()
    loss.backward()
    opt.step()

    # base unchanged
    assert torch.allclose(base_lin.weight.detach(), base_w)
    changed = any([not torch.allclose(a, b.detach()) for a, b in zip(l_snap, params)])
    assert changed


def test_controlnet_frozen_with_lora_optimizer_excludes_controlnet(monkeypatch):
    tmpdir = tempfile.mkdtemp()
    ckpt = os.path.join(tmpdir, 'ckpt.safetensors')
    open(ckpt, 'wb').close()

    cfg = ModelConfig(name_or_path='some_model', controlnet_streaming=False, controlnet_offload_strategy='none')
    model = ZImageModel('cpu', cfg)

    # Dummy control class
    class DummyControl(torch.nn.Module):
        def __init__(self):
            super().__init__()
            class Sub(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.zeros(2, 2))
            self.transformer = Sub()
            self.in_channels = 3
            self.control_layers = []
            self.control_in_dim = 1
            self.control_all_x_embedder = True

        def state_dict(self):
            return {'transformer.weight': self.transformer.weight}

        def load_state_dict(self, sd, strict=False):
            return ({}, {})

        def parameters(self):
            return [self.transformer.weight]

        def __call__(self, x, t, cap):
            return torch.zeros(1)

    # patch classes used in load_controlnet_transformer
    monkeypatch.setattr('extensions_built_in.diffusion_models.z_image.z_image.ZImageControlTransformer2DModel', lambda **k: DummyControl())
    class DummyBase:
        def __init__(self):
            self._p = torch.nn.Parameter(torch.zeros(2, 2))
            class Cfg:
                def to_dict(self):
                    return {}
            self.config = Cfg()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return DummyBase()

        def state_dict(self):
            return {'transformer.weight': self._p}

    monkeypatch.setattr('extensions_built_in.diffusion_models.z_image.z_image.ZImageTransformer2DModel', DummyBase)

    # patch safetensors.load_file
    monkeypatch.setattr('safetensors.torch.load_file', lambda path: {'transformer.weight': torch.ones(2,2)})

    # Load controlnet into model
    model.load_controlnet_transformer(tmpdir, 'ckpt.safetensors', freeze=True)

    # controlnet params frozen
    for p in model.controlnet.parameters():
        assert not p.requires_grad

    # Attach lora to a dummy transformer (don't rely on model.get_model_to_train in this unit test)
    class DummyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.Module()
            self.blocks.lin = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.blocks.lin(x)

    unet = DummyTransformer()
    net = LoRASpecialNetwork(text_encoder=torch.nn.Module(), unet=unet, lora_dim=2, alpha=2, transformer_only=True, is_transformer=True, base_model=model)
    net.force_to(torch.device('cpu'), dtype=torch.float32)
    net.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)

    params = net.prepare_optimizer_params(text_encoder_lr=None, unet_lr=1e-3, default_lr=1e-3)
    flat_params = [p for g in params for p in g['params']]

    # Ensure none of the controlnet parameters are included in optimizer param list
    control_params = list(model.controlnet.parameters())
    for cp in control_params:
        assert id(cp) not in [id(x) for x in flat_params]
