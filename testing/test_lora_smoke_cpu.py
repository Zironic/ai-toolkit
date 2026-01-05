import pytest
import torch
from toolkit.lora_special import LoRASpecialNetwork

@pytest.mark.smoke
def test_lora_smoke_single_cpu_step():
    # Minimal smoke test: create LoRA module and run a single optimizer step on CPU
    net = LoRASpecialNetwork(text_encoder=torch.nn.Module(), unet=torch.nn.Module(), lora_dim=4, alpha=4)
    from toolkit.lora_special import LoRAModule

    base_lin = torch.nn.Linear(8, 8)
    base_lin.requires_grad_(False)

    lmod = LoRAModule('smoke_lora', org_module=base_lin, multiplier=1.0, lora_dim=4, alpha=4, network=net)
    lmod.apply_to()

    # set small non-zero lora_up so gradients exist
    if hasattr(lmod, 'lora_up') and hasattr(lmod.lora_up, 'weight'):
        lmod.lora_up.weight.data.uniform_(0.01, 0.02)

    params = list(lmod.parameters())
    assert len(params) > 0

    opt = torch.optim.SGD(params, lr=1e-2)

    x = torch.randn(2, 8)
    y = lmod.lora_up(lmod.lora_down(x)).sum()
    y.backward()
    opt.step()

    # verify param update
    # (at least one param should have changed)
    changed = any((p.grad is not None and p.grad.abs().sum().item() > 0) for p in params)
    assert changed
