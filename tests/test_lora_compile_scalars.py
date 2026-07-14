import pytest
import torch

from scripts.smoke_runtime import configure_cuda_smoke_inductor, gpu_lock
from toolkit.lora_special import LoRAModule
from toolkit.models.DoRA import DoRAModule
from toolkit.models.lokr import LokrModule


class _Network:
    network_type = "lora"


def test_tensor_alpha_becomes_python_scale_for_compile():
    network = _Network()
    original = torch.nn.Linear(8, 8, bias=False)

    module = LoRAModule(
        "compile_scalar",
        original,
        lora_dim=4,
        alpha=torch.tensor(8, dtype=torch.bfloat16),
        network=network,
    )

    assert type(module.scale) is float
    assert module.scale == 2.0
    assert module._runtime_scale.item() == 2.0
    assert "_runtime_scale" not in module.state_dict()
    assert not hasattr(module, "scalar")


def test_dora_unity_scale_is_not_an_unregistered_tensor():
    network = _Network()
    original = torch.nn.Linear(8, 8, bias=False)

    module = DoRAModule(
        "compile_scalar_dora",
        original,
        lora_dim=4,
        alpha=torch.tensor(8, dtype=torch.bfloat16),
        network=network,
    )

    assert type(module.scale) is float
    assert module.scale == 2.0
    assert module._runtime_scale.item() == 2.0
    assert "_runtime_scale" not in module.state_dict()
    assert not hasattr(module, "scalar")


def test_lokr_scale_stays_a_python_scalar():
    network = _Network()
    original = torch.nn.Linear(8, 8, bias=False)

    module = LokrModule(
        "compile_scalar_lokr",
        original,
        lora_dim=2,
        alpha=torch.tensor(4, dtype=torch.bfloat16),
        network=network,
    )

    assert type(module.scale) is float
    assert module.scale == 1.0
    assert module._runtime_scale.item() == 1.0
    assert "_runtime_scale" not in module.state_dict()
    assert module.get_weight().device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA compile")
def test_lora_tensor_alpha_compiles_without_cpu_codegen():
    configure_cuda_smoke_inductor()
    network = _Network()
    network.is_lorm = False
    network.is_active = True
    network.is_merged_in = False
    network._multiplier = 1.0
    network.vector_gates = None
    network.torch_multiplier = torch.ones(1, device="cuda", dtype=torch.float32)
    original = torch.nn.Linear(
        8, 8, bias=False, device="cuda", dtype=torch.bfloat16
    )
    module = LoRAModule(
        "compile_scalar_cuda",
        original,
        lora_dim=4,
        alpha=torch.tensor(8, dtype=torch.bfloat16),
        network=network,
    ).to("cuda")
    assert module._runtime_scale.device.type == "cuda"

    compiled = torch.compile(
        lambda value: module.functional_forward(original.forward, value),
        fullgraph=True,
        dynamic=False,
    )
    value = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)

    with gpu_lock("test_lora_compile_scalars", wait=True):
        result = compiled(value)
        torch.cuda.synchronize()

    assert result.device.type == "cuda"
