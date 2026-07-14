import pytest
import torch

from scripts.smoke_runtime import configure_cuda_smoke_inductor, gpu_lock
from toolkit.lora_special import LoRAModule


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
