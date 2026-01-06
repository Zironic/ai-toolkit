import torch
from toolkit.splitflux import complementary_loss


class DummyLora:
    def __init__(self, down: torch.Tensor, up: torch.Tensor, name: str):
        # emulate attributes used by helper
        self.lora_down = torch.nn.Parameter(down)
        self.lora_up = torch.nn.Parameter(up)
        self.lora_name = name


class DummyNet:
    def __init__(self, modules):
        # modules: list of DummyLora
        self.text_encoder_loras = []
        self.unet_loras = modules


def test_complementary_loss_zero_for_orthogonal():
    # create two modules with orthogonal down/up factors
    # down: r1 x in ; up: out x r1
    in_dim = 6
    out_dim = 8
    # content: rank 2
    D1 = torch.tensor([[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]])
    U1 = torch.tensor([[1.0, 0], [0, 1.0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

    # style: rank 1 orthogonal to both rows/cols
    D2 = torch.tensor([[0.0, 0, 1.0, 0, 0, 0]])
    U2 = torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

    m1 = DummyLora(D1, U1, 'lora_unet.block0.mod1')
    m2 = DummyLora(D2, U2, 'lora_unet.block0.mod1')

    net1 = DummyNet([m1])
    net2 = DummyNet([m2])

    loss = complementary_loss(net1, net2, weight=1.0)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_complementary_loss_positive_when_overlap():
    in_dim = 6
    out_dim = 8
    # content: rank 2
    D1 = torch.tensor([[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]])
    U1 = torch.tensor([[1.0, 0], [0, 1.0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

    # style: rank 1 overlapping with first component
    D2 = torch.tensor([[1.0, 0, 0, 0, 0, 0]])
    U2 = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

    m1 = DummyLora(D1, U1, 'lora_unet.block0.mod1')
    m2 = DummyLora(D2, U2, 'lora_unet.block0.mod1')

    net1 = DummyNet([m1])
    net2 = DummyNet([m2])

    loss = complementary_loss(net1, net2, weight=1.0)
    assert loss.item() > 0


def test_complementary_loss_handles_no_pairs():
    net1 = DummyNet([])
    net2 = DummyNet([])
    loss = complementary_loss(net1, net2)
    assert loss.item() == 0.0