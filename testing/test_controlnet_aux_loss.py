import torch
from toolkit.controlnet_aux import compute_control_edge_loss
from toolkit.config_modules import TrainConfig
from toolkit.data_transfer_object.data_loader import FileItemDTO, DataLoaderBatchDTO
from PIL import Image
from torchvision import transforms
from pathlib import Path


def test_compute_control_edge_loss_scales(tmp_path: Path):
    # create a simple image file so FileItemDTO can be constructed
    img_path = tmp_path / "img1.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    # construct a FileItem and set tensors manually
    ds = type('D', (), {'num_frames': 1})()
    from toolkit.config_modules import DatasetConfig
    ds = DatasetConfig()
    file_item = FileItemDTO(path=str(img_path), dataset_config=ds, size_database={}, dataset_root=str(tmp_path))

    # set an image tensor and a fake control tensor
    file_item.tensor = torch.rand(3, 64, 64)
    file_item.control_tensor = torch.rand(1, 64, 64)

    batch = DataLoaderBatchDTO(file_items=[file_item])

    # compute aux loss
    aux = compute_control_edge_loss(batch.tensor, batch.control_tensor, device=torch.device('cpu'))
    assert aux.item() >= 0.0


def test_control_aux_integration_with_trainer(monkeypatch, tmp_path: Path):
    # create an image file and file_item
    img_path = tmp_path / "img2.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)
    from toolkit.config_modules import DatasetConfig
    ds = DatasetConfig()
    file_item = FileItemDTO(path=str(img_path), dataset_config=ds, size_database={}, dataset_root=str(tmp_path))
    file_item.tensor = torch.zeros(3, 64, 64)
    file_item.control_tensor = torch.ones(1, 64, 64) * 0.5
    batch = DataLoaderBatchDTO(file_items=[file_item])

    # Create fake trainer-like object
    class Fake:
        pass

    fake = Fake()
    fake.train_config = TrainConfig()
    fake.train_config.controlnet_aux_loss = 'edge'
    fake.train_config.controlnet_aux_loss_weight = 2.0
    fake.train_config.loss_type = 'mse'
    fake.train_config.snr_gamma = None
    fake.device_torch = torch.device('cpu')
    # minimal sd stub for calculate_loss
    from types import SimpleNamespace
    fake.sd = SimpleNamespace(prediction_type='epsilon', is_flow_matching=False)
    fake.dfe = None
    fake.adapter = None

    # Create zero pred/target so base loss is zero
    noise_pred = torch.zeros(1, 4, 16, 16)
    noise = torch.zeros_like(noise_pred)
    noisy_latents = torch.zeros_like(noise_pred)
    timesteps = torch.tensor([10])

    # monkeypatch the helper to return a known value
    import toolkit.controlnet_aux as auxmod

    monkeypatch.setattr(auxmod, 'compute_control_edge_loss', lambda img, ctrl, device: torch.tensor(0.123))

    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    # Call SDTrainer.calculate_loss as an unbound function, passing fake as self
    loss = SDTrainer.calculate_loss(fake, noise_pred, noise, noisy_latents, timesteps, batch)
    # expected aux contribution = 0.123 * weight 2.0
    assert abs(loss.item() - (0.123 * 2.0)) < 1e-6, f"Unexpected aux loss {loss.item()}"