import types
import torch
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.config_modules import DatasetConfig


class DummySDV:
    def __init__(self):
        self.transformer = types.SimpleNamespace(control_in_dim=33)
        self.calls = 0

    def encode_control_images_videox(self, imgs, height=None, width=None, tile=False):
        self.calls += len(imgs)
        outs = []
        for img in imgs:
            if isinstance(img, torch.Tensor):
                h = int(img.shape[-2])
                w = int(img.shape[-1])
            else:
                h, w = 64, 64
            # emulate latent downsample (/8)
            h_lat = max(1, h // 8)
            w_lat = max(1, w // 8)
            outs.append(torch.randn(4, h_lat, w_lat))
        return torch.stack(outs, dim=0)


class DummyTrainer(SDTrainer):
    def __init__(self):
        self.sd = DummySDV()
        self.model_config = types.SimpleNamespace(control_use_tiling=False, control_tiling_size=256, control_tiling_overlap=32)
        self.data_loader = types.SimpleNamespace(dataset=[types.SimpleNamespace(dataset_config=None, dataset_path='dummy', file_list=[])])


def test_precompute_uses_dataset_crop_size(tmp_path):
    cfg = DatasetConfig(dataset_path=str(tmp_path), control_precompute_control=True)
    img1 = tmp_path / "img1.png"
    from PIL import Image
    Image.new('RGB', (768, 512), color=(73, 109, 137)).save(img1)

    fi1 = FileItemDTO(path=str(img1), dataset_config=cfg)
    # Simulate dataset bucket/crop to 768x512
    fi1.full_size_control_images = True
    fi1.crop_width = 768
    fi1.crop_height = 512
    fi1.control_tensor = torch.randint(0, 255, (3, 512, 768), dtype=torch.uint8)

    ds = types.SimpleNamespace(dataset_config=cfg, dataset_path=str(tmp_path), file_list=[fi1])
    t = DummyTrainer()
    t.data_loader = types.SimpleNamespace(dataset=[ds])

    t._precompute_zimage_control_contexts()

    # expect the precompute to have used size 768 (long-side), not default 512
    assert t.sd.calls == 1
    ctx = fi1._preencoded_zimage_control_contexts.get(768)
    assert ctx is not None and isinstance(ctx, torch.Tensor)
    # latent spatial dims should match 768//8 = 96 and 512//8 = 64 (H_lat, W_lat if H=512, W=768)
    B, C, H_lat, W_lat = ctx.shape if ctx.ndim == 4 else (1, *ctx.shape)
    assert (H_lat, W_lat) == (64, 96)
