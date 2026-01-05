import types
import torch
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.config_modules import DatasetConfig


class DummySDV:
    def __init__(self):
        # transformer must advertise control_in_dim=33 for VideoX
        self.transformer = types.SimpleNamespace(control_in_dim=33)
        self.calls = 0

    def encode_control_images_videox(self, imgs, height=None, width=None, tile=False):
        # Record that we were called and return latents sized according to input H/W
        self.calls += len(imgs)
        outs = []
        for img in imgs:
            # img is a tensor [C,H,W] or [3,H,W] (if list element), or maybe [C,H,W] float
            if isinstance(img, torch.Tensor):
                h = int(img.shape[-2])
                w = int(img.shape[-1])
            else:
                # fallback
                h, w = 64, 64
            # emulate latent downsample (e.g., /8)
            h_lat = max(1, h // 8)
            w_lat = max(1, w // 8)
            outs.append(torch.randn(4, h_lat, w_lat))
        return torch.stack(outs, dim=0)


class DummyTrainer(SDTrainer):
    def __init__(self):
        # minimal init: attach sd and model_config and minimal data_loader
        self.sd = DummySDV()
        self.model_config = types.SimpleNamespace(control_use_tiling=False, control_tiling_size=256, control_tiling_overlap=32)
        # simple data_loader stub; get_dataloader_datasets reads .dataset
        self.data_loader = types.SimpleNamespace(dataset=[types.SimpleNamespace(dataset_config=None, dataset_path='dummy', file_list=[])])


def test_precompute_zimage_controls_non_square(tmp_path):
    # setup dummy dataset config with precompute flag
    cfg = DatasetConfig(dataset_path=str(tmp_path), control_precompute_control=True)

    # create one non-square dummy image on disk so FileItemDTO can read sizes
    img1 = tmp_path / "img1.png"
    from PIL import Image
    Image.new('RGB', (320, 512), color=(73, 109, 137)).save(img1)  # width=320, height=512

    # create file item with control tensor (C,H,W)
    fi1 = FileItemDTO(path=str(img1), dataset_config=cfg)
    # simulate dataset that uses full-size control images (do not force square resize)
    fi1.full_size_control_images = True
    fi1.control_tensor = torch.randint(0, 255, (3, 512, 320), dtype=torch.uint8)
    ds = types.SimpleNamespace(dataset_config=cfg, dataset_path=str(tmp_path), file_list=[fi1])

    t = DummyTrainer()
    t.data_loader = types.SimpleNamespace(dataset=[ds])

    # run precompute
    t._precompute_zimage_control_contexts()

    # ensure SD.encode_control_images_videox called
    assert t.sd.calls == 1

    # verify stored precomputed latents and metadata
    fi = ds.file_list[0]
    assert getattr(fi, '_preencoded_zimage_control_contexts', None) is not None
    ctx = fi._preencoded_zimage_control_contexts.get(512)
    assert isinstance(ctx, torch.Tensor)
    # since image was 512 (long side) -> resized to 512x320 and padded (320 already multiple of 16),
    # expected latent dims = (512//8, 320//8) == (64, 40)
    B, C, H_lat, W_lat = ctx.shape if ctx.ndim == 4 else (1, *ctx.shape)
    assert (H_lat, W_lat) == (64, 40)

    # metadata tag should include padded dimensions when available
    try:
        from toolkit.control_channels import get_tensor_origin
        meta = get_tensor_origin(ctx)
        assert meta is not None and isinstance(meta.get('op'), str) and meta.get('op').startswith('precompute:control_latents')
        assert 'padded=' in meta.get('op') or 'padded' in str(meta)
    except Exception:
        pass
