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
        # Record that we were called and return simple latents as [B,C,H,W]
        self.calls += len(imgs)
        outs = []
        for img in imgs:
            # return small latent of shape (C=4, H_lat=16, W_lat=16)
            outs.append(torch.randn(4, 16, 16))
        return torch.stack(outs, dim=0)


class DummyTrainer(SDTrainer):
    def __init__(self):
        # minimal init: attach sd and model_config and minimal data_loader
        self.sd = DummySDV()
        self.model_config = types.SimpleNamespace(control_use_tiling=False, control_tiling_size=256, control_tiling_overlap=32)
        # simple data_loader stub; get_dataloader_datasets reads .dataset
        self.data_loader = types.SimpleNamespace(dataset=[types.SimpleNamespace(dataset_config=None, dataset_path='dummy', file_list=[])])


def test_precompute_zimage_controls(tmp_path):
    # setup dummy dataset config with precompute flag
    cfg = DatasetConfig(dataset_path=str(tmp_path), control_precompute_control=True)
    ds = types.SimpleNamespace(dataset_config=cfg, dataset_path=str(tmp_path), file_list=[])

    # create two small dummy images on disk so FileItemDTO can read sizes
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    from PIL import Image
    Image.new('RGB', (512, 512), color=(73, 109, 137)).save(img1)
    Image.new('RGB', (512, 512), color=(73, 109, 137)).save(img2)

    # create two file items with control tensors (we'll set tensors ourselves)
    fi1 = FileItemDTO(path=str(img1), dataset_config=cfg)
    fi2 = FileItemDTO(path=str(img2), dataset_config=cfg)
    # set control tensors (C,H,W)
    fi1.control_tensor = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
    fi2.control_tensor = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
    ds.file_list = [fi1, fi2]

    t = DummyTrainer()
    t.data_loader = types.SimpleNamespace(dataset=[ds])

    # run precompute
    t._precompute_zimage_control_contexts()

    # ensure SD.encode_control_images_videox called for both images
    assert t.sd.calls == 2
    # ensure file items have precomputed contexts dict with 512 key
    for fi in ds.file_list:
        assert getattr(fi, '_preencoded_zimage_control_contexts', None) is not None
        assert isinstance(fi._preencoded_zimage_control_contexts, dict)
        assert 512 in fi._preencoded_zimage_control_contexts
        ctx = fi._preencoded_zimage_control_contexts[512]
        assert isinstance(ctx, torch.Tensor)
        # shape should be 4D or 5D depending on assembly; expect (C,F,H,W) or (C,1,H,W) or 5D (C,F,H,W)
        assert ctx.ndim in (3, 4, 5)
