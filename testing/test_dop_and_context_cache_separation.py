import os
import types
import torch
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.config_modules import DatasetConfig
from toolkit.prompt_utils import PromptEmbeds


def test_dop_and_context_cache_are_separate(tmp_path):
    # Create dataset config enabling control precompute + persisting control contexts
    cfg = DatasetConfig(dataset_path=str(tmp_path), control_precompute_control=True)
    cfg.cache_control_contexts_to_disk = True
    cfg.cache_text_embeddings = True  # allow saving DOP embeds to _t_e_cache

    # create one small dummy image on disk so FileItemDTO can read sizes
    img1 = tmp_path / "img1.png"
    from PIL import Image

    Image.new('RGB', (512, 512), color=(73, 109, 137)).save(img1)

    fi = FileItemDTO(path=str(img1), dataset_config=cfg)
    fi.control_tensor = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)

    # Simulate precompute: save a control_context file via save_control_contexts
    contexts = {512: torch.randn(33, 512 // 8, 512 // 8)}
    fi.save_control_contexts(contexts)

    # Ensure _context_cache produced files
    ctx_dir = os.path.join(tmp_path, '_context_cache')
    ctx_files = [f for f in os.listdir(ctx_dir)] if os.path.isdir(ctx_dir) else []
    assert len(ctx_files) > 0, f"Expected persisted context files in {ctx_dir}, found: {ctx_files}"

    # Now create and save a DOP prompt embedding into _t_e_cache
    dop_path = fi.get_text_embedding_path(recalculate=True, dop_class='DOP_CLASS')
    # make sure target dir exists
    os.makedirs(os.path.dirname(dop_path), exist_ok=True)
    pe = PromptEmbeds(torch.randn(1, 77, 768))
    pe.save(dop_path)

    te_dir = os.path.join(tmp_path, '_t_e_cache')
    te_files = [f for f in os.listdir(te_dir)] if os.path.isdir(te_dir) else []
    assert len(te_files) > 0, f"Expected DOP prompt embed files in {te_dir}, found: {te_files}"

    # Ensure the caches are separate (no file is present in both directories with same name)
    overlap = set(ctx_files).intersection(set(te_files))
    assert len(overlap) == 0, f"Cache files should be separate, but found overlaps: {overlap}"