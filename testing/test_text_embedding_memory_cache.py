import os
import tempfile
from PIL import Image
import torch
from toolkit.config_modules import DatasetConfig
from toolkit.prompt_utils import PromptEmbeds
from toolkit.data_transfer_object.data_loader import FileItemDTO


def test_text_embedding_memory_cache_default_true(tmp_path):
    # create a dataset config with caching enabled
    td = str(tmp_path)
    img_path = os.path.join(td, 'img.png')
    Image.new('RGB', (8, 8), (255, 255, 255)).save(img_path)

    cfg = DatasetConfig(folder_path=td, cache_text_embeddings=True)
    # default cache_text_embeddings_to_memory should be True
    assert cfg.cache_text_embeddings_to_memory is True

    fi = FileItemDTO(path=img_path, dataset_config=cfg, dataset_root=td)
    fi.caption = "a test caption"
    fi.is_text_embedding_cached = True

    # prepare a saved embedding on disk
    p = fi.get_text_embedding_path(recalculate=True)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    pe = PromptEmbeds(torch.zeros((1, 16, 8)))
    pe.save(p)

    # load and ensure it stays after cleanup
    fi.load_prompt_embedding()
    assert fi.prompt_embeds is not None
    fi.cleanup_text_embedding()
    assert fi.prompt_embeds is not None


def test_text_embedding_memory_cache_can_be_disabled(tmp_path):
    td = str(tmp_path)
    img_path = os.path.join(td, 'img.png')
    Image.new('RGB', (8, 8), (255, 255, 255)).save(img_path)

    cfg = DatasetConfig(folder_path=td, cache_text_embeddings=True, cache_text_embeddings_to_memory=False)
    assert cfg.cache_text_embeddings_to_memory is False

    fi = FileItemDTO(path=img_path, dataset_config=cfg, dataset_root=td)
    fi.caption = "a test caption"
    fi.is_text_embedding_cached = True

    # prepare a saved embedding on disk
    p = fi.get_text_embedding_path(recalculate=True)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    pe = PromptEmbeds(torch.zeros((1, 16, 8)))
    pe.save(p)

    fi.load_prompt_embedding()
    assert fi.prompt_embeds is not None
    fi.cleanup_text_embedding()
    assert fi.prompt_embeds is None
