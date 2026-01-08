import os
import tempfile
import torch
from PIL import Image

from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset
from toolkit.prompt_utils import PromptEmbeds


class FakeSD:
    def __init__(self):
        self.device_torch = 'cpu'
        self.torch_dtype = torch.float32
        self.encode_control_in_text_embeddings = False
        self.has_multiple_control_images = False

    def get_bucket_divisibility(self):
        return 1

    def set_device_state_preset(self, *_args, **_kwargs):
        return

    def encode_prompt(self, prompt, **_kwargs):
        # Return a tiny PromptEmbeds object (batch size 1)
        # We simulate the text embeddings tensor with shape [1, seq, dim]
        t = torch.zeros((1, 16, 8))
        return PromptEmbeds(t)


def test_dop_cache_path_and_load():
    with tempfile.TemporaryDirectory() as td:
        # create a tiny test image
        img_path = os.path.join(td, 'img.png')
        Image.new('RGB', (8, 8), (255, 255, 255)).save(img_path)

        # dataset config pointing to the folder
        fake_sd = FakeSD()

        # Create a minimal FileItem-like object that uses the TextEmbedding mixin methods
        from toolkit.dataloader_mixins import TextEmbeddingFileItemDTOMixin

        class DummyFile(TextEmbeddingFileItemDTOMixin):
            def __init__(self, path, caption):
                # don't call super; just set expected attributes
                self.path = path
                self.caption = caption
                self.encode_control_in_text_embeddings = False
                self.text_embedding_space_version = 'sd1'
                self.text_embedding_version = 1
                self._text_embedding_path = None
                self._dop_text_embedding_path = None
                self.is_text_embedding_cached = True
                self.dop_prompt_embeds = None
                self.dataset_config = SimpleNamespace(caption_ext='txt')

        fi = DummyFile(img_path, "a photo of [trigger] in the wild")

        # ensure base text embedding path differs from dop path
        base_path = fi.get_text_embedding_path(recalculate=True)
        dop_path = fi.get_text_embedding_path(recalculate=True, dop_class='DOP_CLASS')
        assert base_path != dop_path

        # simulate creating dop embedding and saving to dop_path
        if os.path.exists(dop_path):
            os.remove(dop_path)
        dop_caption = fi.caption.replace('[trigger]', 'DOP_CLASS')
        pe = fake_sd.encode_prompt(dop_caption)
        pe.save(dop_path)

        # now load via helper
        fi.load_dop_prompt_embedding('DOP_CLASS')
        assert fi.dop_prompt_embeds is not None
        assert os.path.exists(dop_path)


def test_dop_cache_key_changes_on_trigger_and_class():
    # reuse the DummyFile from above by constructing a new instance
    from toolkit.dataloader_mixins import TextEmbeddingFileItemDTOMixin
    class DummyFile(TextEmbeddingFileItemDTOMixin):
        def __init__(self, path, caption):
            self.path = path
            self.caption = caption
            self.encode_control_in_text_embeddings = False
            self.text_embedding_space_version = 'sd1'
            self.text_embedding_version = 1
            self._text_embedding_path = None
            self._dop_text_embedding_path = None
            self.dataset_config = SimpleNamespace(caption_ext='txt')

    tmp_dir = tempfile.TemporaryDirectory()
    img_path = os.path.join(tmp_dir.name, 'img2.png')
    Image.new('RGB', (8, 8), (255, 255, 255)).save(img_path)
    fi = DummyFile(img_path, 'original caption')

    p0 = fi.get_text_embedding_path(recalculate=True, dop_class='classA')
    from toolkit.cache_utils import compute_param_digest
    dop_repl_digest = compute_param_digest({'trigger_word': 't1', 'replacements': []})
    p1 = fi.get_text_embedding_path(recalculate=True, dop_class='classA', trigger_word='t1', dop_replacements_digest=dop_repl_digest)
    assert p0 != p1

    p2 = fi.get_text_embedding_path(recalculate=True, dop_class='classB', trigger_word='t1', dop_replacements_digest=dop_repl_digest)
    assert p1 != p2

    # change caption
    fi.caption = 'changed caption'
    p3 = fi.get_text_embedding_path(recalculate=True, dop_class='classB', trigger_word='t1', dop_replacements_digest=dop_repl_digest)
    assert p2 != p3

    # empty trigger and dop class handling
    p4 = fi.get_text_embedding_path(recalculate=True, dop_class='', trigger_word='', dop_replacements_digest=compute_param_digest({'trigger_word': '', 'replacements': []}))
    p5 = fi.get_text_embedding_path(recalculate=True, dop_class=None, trigger_word=None, dop_replacements_digest=None)
    assert p4 != p5
    tmp_dir.cleanup()
