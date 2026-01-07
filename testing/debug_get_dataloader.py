import os
import tempfile
from PIL import Image
from types import SimpleNamespace

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from toolkit.data_loader import get_dataloader_from_datasets


def run():
    with tempfile.TemporaryDirectory() as td:
        img_path = os.path.join(td, 'img.png')
        Image.new('RGB', (8, 8), (255, 255, 255)).save(img_path)
        sd = SimpleNamespace()
        sd.get_bucket_divisibility = lambda : 1
        sd.encode_prompt = lambda *a, **k: __import__('toolkit').prompt_utils.PromptEmbeds(__import__('torch').zeros((1,16,8)))
        cfgs = [{'folder_path': td}]
        dl = get_dataloader_from_datasets(cfgs, 1, sd)
        print('dl:', dl)

if __name__ == '__main__':
    run()
