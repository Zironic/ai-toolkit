from pathlib import Path
import sys
# ensure repo root is on sys.path so `toolkit` package can be imported when running scripts directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from toolkit.config_modules import DatasetConfig
from toolkit.data_transfer_object.data_loader import FileItemDTO

p = Path('C:/Users/johan/AppData/Local/Temp/pytest-of-johan/pytest-13/test_precomputed_residuals_are0')
print('p exists:', p.exists())
if not p.exists():
    print('No test folder found; aborting')
    raise SystemExit(0)

print('dir contents:', [x.name for x in p.iterdir()])
resdir = p / 'residuals'
print('resdir exists:', resdir.exists())
if resdir.exists():
    print('resdir contents:', [x.name for x in resdir.iterdir()])

img = p / 'image_0001.jpg'
print('image exists:', img.exists())

candidate = resdir / (img.stem + '_residuals.pt')
print('candidate path:', candidate)
print('candidate exists:', candidate.exists())

if candidate.exists():
    try:
        loaded = torch.load(str(candidate), map_location='cpu')
        print('loaded type:', type(loaded))
        if isinstance(loaded, (list, tuple)):
            print('elements types:', [type(x) for x in loaded])
            for i, x in enumerate(loaded):
                print(f'elem {i} shape/type:', (getattr(x, 'shape', None), type(x)))
        else:
            print('loaded object:', repr(loaded)[:200])
    except Exception as e:
        print('torch.load failed:', e)

# Instantiate FileItemDTO the same way the test does
print('\nConstructing FileItemDTO')
ds = DatasetConfig(control_residuals_path=str(resdir))
print('ds.control_residuals_path:', getattr(ds, 'control_residuals_path', None))
fi = FileItemDTO(path=str(img), dataset_config=ds, dataset_root=str(p))
print('FileItemDTO.control_residuals:', getattr(fi, 'control_residuals', None))

# debug dataset_config on fi
print('fi.dataset_config exists:', getattr(fi, 'dataset_config', None) is not None)
if getattr(fi, 'dataset_config', None) is not None:
    print('fi.dataset_config.control_residuals_path:', getattr(fi.dataset_config, 'control_residuals_path', None))

# If None, show attributes set on fi
print('fi attributes:', [k for k in dir(fi) if not k.startswith('_') and k in ['control_residuals','control_path']])
print('fi.control_path:', getattr(fi, 'control_path', None))
print('Done')
