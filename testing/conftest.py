# Lightweight test fixtures to stub heavy external dependencies (diffusers) for unit tests
import sys
import types
from pathlib import Path

# Sanitize sys.modules: remove modules with malformed __file__ attributes (these can
# break inspect.getsource during real torch import, causing errors like
# "AttributeError: type object '__file__' has no attribute 'endswith'"). This generally
# happens when tests or earlier setup assign non-string __file__ values to modules.
for _name, _mod in list(sys.modules.items()):
    try:
        if hasattr(_mod, '__file__'):
            f = getattr(_mod, '__file__')
            if f is not None and not isinstance(f, str):
                # remove suspicious module to avoid inspect failures
                del sys.modules[_name]
    except Exception:
        # be conservative: ignore any odd modules we can't inspect
        pass

# Ensure 'diffusers' module has the minimal set of attributes needed by unit tests.
# If real diffusers is installed but missing the expected symbols, add safe stubs.
# Only inject a fake 'diffusers' when the real package is not installed to avoid
# shadowing full installs that tests may rely on.
import importlib.util
if importlib.util.find_spec('diffusers') is None:
    fake_diff = types.ModuleType('diffusers')
    # set a minimal ModuleSpec so importlib.find_spec and other import-time probes behave
    fake_diff.__spec__ = importlib.util.spec_from_loader('diffusers', loader=None)
    # ensure __file__ is a string to avoid inspect issues
    fake_diff.__file__ = str(Path(__file__).resolve())
    # Provide a flexible getattr handler and a few explicit placeholders so imports like
    # `from diffusers import X` won't crash during test collection. Missing names will
    # yield lightweight placeholder classes.
    def _fake_getattr(name):
        return type(name, (), {})
    fake_diff.__getattr__ = _fake_getattr
    # minimal explicit class/placeholders commonly used
    fake_diff.UNet2DConditionModel = type('UNet2DConditionModel', (), {})
    fake_diff.PixArtTransformer2DModel = type('PixArtTransformer2DModel', (), {})
    fake_diff.AuraFlowTransformer2DModel = type('AuraFlowTransformer2DModel', (), {})
    fake_diff.WanTransformer3DModel = type('WanTransformer3DModel', (), {})
    fake_diff.Transformer2DModel = type('Transformer2DModel', (), {})
    fake_diff.T2IAdapter = object
    fake_diff.AutoencoderTiny = object
    fake_diff.AutoencoderKL = type('AutoencoderKL', (), {})
    fake_diff.ControlNetModel = type('ControlNetModel', (), {})
    fake_diff.EMAModel = object
    fake_diff.StableDiffusionPipeline = type('StableDiffusionPipeline', (), {})
    fake_diff.StableDiffusionXLPipeline = type('StableDiffusionXLPipeline', (), {})
    fake_diff.PixArtAlphaPipeline = type('PixArtAlphaPipeline', (), {})
    fake_diff.PixArtSigmaPipeline = type('PixArtSigmaPipeline', (), {})
    fake_diff.FluxFillPipeline = type('FluxFillPipeline', (), {})
    fake_diff.DiffusionPipeline = type('DiffusionPipeline', (), {})

    # Schedulers placeholders
    for _name in ['DDPMScheduler','EulerAncestralDiscreteScheduler','DPMSolverMultistepScheduler','DPMSolverSinglestepScheduler','LMSDiscreteScheduler','PNDMScheduler','DDIMScheduler','EulerDiscreteScheduler','HeunDiscreteScheduler','KDPM2DiscreteScheduler','KDPM2AncestralDiscreteScheduler']:
        setattr(fake_diff, _name, type(_name, (), {}))

    # Inject fake module into sys.modules, overriding any real install for tests
    sys.modules['diffusers'] = fake_diff

# Also provide a minimal 'optimum' stub so importers that probe optimum (optimum.quant) don't execute
# heavyweight code during test collection. This keeps collection deterministic.
fake_optimum = types.ModuleType('optimum')
fake_optimum.__spec__ = importlib.util.spec_from_loader('optimum', loader=None)
# ensure __file__ is a string for consistency
fake_optimum.__file__ = str(Path(__file__).resolve())
sys.modules['optimum'] = fake_optimum
# Create stub packages under optimum for the submodules imported by toolkit
opt_quanto = types.ModuleType('optimum.quanto')
opt_quanto.__spec__ = importlib.util.spec_from_loader('optimum.quanto', loader=None)
# add minimal symbols used by toolkit.network_mixins
class QTensor:
    pass
class QBytesTensor:
    pass
opt_quanto.QTensor = QTensor
opt_quanto.QBytesTensor = QBytesTensor
sys.modules['optimum.quanto'] = opt_quanto
sys.modules['optimum.quanto.models'] = types.ModuleType('optimum.quanto.models')
sys.modules['optimum.quanto.models'].__spec__ = importlib.util.spec_from_loader('optimum.quanto.models', loader=None)
# Provide a minimal tensor module expected by some imports
opt_tensor = types.ModuleType('optimum.quanto.tensor')
opt_tensor.QTensor = QTensor
opt_tensor.QBytesTensor = QBytesTensor
opt_tensor.__spec__ = importlib.util.spec_from_loader('optimum.quanto.tensor', loader=None)
sys.modules['optimum.quanto.tensor'] = opt_tensor
# Backwards/other possible names
sys.modules['optimum.quant'] = types.ModuleType('optimum.quant')
sys.modules['optimum.quant'].__spec__ = importlib.util.spec_from_loader('optimum.quant', loader=None)
sys.modules['optimum.quant.models'] = types.ModuleType('optimum.quant.models')
sys.modules['optimum.quant.models'].__spec__ = importlib.util.spec_from_loader('optimum.quant.models', loader=None)




# Only inject fake diffusers submodules if the package isn't installed
if importlib.util.find_spec('diffusers') is None:
    # Ensure utils submodules used by toolkit exist
    if 'diffusers.utils.torch_utils' not in sys.modules:
        fake_utils = types.ModuleType('diffusers.utils')
        fake_utils.__file__ = str(Path(__file__).resolve())
        fake_torch_utils_mod = types.ModuleType('diffusers.utils.torch_utils')
        fake_torch_utils_mod.__file__ = str(Path(__file__).resolve())
        setattr(fake_torch_utils_mod, 'is_compiled_module', lambda m: False)
        sys.modules['diffusers.utils.torch_utils'] = fake_torch_utils_mod
        fake_utils.torch_utils = fake_torch_utils_mod
        sys.modules['diffusers.utils'] = fake_utils

    # Ensure pipelines submodules referenced in tests exist
    if 'diffusers.pipelines' not in sys.modules:
        fake_pipelines = types.ModuleType('diffusers.pipelines')
        fake_pipelines.__file__ = str(Path(__file__).resolve())
        sys.modules['diffusers.pipelines'] = fake_pipelines

    # minimal pixart_alpha bits used in tests
    if 'diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma' not in sys.modules:
        fake_pipeline_pixart_sigma = types.ModuleType('diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma')
        fake_pipeline_pixart_sigma.__file__ = str(Path(__file__).resolve())
        setattr(fake_pipeline_pixart_sigma, 'ASPECT_RATIO_1024_BIN', b'')
        setattr(fake_pipeline_pixart_sigma, 'ASPECT_RATIO_512_BIN', b'')
        setattr(fake_pipeline_pixart_sigma, 'ASPECT_RATIO_2048_BIN', b'')
        setattr(fake_pipeline_pixart_sigma, 'ASPECT_RATIO_256_BIN', b'')
        sys.modules['diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma'] = fake_pipeline_pixart_sigma

    # stable_diffusion_xl stub
    if 'diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl' not in sys.modules:
        fake_sdxl = types.ModuleType('diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl')
        fake_sdxl.__file__ = str(Path(__file__).resolve())
        setattr(fake_sdxl, 'rescale_noise_cfg', lambda *a, **k: None)
        sys.modules['diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl'] = fake_sdxl
