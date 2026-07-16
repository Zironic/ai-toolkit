import os
from typing import List
from toolkit.models.base_model import BaseModel
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.config_modules import ModelConfig
from toolkit.paths import TOOLKIT_ROOT
import importlib
import pkgutil

from toolkit.models.wan21 import Wan21, Wan21I2V
from toolkit.models.cogview4 import CogView4

BUILT_IN_MODELS = [
    Wan21,
    Wan21I2V,
    CogView4,
]


LEGACY_MODEL_ARCHES = {
    'sd1',
    'sd2',
    'sd3',
    'sdxl',
    'pixart',
    'pixart_sigma',
    'auraflow',
    'flux',
    'lumina2',
    'vega',
    'ssd',
}


def get_all_models() -> List[BaseModel]:
    extension_folders = ['extensions', 'extensions_built_in']

    # This will hold the classes from all extension modules
    all_model_classes: List[BaseModel] = BUILT_IN_MODELS

    # Iterate over all directories (i.e., packages) in the "extensions" directory
    for sub_dir in extension_folders:
        extensions_dir = os.path.join(TOOLKIT_ROOT, sub_dir)
        for (_, name, _) in pkgutil.iter_modules([extensions_dir]):
            try:
                # Import the module
                module = importlib.import_module(f"{sub_dir}.{name}")
                # Get the value of the AI_TOOLKIT_MODELS variable
                models = getattr(module, "AI_TOOLKIT_MODELS", None)
                # Check if the value is a list
                if isinstance(models, list):
                    # Iterate over the list and add the classes to the main list
                    all_model_classes.extend(models)
            except ImportError as e:
                print(f"Failed to import the {name} module. Error: {str(e)}")
    return all_model_classes


def get_model_class(config: ModelConfig):
    all_models = get_all_models()
    for ModelClass in all_models:
        if ModelClass.arch == config.arch:
            return ModelClass
    if config.arch in LEGACY_MODEL_ARCHES:
        return StableDiffusion
    raise ValueError(
        f"No model implementation is registered for architecture {config.arch!r}. "
        "Check model.arch and any extension import errors printed above."
    )


def resolve_text_embedding_space_version(config: ModelConfig) -> str:
    """Resolve a model's prompt-cache namespace without instantiating it.

    Both the TE worker and the skip-TE trainer need this value before model
    weights are loaded. New model extensions should implement the class-level
    ``get_text_embedding_space_version(model_config)`` hook. A string class
    attribute remains supported for existing third-party extensions.
    """
    model_class = get_model_class(config)

    declared_version = vars(model_class).get("text_embedding_space_version")
    if isinstance(declared_version, str):
        version = declared_version
    else:
        resolver = getattr(model_class, "get_text_embedding_space_version", None)
        if callable(resolver):
            version = resolver(config)
        else:
            version = str(config.arch)

    if not isinstance(version, str) or not version:
        raise ValueError(
            f"{model_class.__name__}.get_text_embedding_space_version() must "
            "return a non-empty string"
        )
    return version
