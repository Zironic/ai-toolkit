from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module

global_accelerator = None


def get_accelerator() -> Accelerator:
    global global_accelerator
    if global_accelerator is None:
        global_accelerator = Accelerator()
    return global_accelerator

def unwrap_model(model):
    if model is None:
        return None
    try:
        accelerator = get_accelerator()
        unwrapped = accelerator.unwrap_model(model)
        unwrapped = unwrapped._orig_mod if is_compiled_module(unwrapped) else unwrapped
        return unwrapped
    except Exception as e:
        # If unwrapping fails, return the original model instead of None
        return model
