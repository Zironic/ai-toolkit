"""
Minimal shim for ComfyUI dependencies to allow standalone SAM3 usage.
This provides basic implementations of comfy.model_management functions.
"""
import torch
from pathlib import Path


class ModelManagement:
    """Mock ComfyUI model_management module."""
    
    @staticmethod
    def get_torch_device():
        """Get the torch device to use for model inference."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def unet_offload_device():
        """Get the device to offload models to when not in use."""
        return torch.device("cpu")
    
    @staticmethod
    def load_models_gpu(models):
        """Load models to GPU (no-op in standalone mode)."""
        pass
    
    @staticmethod
    def module_size(module):
        """Calculate module size in bytes."""
        param_size = sum(p.nelement() * p.element_size() for p in module.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in module.buffers())
        return param_size + buffer_size


# Create mock comfy module
class ComfyShim:
    model_management = ModelManagement()


# Mock folder_paths module
class FolderPaths:
    base_path = str(Path(__file__).parent.parent)  # AI-Toolkit root
