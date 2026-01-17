# Minimal stubs for VideoX distributed training utilities
# These are only needed for multi-GPU setups; we provide no-op versions for single-GPU

import torch
import torch.nn as nn

class ZMultiGPUsSingleStreamAttnProcessor:
    """Stub for multi-GPU attention processor - not used in single-GPU mode"""
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Multi-GPU attention processor called but multi-GPU is not enabled")

def get_sequence_parallel_rank():
    """Return 0 for single-GPU (no sequence parallelism)"""
    return 0

def get_sequence_parallel_world_size():
    """Return 1 for single-GPU (no sequence parallelism)"""
    return 1

def initialize_model_parallel(*args, **kwargs):
    """No-op for single-GPU"""
    pass
