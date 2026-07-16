import os

import torch


def configure_cacheable_tensor_subclasses() -> tuple[str, ...]:
    """Version cache-safe tensor-subclass constructors used by CUDA graphs.

    AOTAutograd rejects non-Torch ``call_function`` targets unless their
    semantics are explicitly included in the persistent cache key. TorchAO's
    ``Float8Tensor`` constructor is emitted while functional_call rebuilds an
    arena weight from its qdata and scale leaves. That reconstruction is pure,
    so key it by both the Torch and TorchAO versions instead of bypassing the
    cache (or weakening strict cache validation). Arena's gradient-safe FP8
    path also emits captured custom-autograd forward/backward graphs; enable
    PyTorch's supported cache-keying mode for those graph bodies.
    """
    try:
        import torchao
        from torchao.quantization import Float8Tensor
        from torch._inductor import config as inductor_config
    except (ImportError, AttributeError):
        return ()

    name = f"{Float8Tensor.__module__}.{Float8Tensor.__name__}"
    cache_hash = (
        "ai-toolkit-float8-tensor-unflatten-v1;"
        f"torch={torch.__version__};"
        f"torchao={getattr(torchao, '__version__', 'unknown')}"
    )
    marked = dict(inductor_config.unsafe_marked_cacheable_functions)
    marked[name] = cache_hash
    inductor_config.unsafe_marked_cacheable_functions = marked
    torch._functorch.config.autograd_cache_allow_custom_autograd_functions = True
    return (name,)


def configure_cuda_only_inductor() -> None:
    """Configure CUDA compilation without requiring a CPU toolchain."""
    torch._dynamo.config.suppress_errors = False
    configure_cacheable_tensor_subclasses()
    if os.name == "nt":
        # Inductor otherwise dry-compiles a CPU vector-ISA probe even when the
        # requested graph is CUDA-only. This does not enable a CPU fallback.
        from torch._inductor import config as inductor_config

        inductor_config.cpp.vec_isa_ok = False


def configure_quantized_compile_tuning(model_config) -> bool | None:
    """Apply an explicit coordinate-descent policy after TorchAO quantization."""
    if not getattr(model_config, "compile", False):
        return None
    if not getattr(model_config, "quantize", False):
        return None

    requested = getattr(model_config, "compile_coordinate_descent", None)
    if requested is None:
        return None

    enabled = bool(requested)
    torch._inductor.config.coordinate_descent_tuning = enabled
    torch._inductor.config.coordinate_descent_check_all_directions = enabled
    return enabled
