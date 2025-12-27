"""ControlNet offload abstraction and helpers.

Accelerate is integrated into the toolkit and is the primary, recommended
offload mechanism for production/DDP-safe flows. This module provides a safe,
minimal `manual_swap` implementation (CPU-only) for local/dev testing and unit
tests, and contains the abstraction points where Accelerate-based dispatch
will be integrated for production use. Includes debug prints so behavior is
observable in CI/unit tests.
"""
from typing import Optional, Tuple
import torch
import time


def is_accelerate_available() -> bool:
    try:
        import accelerate  # noqa: F401
        return True
    except Exception:
        return False


def _debug(msg: str, *args):
    # Use established project logging style (simple prints and prefixed tags).
    print(f"[CONTROLNET-OFFLOAD] {msg}".format(*args))


def offload_adapter(adapter: torch.nn.Module, strategy: str = "none") -> None:
    """Offload adapter according to chosen strategy.

    Supported strategies (initial implementation):
      - none: do nothing
      - manual_swap: move model to CPU by calling .to('cpu') on parameters/buffers
      - accelerate: Not implemented here; raises NotImplementedError if invoked and accelerate not available
      - memory_manager: Not implemented stub

    This function prints debug info about parameter devices and sizes.
    """
    _debug("offload_adapter(strategy=%s) called", strategy)

    if strategy == "none":
        _debug("strategy is 'none' — no action taken")
        return

    # manual_swap remains a lightweight fallback for CPU-only development/testing.

    if strategy == "manual_swap":
        _debug("Manually moving adapter to CPU")
        t0 = time.time()
        adapter.to(torch.device("cpu"))
        _debug("Moved adapter to CPU in %.3fs", time.time() - t0)
        return

    if strategy == "accelerate":
        if not is_accelerate_available():
            raise RuntimeError("Accelerate requested for offload but package not available")
        # Use the toolkit accelerator to get the current device and move the adapter there.
        try:
            from toolkit.accelerator import get_accelerator

            acc = get_accelerator()
            device = acc.device
            _debug("Moving adapter to accelerator device %s", device)
            t0 = time.time()
            adapter.to(device)
            _debug("Moved adapter to %s in %.3fs", device, time.time() - t0)
            return
        except Exception as e:
            _debug("Accelerate-based offload failed: %s", str(e))
            raise


    if strategy == "memory_manager":
        raise NotImplementedError("memory_manager-based offload is not yet implemented")

    raise ValueError(f"Unknown offload strategy: {strategy}")


def bring_adapter(adapter: torch.nn.Module, device: torch.device, strategy: str = "none") -> None:
    """Bring adapter to specified device (inverse of offload)."""
    _debug("bring_adapter(device=%s, strategy=%s) called", device, strategy)

    if strategy == "none":
        _debug("strategy is 'none' — moving adapter to %s", device)
        adapter.to(device)
        return

    if strategy == "manual_swap":
        t0 = time.time()
        adapter.to(device)
        _debug("Brought adapter to %s in %.3fs", device, time.time() - t0)
        return

    if strategy == "accelerate":
        try:
            from toolkit.accelerator import get_accelerator

            acc = get_accelerator()
            _debug("Using accelerator device %s to bring adapter", acc.device)
            t0 = time.time()
            adapter.to(device)
            _debug("Brought adapter to %s in %.3fs", device, time.time() - t0)
            return
        except Exception as e:
            _debug("Failed to bring adapter via accelerate: %s", str(e))
            raise

    if strategy == "memory_manager":
        raise NotImplementedError(f"bring_adapter for strategy {strategy} must be implemented separately")

    raise ValueError(f"Unknown offload strategy: {strategy}")


def compute_control_residuals(
    adapter: torch.nn.Module,
    control_tensor: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    device: Optional[torch.device] = None,
    residual_storage: str = "gpu",
) -> Tuple[torch.Tensor, ...]:
    """Compute control residuals using adapter and return detached tensors.

    This is a minimal, testable implementation that runs the adapter in eval mode
    and returns the adapter output(s) as detached tensors. Callers are expected to
    duplicate or reshape tensors appropriately for CFG usage.

    For the initial iteration, we keep the implementation simple and DDP-safe by
    avoiding any global in-place device moves — callers should manage adapter
    offload/bring-back at higher level.
    """
    if adapter is None:
        raise ValueError("adapter is required to compute control residuals")

    if device is None:
        device = noisy_latents.device

    _debug("compute_control_residuals: device=%s, residual_storage=%s", device, residual_storage)

    adapter_device = next(adapter.parameters(), None)
    if adapter_device is None:
        _debug("Adapter has no parameters; running forward on provided device")
    else:
        _debug("Adapter parameter example device: %s", adapter_device.device)

    # Move tensors to device for computation
    control_tensor = control_tensor.to(device)
    noisy_latents = noisy_latents.to(device)
    timesteps = timesteps.to(device)

    was_training = adapter.training
    adapter.eval()

    with torch.no_grad():
        # Assume adapter returns a tensor or tuple of tensors; support both
        out = adapter(control_tensor, noisy_latents, timesteps)

    adapter.train(was_training)

    # Normalize to tuple
    if isinstance(out, torch.Tensor):
        residuals = (out.detach(),)
    else:
        residuals = tuple([o.detach() if isinstance(o, torch.Tensor) else o for o in out])

    # Optionally move residuals to CPU pinned memory for lower-GPU usage
    if residual_storage == "cpu_pinned":
        moved = []
        for r in residuals:
            if isinstance(r, torch.Tensor):
                moved.append(r.cpu().pin_memory())
            else:
                moved.append(r)
        residuals = tuple(moved)
        _debug("Moved residuals to CPU pinned memory")

    # Ensure residuals are non-grad
    residuals = tuple([r.detach() if isinstance(r, torch.Tensor) else r for r in residuals])

    _debug("Computed %d residual tensors", len(residuals))
    return residuals
