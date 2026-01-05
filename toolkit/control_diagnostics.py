"""Diagnostics helpers for ControlNet adapters and channel adaptation.

Provides:
- `inspect_adapter(adapter)` prints module conv in_channels and a summary of findings.
- `diagnose_adapter(adapter, latents, control_images)` runs adaptation paths and prints step-by-step logs
  (chosen expected_in, latents adaptation, control_images adaptation).

These helpers use `print_acc` so logs are consistent with runtime traces.
"""
from typing import Any, Optional
import torch
from .print import print_acc
from .control_util import infer_expected_in_ch
from .control_channels import adapt_control_images, adapt_noisy_latents_for_adapter, format_origin


def _collect_conv_in_chs(adapter: Any):
    convs = []
    try:
        for m in adapter.modules():
            w = getattr(m, 'weight', None)
            if w is None:
                continue
            try:
                convs.append(int(w.shape[1]))
            except Exception as e:
                print_acc(f"[DIAG] could not read weight.shape for module {type(m)}: {e}")
    except Exception as e:
        print_acc(f"[DIAG] adapter.modules() not available or failed: {e}")
    return convs


def inspect_adapter(adapter: Any) -> Optional[int]:
    """Diagnostics are disabled in automated training; this helper is inert.

    Use `toolkit.control_diagnostics` interactively for debugging instead.
    """
    return None


def diagnose_adapter(adapter: Any, latents: Optional[torch.Tensor], control_images: Optional[torch.Tensor]):
    """Diagnostics are disabled in automated trainings to avoid noisy logs.

    Use `toolkit.control_diagnostics` interactively for debugging instead.
    """
    # no-op in training environment
    return


if __name__ == '__main__':
    # CLI convenience: allow running diag as a script for a local adapter by importing it
    print_acc("[DIAG] control_diagnostics module loaded as script; use from Python shell to call diagnose_adapter")
