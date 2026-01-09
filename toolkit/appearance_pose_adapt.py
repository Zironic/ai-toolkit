"""Appearance-Pose Adapter (APPA) - residuals-mode implementation.

This module implements a minimal, testable APPA that transforms ControlNet per-block
residuals (down-block residuals and mid-block residual) in-place while preserving
shapes and dtypes. The transformation is intentionally small (1x1 conv per block)
so it's cheap and easy to test and debug.

API:
- class AppearancePoseAdapter(nn.Module)
    - transform_residuals(down_block_res_samples: List[Tensor], mid_block_res_sample: Optional[Tensor], timesteps: Optional[Tensor]=None, batch_images: Optional[Tensor]=None) -> Tuple[List[Tensor], Optional[Tensor]]

Behavior:
- Preserves shapes and dtypes.
- Parameters are trainable and participate in autograd.
- Works on CPU; offload helpers in toolkit.controlnet_offload can be used to move module around.

"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class _IdentityResidual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AppearancePoseAdapter(nn.Module):
    """Minimal residuals-mode APPA.

    For each element in `down_block_res_samples` (a list of tensors with shape [B, C_i, H_i, W_i]),
    we create a small 1x1 conv that maps C_i -> C_i. The mid-block residual (if present) is handled
    similarly. The module preserves shapes and dtypes.
    """

    def __init__(self, channel_list: Optional[List[int]] = None, mid_channels: Optional[int] = None):
        """Create an APPA instance.

        Args:
            channel_list: optional list of channel counts for each down block. If provided,
                one conv per channel count is created. If not provided, convs will be created
                lazily on first call based on input tensors.
            mid_channels: optional mid-block channels size (create mid conv lazily if None).
        """
        super().__init__()
        self._built = False
        self.channel_list = channel_list
        self.mid_channels = mid_channels
        # containers for per-level convs
        self.down_convs = nn.ModuleList()
        self.mid_conv = None if mid_channels is None else nn.Conv2d(mid_channels, mid_channels, kernel_size=1)

        # A tiny MLP for optional timestep conditioning (unused in tests but present for completeness)
        self.timestep_mlp = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))

    def _lazy_build(self, down_block_res_samples: List[torch.Tensor], mid_block_res_sample: Optional[torch.Tensor] = None):
        # Build per-level convs based on inputs
        if self.channel_list is not None and len(self.channel_list) >= len(down_block_res_samples):
            chs = self.channel_list
        else:
            chs = [int(x.shape[1]) for x in down_block_res_samples]
            self.channel_list = chs

        # ensure down_convs has same length
        for i, c in enumerate(chs):
            if i >= len(self.down_convs):
                conv = nn.Conv2d(c, c, kernel_size=1)
                # initialize bias to zero for smaller initial change
                nn.init.zeros_(conv.bias)
                nn.init.xavier_uniform_(conv.weight)
                self.down_convs.append(conv)

        if mid_block_res_sample is not None and self.mid_conv is None:
            c = int(mid_block_res_sample.shape[1])
            self.mid_conv = nn.Conv2d(c, c, kernel_size=1)
            nn.init.zeros_(self.mid_conv.bias)
            nn.init.xavier_uniform_(self.mid_conv.weight)

        self._built = True

    def transform_residuals(
        self,
        down_block_res_samples: List[torch.Tensor],
        mid_block_res_sample: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        batch_images: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Transform residuals and return new tensors with same shapes and dtypes.

        This method is intentionally simple and fully differentiable.
        """
        if not isinstance(down_block_res_samples, (list, tuple)):
            raise ValueError("down_block_res_samples must be a list or tuple of tensors")

        if not self._built:
            self._lazy_build(down_block_res_samples, mid_block_res_sample)

        out_down = []
        for i, src in enumerate(down_block_res_samples):
            if src is None:
                out_down.append(None)
                continue
            conv = self.down_convs[i] if i < len(self.down_convs) else _IdentityResidual()
            # ensure dtype and device are preserved
            dev = src.device
            dt = src.dtype
            src_cast = src.to(device=dev, dtype=dt)
            out = conv(src_cast)
            # ensure same shape
            if out.shape != src.shape:
                # if channels equal but conv returned different shape due to accidental behavior,
                # resize channels (should not happen with 1x1 conv), but be safe
                out = out.view(src.shape)
            # cast back to original dtype (conv shouldn't change dtype though)
            out = out.to(dtype=dt, device=dev)
            out_down.append(out)

        if mid_block_res_sample is None:
            out_mid = None
        else:
            conv = self.mid_conv if self.mid_conv is not None else _IdentityResidual()
            src = mid_block_res_sample
            dev = src.device
            dt = src.dtype
            out = conv(src.to(device=dev, dtype=dt))
            out = out.to(dtype=dt, device=dev)
            out_mid = out

        return out_down, out_mid


# small convenience alias for tests and imports
def make_residuals_adapter(channel_list: Optional[List[int]] = None, mid_channels: Optional[int] = None) -> AppearancePoseAdapter:
    return AppearancePoseAdapter(channel_list=channel_list, mid_channels=mid_channels)
