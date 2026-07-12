"""Tiled transpose for 1-byte (fp8) weights.

Ada's ``_scaled_mm`` wants its B operand column-major. Our arena stores each
rowwise-fp8 weight as a contiguous (out, in) block, which gives the *forward*
that layout for free (``qdata.t()`` is already column-major) but leaves the
backward's grad-input GEMM needing the opposite orientation -- so it has to
materialize a transposed copy of the weight, once per streamed Linear per step.

``qdata.t().contiguous()`` lowers to an elementwise clone. For a 1-byte dtype
that means uncoalesced single-byte writes: measured at 46-75 GB/s on an RTX 4070
(~450 GB/s of bandwidth), i.e. ~266 ms/step across Krea2's 448 streamed Linears.
A 64x64 tiled copy reads and writes each tile coalesced and measures 219-347
GB/s -- 4-7.5x faster on the real weight shapes.

Exposed as a custom op so it stays a single opaque, functional node inside the
compiled backward instead of being re-lowered to the same slow clone.
"""

from __future__ import annotations

import torch

try:  # triton ships with torch's CUDA builds; CPU-only envs may not have it.
    import triton
    import triton.language as tl

    @triton.jit
    def _fp8_transpose_kernel(
        src_ptr,
        dst_ptr,
        M,
        N,
        stride_sm,
        stride_sn,
        stride_dm,
        stride_dn,
        BLOCK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK + tl.arange(0, BLOCK)
        rn = pid_n * BLOCK + tl.arange(0, BLOCK)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        # Read a [BLOCK, BLOCK] tile of src (M, N) coalesced along N ...
        tile = tl.load(
            src_ptr + rm[:, None] * stride_sm + rn[None, :] * stride_sn,
            mask=mask,
            other=0,
        )
        # ... and write it into dst (N, M) coalesced along M.
        tl.store(
            dst_ptr + rn[:, None] * stride_dm + rm[None, :] * stride_dn,
            tl.trans(tile),
            mask=tl.trans(mask),
        )

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only on triton-less installs
    _HAVE_TRITON = False


BLOCK = 64


def _tiled_supported(x: torch.Tensor) -> bool:
    return (
        _HAVE_TRITON
        and x.device.type == "cuda"
        and x.ndim == 2
        and x.element_size() == 1
        and x.numel() > 0
    )


@torch.library.custom_op("mm::transpose_contiguous_1byte", mutates_args=())
def transpose_contiguous_1byte(x: torch.Tensor) -> torch.Tensor:
    """Contiguous (N, M) copy of a 2-D (M, N) 1-byte tensor.

    Equivalent to ``x.t().contiguous()``; ``.t()`` the result to get x's own
    shape back as a column-major operand.
    """
    if not _tiled_supported(x):
        return x.t().contiguous()

    src = x.view(torch.uint8)
    m, n = src.shape
    dst = torch.empty((n, m), device=src.device, dtype=torch.uint8)
    grid = (triton.cdiv(m, BLOCK), triton.cdiv(n, BLOCK))
    _fp8_transpose_kernel[grid](
        src,
        dst,
        m,
        n,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        BLOCK=BLOCK,
    )
    return dst.view(x.dtype)


@transpose_contiguous_1byte.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (x.shape[1], x.shape[0]), dtype=x.dtype, device=x.device
    )


def column_major(x: torch.Tensor) -> torch.Tensor:
    """``x`` (M, N) as a column-major (M, N) operand for _scaled_mm."""
    return torch.ops.mm.transpose_contiguous_1byte(x).t()
