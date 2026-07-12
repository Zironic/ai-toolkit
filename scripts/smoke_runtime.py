"""Shared startup policy for manual smoke scripts."""

from __future__ import annotations

import torch


VRAM_CONTENTION_LIMIT = 0.30
GIB = 1024 ** 3


def add_contention_args(parser) -> None:
    parser.add_argument(
        "--ignore-contention",
        action="store_true",
        help=(
            "run even when more than 30%% of VRAM is already in use at "
            "startup"
        ),
    )


def fail_if_vram_contended(
    device,
    *,
    ignore_contention: bool,
    limit: float = VRAM_CONTENTION_LIMIT,
) -> None:
    """Reject a CUDA smoke when startup VRAM usage exceeds ``limit``."""
    device = torch.device(device)
    if ignore_contention or device.type != "cuda" or not torch.cuda.is_available():
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = int(total_bytes) - int(free_bytes)
    used_fraction = used_bytes / int(total_bytes)
    if used_fraction <= float(limit):
        return

    raise SystemExit(
        "CUDA smoke refused to start: "
        f"{used_fraction * 100.0:.1f}% VRAM is already in use "
        f"({used_bytes / GIB:.2f}/{int(total_bytes) / GIB:.2f} GiB), above "
        f"the {float(limit) * 100.0:.0f}% contention limit. Close other GPU "
        "workloads or pass --ignore-contention to run anyway."
    )
