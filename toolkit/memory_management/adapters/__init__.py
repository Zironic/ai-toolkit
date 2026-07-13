"""Minimal architecture adapters for generic memory-management runtimes."""

from toolkit.memory_management.adapters.single_stream_mmdit import (
    SingleStreamMMDiTAdapter,
)
from toolkit.memory_management.adapters.protocol import (
    ArchitectureAdapter,
    validate_architecture_adapter,
)

__all__ = [
    "ArchitectureAdapter",
    "SingleStreamMMDiTAdapter",
    "validate_architecture_adapter",
]
