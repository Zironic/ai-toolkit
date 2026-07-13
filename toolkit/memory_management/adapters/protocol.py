"""Architecture contract for block-arena execution."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ArchitectureAdapter(Protocol):
    """Model-specific facts required by the model-neutral arena core.

    Model integrations retain their complete forward: input construction,
    shared block arguments, and final projection. The runtime receives only a
    hidden state plus an opaque block-argument pytree and delegates its meaning
    to this adapter.
    """

    architecture_key: str

    def validate_transformer(self, transformer: Any) -> None: ...

    def execution_blocks(self, transformer: Any) -> tuple: ...

    def block_key(self, transformer: Any, index: int) -> str: ...

    def leaf_entries(self, block: Any) -> tuple: ...

    def collect_execution_adapters(
        self,
        transformer: Any,
        network: Any | None,
    ) -> dict: ...

    def build_adapter_args(
        self,
        index: int,
        adapters_by_block: dict,
        multiplier: Any | None = None,
    ) -> Any: ...

    def can_run_current_call(self, block_args: Any, **kwargs: Any) -> bool: ...

    def bind_block_operations(self, block: Any, device: Any) -> tuple: ...

    def forward_block(
        self,
        block: Any,
        hidden: Any,
        block_args: Any,
        leaf_args: Any,
        linear_operations: Any,
        adapter_args: Any,
        *,
        training: bool,
    ) -> Any: ...


def validate_architecture_adapter(adapter: Any) -> None:
    """Reject incomplete adapters before canonical model mutation."""
    if not isinstance(getattr(adapter, "architecture_key", None), str):
        raise TypeError("arena architecture adapter requires architecture_key")
    required = (
        "validate_transformer",
        "execution_blocks",
        "block_key",
        "leaf_entries",
        "collect_execution_adapters",
        "build_adapter_args",
        "can_run_current_call",
        "bind_block_operations",
        "forward_block",
    )
    missing = tuple(
        name for name in required if not callable(getattr(adapter, name, None))
    )
    if missing:
        raise TypeError(
            "incomplete arena architecture adapter: missing " + ", ".join(missing)
        )
