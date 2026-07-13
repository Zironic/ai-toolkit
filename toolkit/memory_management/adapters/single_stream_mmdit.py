"""Architecture facts for a single-stream MMDiT transformer."""

from __future__ import annotations


class SingleStreamMMDiTAdapter:
    """Narrow adapter used by the generic immutable runtime.

    The adapter describes block structure and functional block math only. It
    deliberately owns no residency, transfer, checkpoint, compile, or memory
    policy.
    """

    architecture_key = "single_stream_mmdit"
    _leaf_paths = (
        "attn.wq",
        "attn.wk",
        "attn.wv",
        "attn.gate",
        "attn.wo",
        "mlp.gate",
        "mlp.up",
        "mlp.down",
    )

    def validate_transformer(self, transformer):
        blocks = getattr(transformer, "blocks", None)
        if blocks is None:
            raise TypeError("arena offload requires a transformer.blocks sequence")
        try:
            blocks = tuple(blocks)
        except TypeError as error:
            raise TypeError(
                "arena offload requires a transformer.blocks sequence"
            ) from error
        if not blocks:
            raise ValueError("arena offload requires at least one execution block")
        for index, block in enumerate(blocks):
            try:
                entries = self.leaf_entries(block)
            except (AttributeError, TypeError) as error:
                raise TypeError(
                    "unsupported single-stream MMDiT layout at "
                    f"blocks.{index}"
                ) from error
            for path, child in entries:
                if not hasattr(child, "weight"):
                    raise TypeError(
                        "unsupported single-stream MMDiT leaf at "
                        f"blocks.{index}.{path}"
                    )

    def execution_blocks(self, transformer):
        return tuple(transformer.blocks)

    def can_run_current_call(
        self,
        block_args,
        *,
        ref_kv_capture=None,
        blockcaches=None,
    ) -> bool:
        tvec, _freqs, _mask = block_args
        return not isinstance(tvec, tuple) and ref_kv_capture is None and blockcaches is None

    def block_key(self, transformer, index: int) -> str:
        del transformer
        return f"blocks.{int(index)}"

    def leaf_entries(self, block):
        entries = []
        for path in self._leaf_paths:
            module = block
            for component in path.split("."):
                module = getattr(module, component)
            entries.append((path, module))
        return tuple(entries)

    @staticmethod
    def _forward_owners_on(child):
        owners = []
        seen = set()
        pending = [getattr(child, "__dict__", {}).get("forward")]
        while pending:
            owner = getattr(pending.pop(0), "__self__", None)
            if owner is None or owner is child or id(owner) in seen:
                continue
            seen.add(id(owner))
            owners.append(owner)
            pending.append(getattr(owner, "org_forward", None))
        return owners

    @staticmethod
    def _unsupported_adapter(owner, target_path):
        name = type(owner).__name__ if owner is not None else "unknown forward owner"
        raise RuntimeError(
            "arena offload supports LoRAModule, LokrModule, DoRAModule, and "
            f"linear FullModule adapters; found {name} on {target_path}"
        )

    def collect_adapter_entry(self, child, target_path):
        owners = self._forward_owners_on(child)
        if not owners:
            forward = getattr(child, "__dict__", {}).get("forward")
            if forward is not None:
                self._unsupported_adapter(
                    getattr(forward, "__self__", None), target_path
                )
            return None
        if len(owners) != 1:
            raise RuntimeError(
                "arena offload supports one adapter per canonical Linear; "
                f"found {len(owners)} installed adapters on {target_path}"
            )
        owner = owners[0]
        supported = {"LoRAModule", "LokrModule", "DoRAModule", "FullModule"}
        network_ref = getattr(owner, "network_ref", None)
        network = network_ref() if network_ref is not None else None
        if (
            type(owner).__name__ not in supported
            or not callable(getattr(owner, "functional_forward", None))
            or network is None
            or getattr(network, "is_lorm", False)
        ):
            self._unsupported_adapter(owner, target_path)
        return owner

    def collect_execution_adapters(self, transformer):
        """Collect installed adapters after the training network is attached."""
        adapters = {}
        for index, block in enumerate(self.execution_blocks(transformer)):
            block_adapters = {}
            block_key = self.block_key(transformer, index)
            for name, child in self.leaf_entries(block):
                owner = self.collect_adapter_entry(child, f"{block_key}.{name}")
                if owner is not None:
                    block_adapters[name] = owner
            if block_adapters:
                adapters[index] = block_adapters
        return adapters

    def build_lora_args(self, index: int, loras_by_block, multiplier=None):
        adapters = (loras_by_block or {}).get(int(index))
        if not adapters:
            return None
        args = []
        for leaf_name in self._leaf_paths:
            entry = adapters.get(leaf_name)
            if entry is None or callable(getattr(entry, "functional_forward", None)):
                args.append(entry)
            elif multiplier is None:
                args.append((entry.a, entry.b, entry.scale))
            else:
                args.append((entry.a, entry.b, entry.scale * multiplier))
        return tuple(args)

    def forward_block(
        self,
        block,
        hidden,
        block_args,
        leaf_args,
        fp8_flags,
        lora_args,
        *,
        training: bool,
    ):
        tvec, freqs, mask = block_args
        return block.forward_streamed(
            hidden,
            tvec,
            freqs,
            mask,
            leaf_args,
            fp8_flags,
            training=training,
            loras=lora_args,
        )
