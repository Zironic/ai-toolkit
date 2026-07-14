"""Architecture facts for a single-stream MMDiT transformer."""

from __future__ import annotations

from toolkit.quantization.fp8_linear import bind_storage_operation


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
    def _unsupported_adapter(owner, target_path):
        name = type(owner).__name__ if owner is not None else "unknown adapter"
        raise RuntimeError(
            "arena offload supports LoRAModule, LokrModule, DoRAModule, and "
            f"linear FullModule adapters; found {name} on {target_path}"
        )

    @staticmethod
    def _target_module(owner):
        target_ref = getattr(owner, "orig_module_ref", None)
        if callable(target_ref):
            return target_ref()
        targets = getattr(owner, "org_module", None)
        if isinstance(targets, (tuple, list)) and targets:
            return targets[0]
        return None

    def _validate_adapter_entry(self, owner, target_path, network):
        supported = {"LoRAModule", "LokrModule", "DoRAModule", "FullModule"}
        network_ref = getattr(owner, "network_ref", None)
        owner_network = network_ref() if network_ref is not None else None
        if (
            type(owner).__name__ not in supported
            or not callable(getattr(owner, "functional_forward", None))
            or owner_network is not network
            or getattr(owner_network, "is_lorm", False)
        ):
            self._unsupported_adapter(owner, target_path)
        return owner

    def collect_execution_adapters(self, transformer, network):
        """Collect adapters from the explicit trainer network contract."""
        if network is None:
            return {}
        provider = getattr(network, "arena_execution_adapters", None)
        candidates = (
            provider()
            if callable(provider)
            else getattr(network, "unet_loras", None)
        )
        if candidates is None:
            raise RuntimeError(
                "arena offload requires network.unet_loras or "
                "network.arena_execution_adapters()"
            )

        targets = {}
        target_paths = {}
        for index, block in enumerate(self.execution_blocks(transformer)):
            block_key = self.block_key(transformer, index)
            for name, child in self.leaf_entries(block):
                key = id(child)
                targets[key] = (index, name)
                target_paths[key] = f"{block_key}.{name}"

        adapters = {}
        for owner in candidates:
            target = self._target_module(owner)
            location = targets.get(id(target))
            if location is None:
                continue
            index, name = location
            block_adapters = adapters.setdefault(index, {})
            if name in block_adapters:
                raise RuntimeError(
                    "arena offload supports one adapter per canonical Linear; "
                    f"found multiple installed adapters on {target_paths[id(target)]}"
                )
            block_adapters[name] = self._validate_adapter_entry(
                owner,
                target_paths[id(target)],
                network,
            )
        return adapters

    def build_adapter_args(self, index: int, adapters_by_block, multiplier=None):
        adapters = (adapters_by_block or {}).get(int(index))
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

    def bind_block_operations(self, storage_views, device):
        return tuple(
            bind_storage_operation(
                view.tensors,
                execution_key=view.spec.execution_key,
                weight_leaf_count=view.spec.weight_leaf_count,
                device=device,
            )
            for view in storage_views
        )

    def forward_block(
        self,
        block,
        hidden,
        block_args,
        leaf_args,
        linear_operations,
        adapter_args,
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
            linear_operations,
            training=training,
            loras=adapter_args,
        )
