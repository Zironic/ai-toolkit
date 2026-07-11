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

    def build_lora_args(self, index: int, loras_by_block, multiplier=None):
        loras = (loras_by_block or {}).get(int(index))
        if not loras:
            return None

        args = []
        for leaf_name in self._leaf_paths:
            entry = loras.get(leaf_name)
            if entry is None:
                args.append(None)
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
