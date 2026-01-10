"""Utilities to inspect LoRA/adapter files and detect control-specific weights or fused residuals.

Usage (CLI):
    python -m toolkit.lora_inspect path/to/lora.safetensors

Functions expose programmatic API:
    inspect_lora_state_dict(state_dict) -> Dict[str, Any]

Heuristics:
- Keys matching control-related patterns (control, controlnet, zimage, down_block, mid_block, residual)
- Keys modifying input embedders (x_embedder, img_embedder, patch_embedding, in_out)
- LoRA-style keys (lora_A, lora_B, .lora_)

This is intentionally conservative and reports matches for manual review.
"""

from __future__ import annotations

import os
import re
import json
from typing import Dict, Any, Iterable, List, Tuple

# Torch and safetensors imports are lazy to avoid heavy import side-effects during test collection
_load_extra_deps = None

def _ensure_extra_deps():
    global _load_extra_deps
    if _load_extra_deps is None:
        try:
            import torch  # type: ignore
            from safetensors.torch import load_file as load_safetensors  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to import torch/safetensors: {e}") from e
        _load_extra_deps = (torch, load_safetensors)
    return _load_extra_deps

# Patterns to detect control-specific or fused residuals
CONTROL_PATTERNS = [
    r"\bcontrol\b",
    r"controlnet",
    r"\bctrl\b",
    r"\bcontrol_",
    r"z[-_]?image",
    r"video_x",
]

RESIDUAL_PATTERNS = [
    r"residual",
    r"additional_residual",
    r"down_block",
    r"mid_block",
    r"down_intrablock",
]

INPUT_EMBEDDER_PATTERNS = [
    r"x_embedder",
    r"img_embedder",
    r"patch_embedding",
    r"in_out",
    r"input_embedder",
    r"frame_embedder",
    r"img_embed",
]

LORA_PATTERNS = [
    r"\.lora_",
    r"lora_A",
    r"lora_B",
    r"lora_down",
    r"lora_up",
]

COMPILED_CONTROL = [re.compile(p, re.IGNORECASE) for p in CONTROL_PATTERNS]
COMPILED_RESIDUAL = [re.compile(p, re.IGNORECASE) for p in RESIDUAL_PATTERNS]
COMPILED_INPUT = [re.compile(p, re.IGNORECASE) for p in INPUT_EMBEDDER_PATTERNS]
COMPILED_LORA = [re.compile(p, re.IGNORECASE) for p in LORA_PATTERNS]


def _load_state_dict(path: str) -> Dict[str, Any]:
    """Load state dict from safetensors or torch checkpoint in a conservative, CPU-safe way."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    torch, load_safetensors = _ensure_extra_deps()

    if path.endswith(".safetensors") or path.endswith(".safetensor"):
        raw = load_safetensors(path)
        # safetensors returns numpy arrays; convert to torch tensors lazily
        return {k: torch.as_tensor(v) for k, v in raw.items()}

    # fallback to torch.load for .pt/.pth/.bin/.ckpt
    raw = torch.load(path, map_location="cpu")
    # Some checkpoints wrap dicts under 'state_dict' or 'model'
    if isinstance(raw, dict) and (
        "state_dict" in raw or "model" in raw or "model_state_dict" in raw
    ):
        # try common keys
        for key in ("state_dict", "model", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    if isinstance(raw, dict):
        return raw
    raise ValueError("Unsupported checkpoint format or unexpected object type")


def inspect_lora_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect keys in a LoRA-style state_dict and return a concise report.

    Report includes:
      - total_keys
      - lora_keys_count
      - control_like_keys (list)
      - residual_like_keys (list)
      - input_embedder_like_keys (list)
      - sample_keys (first 20 keys)

    Heuristics are conservative and intended for manual verification.
    """
    keys = list(state_dict.keys())
    total = len(keys)

    lora_keys: List[str] = []
    control_like: List[str] = []
    residual_like: List[str] = []
    input_embedder_like: List[str] = []

    for k in keys:
        low = k.lower()
        # LoRA indicators
        if any(p.search(k) for p in COMPILED_LORA) or "lora" in low:
            lora_keys.append(k)
        # Control patterns
        if any(p.search(k) for p in COMPILED_CONTROL):
            control_like.append(k)
        # Residual-like
        if any(p.search(k) for p in COMPILED_RESIDUAL):
            residual_like.append(k)
        # Input embedder like
        if any(p.search(k) for p in COMPILED_INPUT):
            input_embedder_like.append(k)

    report = {
        "total_keys": total,
        "lora_keys_count": len(lora_keys),
        "lora_keys_sample": lora_keys[:20],
        "control_like_count": len(control_like),
        "control_like_sample": control_like[:20],
        "residual_like_count": len(residual_like),
        "residual_like_sample": residual_like[:20],
        "input_embedder_like_count": len(input_embedder_like),
        "input_embedder_like_sample": input_embedder_like[:20],
        "sample_keys": keys[:30],
    }
    return report


def inspect_lora_file(path: str) -> Dict[str, Any]:
    sd = _load_state_dict(path)
    return inspect_lora_state_dict(sd)


def dump_report(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect LoRA state dict for control-specific weights or fused residuals")
    parser.add_argument("file", help="Path to .safetensors or .pt/.pth LoRA file")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    report = inspect_lora_file(args.file)
    if args.json:
        print(dump_report(report))
    else:
        print("LoRA Inspection Report:\n")
        print(f"Total keys: {report['total_keys']}")
        print(f"LoRA-like keys: {report['lora_keys_count']}")
        print(f"Control-like keys: {report['control_like_count']}")
        print(f"Residual-like keys: {report['residual_like_count']}")
        print(f"Input-embedder-like keys: {report['input_embedder_like_count']}")
        print("\nExamples of control-like keys:")
        for k in report["control_like_sample"]:
            print("  ", k)
        if len(report["control_like_sample"]) == 0:
            print("  (none detected)")
        print("\nExamples of residual-like keys:")
        for k in report["residual_like_sample"]:
            print("  ", k)
        if len(report["residual_like_sample"]) == 0:
            print("  (none detected)")

