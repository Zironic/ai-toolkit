"""Derive torch.compile dynamic-shape bounds for the compiled block trunk.

The immutable runtime compiles one kernel per block and marks the hidden-state
sequence dim dynamic (`compile_dynamic_hints`). Without bounds, every distinct
resolution bucket / prompt length is a fresh specialization and a recompile.

The bounds are computable ahead of the first compiled call from three inputs --
no eager probe:

  observed dataset shapes (latent H/W, text length)
  + the loaded transformer config (patch factor)
  + the compiled-block ABI (`SequenceLayout`: alignment, text packing)
  = precompile bounds

`SequenceLayout` describes the toolkit's compiled-block ABI, not the weights, so
it comes from the model integration rather than the HF config JSON.

Bounds are taken as *independent extremes* over the observed fields, never as
min/max of per-sample totals. Text features are padded to the batch max while
images are bucketed, so a batch can combine the largest image with the longest
prompt in its bucket even when no single sample did. Pairing the extremes of
each field is the only bound that covers every batch the sampler can build.
Overshooting the range is free; undershooting is a hard constraint violation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObservedInputShape:
    """One (latent, text) shape combination the run can actually produce."""

    latent_height: int
    latent_width: int
    text_length: int = 0


@dataclass(frozen=True)
class SequenceLayout:
    """How the model packs latents + text into the compiled trunk's sequence."""

    sequence_alignment: int = 1
    includes_text: bool = True
    extra_tokens: int = 0


@dataclass(frozen=True)
class SequenceBounds:
    minimum: int
    maximum: int


def _config_value(config: Any, *names: str, default=None):
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _as_pair(value: Any) -> tuple[int, int] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return int(value[-2]), int(value[-1])
        except (TypeError, ValueError):
            return None
    return None


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return int(value)
    return ((int(value) + alignment - 1) // alignment) * alignment


def estimate_hidden_sequence_bounds(
    *,
    transformer: Any,
    observed_shapes: Iterable[ObservedInputShape],
    layout: SequenceLayout,
) -> SequenceBounds | None:
    """Bounds for the trunk's sequence dim, or None if they can't be derived.

    None means "do not constrain": the caller marks the dim dynamic without a
    range (or leaves it alone) rather than risking a wrong bound.
    """
    config = getattr(transformer, "config", None)
    if config is None:
        return None

    patch = _as_pair(
        _config_value(config, "patch_size", "patch", "latent_patch_size")
    )
    if patch is None:
        return None
    patch_h, patch_w = patch
    if patch_h < 1 or patch_w < 1:
        return None

    image_tokens: list[int] = []
    text_lengths: list[int] = []

    for shape in observed_shapes:
        if (
            shape.latent_height % patch_h != 0
            or shape.latent_width % patch_w != 0
        ):
            # A shape the patchifier cannot produce: our model of the sequence
            # is wrong, so any bound we derive would be wrong too.
            return None
        image_tokens.append(
            (shape.latent_height // patch_h) * (shape.latent_width // patch_w)
        )
        text_lengths.append(int(shape.text_length))

    if not image_tokens:
        return None

    text_low = min(text_lengths) if layout.includes_text else 0
    text_high = max(text_lengths) if layout.includes_text else 0
    extra = int(layout.extra_tokens)

    low = min(image_tokens) + text_low + extra
    high = max(image_tokens) + text_high + extra

    minimum = align_up(low, layout.sequence_alignment)
    maximum = align_up(high, layout.sequence_alignment)
    if minimum < 1 or maximum < minimum:
        return None
    return SequenceBounds(minimum=minimum, maximum=maximum)
