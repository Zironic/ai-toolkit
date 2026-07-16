"""Sequence-bound estimation for the compiled block trunk (CPU only)."""

from types import SimpleNamespace

import pytest

from toolkit.compile_shape_bounds import (
    ObservedInputShape,
    SequenceBounds,
    SequenceLayout,
    align_up,
    estimate_hidden_sequence_bounds,
    estimate_hidden_sequence_variants,
)

KREA_LAYOUT = SequenceLayout(
    sequence_alignment=256, includes_text=True, extra_tokens=0
)


def _transformer(patch=2):
    return SimpleNamespace(config=SimpleNamespace(patch=patch))


def _bounds(shapes, layout=KREA_LAYOUT, transformer=None):
    return estimate_hidden_sequence_bounds(
        transformer=transformer or _transformer(),
        observed_shapes=shapes,
        layout=layout,
    )


def test_align_up():
    assert align_up(4096, 256) == 4096
    assert align_up(4097, 256) == 4352
    assert align_up(10, 1) == 10


def test_krea_1024_square_matches_runtime_padding():
    # 1024px -> 128x128 latent -> 64x64 patches -> 4096 image tokens, + 512 text
    # -> already 256-aligned.
    bounds = _bounds([ObservedInputShape(128, 128, 512)])
    assert bounds == SequenceBounds(4608, 4608)


def test_bounds_pair_field_extremes_not_per_sample_totals():
    # Text features pad to the batch max while images are bucketed, so the
    # largest image can share a batch with the longest prompt even though no
    # single sample did. The max bound must cover that pairing.
    shapes = [
        ObservedInputShape(128, 128, 8),  # big image, short prompt
        ObservedInputShape(64, 64, 512),  # small image, long prompt
    ]
    bounds = _bounds(shapes)
    # per-sample totals would top out at 4096 + 8 -> 4352.
    assert bounds.maximum == align_up(4096 + 512, 256) == 4608
    assert bounds.minimum == align_up(1024 + 8, 256) == 1024 + 256


def test_text_ignored_when_layout_excludes_it():
    layout = SequenceLayout(sequence_alignment=256, includes_text=False)
    bounds = _bounds([ObservedInputShape(128, 128, 512)], layout=layout)
    assert bounds == SequenceBounds(4096, 4096)


def test_extra_tokens_widen_both_ends():
    layout = SequenceLayout(sequence_alignment=1, includes_text=False, extra_tokens=3)
    bounds = _bounds([ObservedInputShape(64, 64, 0)], layout=layout)
    assert bounds == SequenceBounds(1024 + 3, 1024 + 3)


@pytest.mark.parametrize(
    "transformer",
    [
        SimpleNamespace(),  # no config
        SimpleNamespace(config=SimpleNamespace(channels=16)),  # no patch factor
    ],
)
def test_no_bounds_without_patch_factor(transformer):
    assert _bounds([ObservedInputShape(128, 128, 512)], transformer=transformer) is None


def test_no_bounds_when_latent_is_not_patchable():
    # An odd latent side cannot be produced by a patch-2 patchifier: our model of
    # the sequence is wrong, so refuse rather than emit a wrong bound.
    assert _bounds([ObservedInputShape(127, 128, 512)]) is None


def test_no_bounds_without_shapes():
    assert _bounds([]) is None


def test_patch_size_pair_config():
    transformer = SimpleNamespace(config={"patch_size": [1, 2, 2]})
    bounds = _bounds([ObservedInputShape(128, 128, 0)], transformer=transformer)
    assert bounds == SequenceBounds(4096, 4096)


def test_exact_variants_deduplicate_equal_sequence_sizes():
    variants = estimate_hidden_sequence_variants(
        transformer=_transformer(),
        observed_shapes=[
            ObservedInputShape(128, 64, 512),
            ObservedInputShape(64, 128, 512),
            ObservedInputShape(64, 64, 8),
        ],
        layout=KREA_LAYOUT,
    )
    assert variants == frozenset({2560, 1280})


def test_exact_variants_refuse_unknown_layout_inputs():
    assert (
        estimate_hidden_sequence_variants(
            transformer=SimpleNamespace(),
            observed_shapes=[ObservedInputShape(64, 64, 8)],
            layout=KREA_LAYOUT,
        )
        is None
    )
