"""Compatibility import; FP8 execution lives in toolkit.quantization."""

from toolkit.quantization.fp8_transpose import (  # noqa: F401
    column_major,
    transpose_contiguous_1byte,
)
