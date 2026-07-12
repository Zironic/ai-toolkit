"""compiler_stance_supported must actually exercise the stance, not just call
set_stance() -- that function accepts any string silently and only raises on
the first real compiled dispatch (ticket: aot_eager_then_compile gating).
"""

from toolkit.compile_cache import compiler_stance_supported


def test_default_stance_supported():
    assert compiler_stance_supported("default") is True


def test_unknown_stance_rejected():
    assert compiler_stance_supported("totally_bogus_stance_xyz") is False


def test_probe_restores_default_stance():
    import torch

    compiler_stance_supported("totally_bogus_stance_xyz")
    # A failed probe must not leave a stale stance behind for later compiles.
    assert torch._dynamo.eval_frame._stance.stance == "default"
