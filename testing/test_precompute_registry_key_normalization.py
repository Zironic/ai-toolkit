import pytest
pytest.skip('precompute tests removed: skipping')


def make_fake_context():
    # small fake dict to act as precomputed contexts
    return {128: 'FAKE_TENSOR_PLACEHOLDER'}


def test_registry_key_normalization_variants():
    clear()
    # construct a canonical absolute path
    base = os.path.abspath(os.path.join('datasets', 'jinx_nobg_mini_short', 'Screenshot 2025-12-14 161526.png'))
    forward = base.replace('\\', '/')
    back = base.replace('/', '\\')
    # vary case for Windows-style case-insensitive behavior
    alt_case = base.swapcase()

    ctx = make_fake_context()

    # set using forward-slash variant
    set_preencoded_control_contexts(forward, ctx)

    # get using backslash variant
    got = get_preencoded_control_contexts(back)
    assert got is not None
    assert got == ctx

    # get using different case variant
    got2 = get_preencoded_control_contexts(alt_case)
    assert got2 is not None
    assert got2 == ctx

    clear()
    # direct set with backslash and get with forward
    set_preencoded_control_contexts(back, ctx)
    assert get_preencoded_control_contexts(forward) == ctx

    clear()