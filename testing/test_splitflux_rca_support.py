import types
from toolkit.splitflux import rca_supported_arch


def make_mc(arch):
    mc = types.SimpleNamespace()
    mc.arch = arch
    return mc


def test_rca_supported_for_flux1():
    mc = make_mc('flux')
    assert rca_supported_arch(mc) is True


def test_rca_disabled_for_flux2():
    mc = make_mc('flux2')
    assert rca_supported_arch(mc) is False


def test_rca_disabled_for_zimage():
    mc = make_mc('zimage')
    assert rca_supported_arch(mc) is False


def test_rca_disabled_for_none():
    mc = types.SimpleNamespace()
    assert rca_supported_arch(mc) is False
