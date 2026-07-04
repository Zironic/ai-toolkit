import types

import pytest
import torch

from toolkit.memory_management.ingraph_stream import (
    CompileRegionError,
    assert_compile_region_clean,
    compile_region_reasons,
)


def test_compile_region_audit_accepts_clean_module():
    module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

    assert compile_region_reasons(module) == []
    assert_compile_region_clean(module)


def test_compile_region_audit_rejects_layer_memory_manager_marker():
    module = torch.nn.Linear(4, 4)
    module._layer_memory_manager = object()

    assert compile_region_reasons(module) == ["legacy_layer_manager_present"]
    with pytest.raises(CompileRegionError) as error:
        assert_compile_region_clean(module)
    assert error.value.reasons == ("legacy_layer_manager_present",)


def test_compile_region_audit_rejects_hooks():
    module = torch.nn.Linear(4, 4)
    module.register_forward_pre_hook(lambda _module, args: args)

    assert compile_region_reasons(module) == ["hook_present"]
    with pytest.raises(CompileRegionError) as error:
        assert_compile_region_clean(module)
    assert error.value.reasons == ("hook_present",)


def test_compile_region_audit_rejects_instance_forward_swap():
    module = torch.nn.Linear(4, 4)
    original = module.forward

    def swapped(self, x):
        return original(x)

    module.forward = types.MethodType(swapped, module)

    assert compile_region_reasons(module) == ["forward_hijack_present"]
    with pytest.raises(CompileRegionError) as error:
        assert_compile_region_clean(module)
    assert error.value.reasons == ("forward_hijack_present",)

