"""The in-graph strip/restore pair must preserve an attached LoRA hijack.

`enable_ingraph_training` strips the streaming machinery off each block Linear so
the compiled trunk sees plain math. `disable_ingraph_training` puts it back. That
round trip runs for real at every sampling boundary once the trunk is up: take it
down, sample, stand it back up.

The subtlety is where the streaming forward actually lives. When a LoRA is
attached, `MemoryManagerModule._capture_base_forward` installs the streaming
forward into the LoRA's `org_forward` slot and leaves `child.forward` as the LoRA
hijack. The strip captured `org_forward` as its "managed forward" but then also
deleted `child.forward` -- dropping the hijack with nothing recording it.

Nothing caught it because `disable_ingraph_training` was only ever called from
inside `enable_ingraph_training`, where the restore list is empty. The first real
disable -> sample -> enable cycle stripped every LoRA off the model, sampled the
bare base, and rebuilt a trunk of pure frozen math whose loss had no grad_fn.
"""

import unittest

import torch
import torch.nn as nn

from extensions_built_in.diffusion_models.krea2.src.mmdit import SingleStreamDiT


class _FakeLayerManager:
    """Stands in for MemoryManagerModule: it owns the base-forward slot it
    hijacked, exactly as _capture_base_forward records it."""

    def __init__(self, original_forward, container, attribute):
        self._original_forward = original_forward
        self._forward_container = container
        self._forward_attribute = attribute


class _FakeLora:
    """A LoRA hijack: child.forward is bound to this object, and the real base
    forward hangs off org_forward -- the slot the memory manager streams into."""

    def __init__(self, child):
        self.lora_down = nn.Linear(4, 2, bias=False)
        self.lora_up = nn.Linear(2, 4, bias=False)
        self.scale = 1.0
        self.org_forward = child.forward
        child.forward = self.forward

    def forward(self, x):
        return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.scale


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _Shim:
    """SingleStreamDiT's strip/restore only touch self.blocks and the two static
    hook helpers, so drive them without building a real transformer."""

    _clear_module_forward_hooks = staticmethod(
        SingleStreamDiT._clear_module_forward_hooks
    )
    _strip_ingraph_compile_contaminants = (
        SingleStreamDiT._strip_ingraph_compile_contaminants
    )
    _restore_ingraph_compile_contaminants = staticmethod(
        SingleStreamDiT._restore_ingraph_compile_contaminants
    )

    def __init__(self, blocks):
        self.blocks = blocks


def _streaming_forward(x):
    raise AssertionError("streaming forward should not run in these tests")


class StripRestorePreservesLoraHijackTests(unittest.TestCase):
    """container is the LoRA, attribute is 'org_forward' -- the real LoRA case."""

    def setUp(self):
        self.block = _Block()
        self.child = self.block.linear
        self.pristine_forward = self.child.forward
        self.lora = _FakeLora(self.child)
        # What _capture_base_forward does: take over the LoRA's org_forward slot.
        self.lora.org_forward = _streaming_forward
        self.child._layer_memory_manager = _FakeLayerManager(
            original_forward=self.pristine_forward,
            container=self.lora,
            attribute="org_forward",
        )
        self.child._memory_management_device = torch.device("cpu")
        self.shim = _Shim([self.block])

    def test_strip_removes_the_hijack_and_the_streaming_slot(self):
        restores = self.shim._strip_ingraph_compile_contaminants([0])
        # The compiled trunk must see plain math: no instance forward, no manager.
        self.assertNotIn("forward", self.child.__dict__)
        self.assertFalse(hasattr(self.child, "_layer_memory_manager"))
        self.assertFalse(hasattr(self.child, "_memory_management_device"))
        self.assertIs(self.lora.org_forward, self.pristine_forward)
        self.assertTrue(restores)

    def test_restore_reinstalls_the_lora_hijack_not_just_org_forward(self):
        restores = self.shim._strip_ingraph_compile_contaminants([0])
        self.shim._restore_ingraph_compile_contaminants(restores)

        # The regression: child.forward came back as the LoRA's bound method,
        # not the pristine nn.Linear forward and not simply absent.
        self.assertIn("forward", self.child.__dict__)
        self.assertIs(self.child.forward.__self__, self.lora)
        self.assertTrue(hasattr(self.child.forward.__self__, "lora_down"))
        # ...and the manager's streaming forward is back in the slot it owns.
        self.assertIs(self.lora.org_forward, _streaming_forward)
        self.assertIs(
            self.child._layer_memory_manager._original_forward, self.pristine_forward
        )
        self.assertEqual(self.child._memory_management_device, torch.device("cpu"))

    def test_strip_restore_is_idempotent_across_repeated_boundaries(self):
        for _ in range(3):
            restores = self.shim._strip_ingraph_compile_contaminants([0])
            self.assertNotIn("forward", self.child.__dict__)
            self.shim._restore_ingraph_compile_contaminants(restores)
            self.assertIs(self.child.forward.__self__, self.lora)
            self.assertIs(self.lora.org_forward, _streaming_forward)


class StripRestoreWithoutLoraTests(unittest.TestCase):
    """container is the child itself -- no LoRA ever attached. The strip must
    still clear the instance forward, and the restore must put back the
    *streaming* forward, never the pristine one."""

    def setUp(self):
        self.block = _Block()
        self.child = self.block.linear
        self.pristine_forward = self.child.forward
        self.child.forward = _streaming_forward
        self.child._layer_memory_manager = _FakeLayerManager(
            original_forward=self.pristine_forward,
            container=self.child,
            attribute="forward",
        )
        self.shim = _Shim([self.block])

    def test_roundtrip_restores_the_streaming_forward(self):
        restores = self.shim._strip_ingraph_compile_contaminants([0])
        self.assertNotIn("forward", self.child.__dict__)

        self.shim._restore_ingraph_compile_contaminants(restores)
        self.assertIs(self.child.__dict__["forward"], _streaming_forward)
        self.assertTrue(hasattr(self.child, "_layer_memory_manager"))


if __name__ == "__main__":
    unittest.main()
