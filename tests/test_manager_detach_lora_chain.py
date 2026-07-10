"""The manager's detach must unwind ITS OWN forward from the live chain.

The boundary crash this pins down (lora_hijack_missing, collected 84/224):
the manager attaches BEFORE the LoRA network is applied, so it records
``(module, 'forward')`` as the slot it hijacked. The LoRA then hijacks
``module.forward`` on top, chaining to the streaming forward via its
``org_forward``. The old detach wrote the original forward straight into the
recorded slot -- overwriting the LoRA hijack. Every sampling boundary detach
then silently stripped the LoRA off every streamed leaf; the in-graph
training re-enable's fail-loud count guard is what finally caught it.

``_uninstall_base_forward`` walks the live chain instead and removes only the
manager's own forward, wherever it lives now.
"""

import unittest

import torch.nn as nn

from toolkit.memory_management.manager_modules import BaseLayerMemoryManager


class _FakeLora:
    def __init__(self, child):
        self.org_forward = child.forward
        child.forward = self.forward

    def forward(self, x):
        return self.org_forward(x)


def _streaming_forward(x):
    raise AssertionError("streaming forward should not run in these tests")


def _same_bound(a, b):
    """Bound methods are minted fresh per attribute access, so compare the
    underlying function and receiver instead of object identity."""
    return (
        getattr(a, "__func__", a) is getattr(b, "__func__", b)
        and getattr(a, "__self__", None) is getattr(b, "__self__", None)
    )


def _make_manager(child):
    lmm = BaseLayerMemoryManager(child, manager=object())
    lmm._original_forward = lmm._capture_base_forward()
    lmm._install_base_forward(_streaming_forward)
    child._layer_memory_manager = lmm
    return lmm


class UninstallBaseForwardTests(unittest.TestCase):
    def test_attach_then_lora_detach_preserves_hijack(self):
        # The regression sequence: attach first (records module.forward),
        # LoRA applied on top afterwards.
        child = nn.Linear(4, 4)
        pristine = child.forward
        lmm = _make_manager(child)
        lora = _FakeLora(child)  # hijack over the streaming forward

        self.assertTrue(lmm._uninstall_base_forward())

        # The LoRA hijack survives on the module...
        self.assertIs(child.forward.__self__, lora)
        # ...and its org_forward chains to the pristine base again, with the
        # streaming forward fully unwound.
        self.assertTrue(_same_bound(lora.org_forward, pristine))

    def test_lora_then_attach_detach_restores_org_slot(self):
        # The other arrangement: LoRA applied first, manager attaches after
        # and records the LoRA's org_forward slot.
        child = nn.Linear(4, 4)
        pristine = child.forward
        lora = _FakeLora(child)
        lmm = _make_manager(child)
        self.assertIs(lmm._forward_container, lora)

        self.assertTrue(lmm._uninstall_base_forward())

        self.assertIs(child.forward.__self__, lora)
        self.assertTrue(_same_bound(lora.org_forward, pristine))

    def test_no_lora_detach_restores_module_forward(self):
        child = nn.Linear(4, 4)
        pristine = child.forward
        lmm = _make_manager(child)

        self.assertTrue(lmm._uninstall_base_forward())

        self.assertTrue(_same_bound(child.__dict__["forward"], pristine))

    def test_fallback_when_not_in_chain_writes_recorded_slot(self):
        # If something already unwound us (or the chain was rebuilt), fall
        # back to the attach-time slot -- the pre-fix behavior.
        child = nn.Linear(4, 4)
        pristine = child.forward
        lmm = _make_manager(child)
        del child.__dict__["forward"]  # someone stripped the instance forward

        self.assertTrue(lmm._uninstall_base_forward())
        self.assertTrue(_same_bound(child.__dict__["forward"], pristine))


if __name__ == "__main__":
    unittest.main()
