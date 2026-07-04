from pathlib import Path
path = Path(r'tests/test_pin_budget_caps.py')
text = path.read_text(encoding='utf-8')
text = text.replace('''_ENV = {
    "AI_TOOLKIT_PINNED_WEIGHT_RAM_FLOOR_GIB": "8.0",
    "AI_TOOLKIT_PINNED_WEIGHT_AVAILABLE_FLOOR_GIB": "6.0",
    "AI_TOOLKIT_PINNED_WEIGHT_WDDM_FRACTION": "0.25",
}


class CapAutoPinBudgetTests(unittest.TestCase):
    """The auto pinned-weight budget must be capped by host headroom: pinning
    consumes available RAM ~1:1 and commits against the WDDM shared GPU budget.
    Sizing it by total RAM alone exhausted the shared budget at attach and made
    the next CUDA call fail with a raw cudaErrorMemoryAllocation."""
''', '''_ENV = {
    "AI_TOOLKIT_PINNED_WEIGHT_RAM_FLOOR_GIB": "8.0",
    "AI_TOOLKIT_PINNED_WEIGHT_WDDM_FRACTION": "0.25",
}


class CapAutoPinBudgetTests(unittest.TestCase):
    """The auto pinned-weight budget is capped by the WDDM shared pinned-memory
    proxy, not psutil's reported available pageable RAM. Permanent weight pinning
    page-locks already-loaded CPU weights; the crash mode is shared pinned-memory
    commit, not ordinary available-RAM accounting."""
''')
text = text.replace('''        # 32 GiB box, 20 GiB available: total-floor cap 24, avail cap 14,
        # WDDM proxy cap 8 -> 8 binds.
''', '''        # 32 GiB box: total-floor cap 24, WDDM proxy cap 8 -> 8 binds.
        # Reported available RAM is intentionally not part of this cap.
''')
text = text.replace('''    def test_available_floor_binds_when_ram_tight(self):
        # only 8 GiB available -> 8 - 6 = 2 GiB cap.
        got = self._cap(12 * GIB, 32 * GIB, 8 * GIB)
        self.assertEqual(got, 2 * GIB)

    def test_zero_when_available_below_floor(self):
        got = self._cap(12 * GIB, 32 * GIB, 4 * GIB)
        self.assertEqual(got, 0)
''', '''    def test_reported_available_ram_does_not_bind_weight_pin_budget(self):
        # psutil.available is not the CUDA pinned-memory budget. Even if Windows
        # reports little pageable RAM available, the cap should remain the WDDM
        # proxy fraction for already-loaded weights.
        got = self._cap(12 * GIB, 32 * GIB, 1 * GIB)
        self.assertEqual(got, 8 * GIB)
''')
text = text.replace('''@unittest.skipUnless(torch.cuda.is_available(), "pinning requires CUDA")
class LivePinFloorGuardTests(unittest.TestCase):
    """Pinning must stop when available RAM drops below the floor, regardless
    of remaining budget, so attach can never run the host into the wall."""

    def test_floor_above_available_pins_nothing(self):
        lin = torch.nn.Linear(64, 64, bias=False)
        manager = _FakeManager(budget_bytes=1 << 40)
        with mock.patch.dict(
            os.environ, {"AI_TOOLKIT_PINNED_WEIGHT_AVAILABLE_FLOOR_GIB": "99999"}
        ):
            _move_params_to_cpu_and_pin(lin, manager)
        self.assertEqual(manager.pinned_weight_bytes, 0)
        self.assertFalse(lin.weight.data.is_pinned())

    def test_floor_disabled_pins(self):
        lin = torch.nn.Linear(64, 64, bias=False)
        manager = _FakeManager(budget_bytes=1 << 40)
        with mock.patch.dict(
            os.environ, {"AI_TOOLKIT_PINNED_WEIGHT_AVAILABLE_FLOOR_GIB": "0"}
        ):
            _move_params_to_cpu_and_pin(lin, manager)
        self.assertGreater(manager.pinned_weight_bytes, 0)
        self.assertTrue(lin.weight.data.is_pinned())
''', '''@unittest.skipUnless(torch.cuda.is_available(), "pinning requires CUDA")
class LivePinFloorGuardTests(unittest.TestCase):
    """Permanent weight pinning should not be blocked by psutil.available floors."""

    def test_available_floor_env_does_not_block_weight_pinning(self):
        lin = torch.nn.Linear(64, 64, bias=False)
        manager = _FakeManager(budget_bytes=1 << 40)
        with mock.patch.dict(
            os.environ, {"AI_TOOLKIT_PINNED_WEIGHT_AVAILABLE_FLOOR_GIB": "99999"}
        ):
            _move_params_to_cpu_and_pin(lin, manager)
        self.assertGreater(manager.pinned_weight_bytes, 0)
        self.assertTrue(lin.weight.data.is_pinned())
''')
path.write_text(text, encoding='utf-8')

path = Path(r'tests/test_pinned_budget_cycle.py')
text = path.read_text(encoding='utf-8')
text = text.replace('''    Disables the live available-RAM floor guard (AI_TOOLKIT_PINNED_WEIGHT_AVAILABLE_
    FLOOR_GIB, see test_pin_budget_caps.py) for the duration of this test: these
    tests exercise budget accounting, not host-RAM headroom, and must not depend
    on how much real RAM happens to be free on the machine running them.''', '''    These tests exercise pinned-byte accounting, not host-memory policy, and must
    not depend on how much real RAM happens to be free on the machine running
    them.''')
text = text.replace('''    def setUp(self):
        self._env_patch = mock.patch.dict(
            os.environ, {"AI_TOOLKIT_PINNED_WEIGHT_AVAILABLE_FLOOR_GIB": "0"}
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

''', '')
path.write_text(text, encoding='utf-8')

path = Path(r'tests/test_pin_quantized_weights.py')
text = path.read_text(encoding='utf-8')
text = text.replace('''    FLOOR_GIB, see test_pin_budget_caps.py): these tests exercise pin-path
    behaviour, not host-RAM headroom.''', '''    These tests exercise pin-path behaviour, not host-memory policy.''')
text = text.replace('''    def setUp(self):
        self._env_patch = mock.patch.dict(
            os.environ, {"AI_TOOLKIT_PINNED_WEIGHT_AVAILABLE_FLOOR_GIB": "0"}
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

''', '')
path.write_text(text, encoding='utf-8')
