from pathlib import Path
path = Path(r'tests/test_pin_budget_caps.py')
text = path.read_text(encoding='utf-8')
insert = '''
    def test_bounce_reserve_is_subtracted_before_weight_pin_budget(self):
        got = self._cap(12 * GIB, 32 * GIB, 20 * GIB, reserve_bytes=2 * GIB)
        self.assertEqual(got, 6 * GIB)
'''
text = text.replace('''    def _cap(self, budget, total, available):
        with mock.patch.dict(os.environ, _ENV):
            with mock.patch.object(
                bounce_pool, "_psutil", _FakePsutil(total, available)
            ):
                return MemoryManager._cap_auto_pin_budget(budget)
''', '''    def _cap(self, budget, total, available, reserve_bytes=0):
        with mock.patch.dict(os.environ, _ENV):
            with mock.patch.object(
                bounce_pool, "_psutil", _FakePsutil(total, available)
            ):
                return MemoryManager._cap_auto_pin_budget(
                    budget, reserve_bytes=reserve_bytes
                )
''')
marker = '''    def test_small_budget_passes_through(self):
'''
if insert not in text:
    if marker not in text:
        raise SystemExit('marker not found')
    text = text.replace(marker, insert + '\n' + marker, 1)
path.write_text(text, encoding='utf-8')
