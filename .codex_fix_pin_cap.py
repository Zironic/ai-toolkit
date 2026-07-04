from pathlib import Path
path = Path(r'toolkit/memory_management/manager.py')
text = path.read_text(encoding='utf-8')
repls = [
('''        # Pinning stops weights being paged out and re-faulted on every fetch,
        # but it is NOT free: pinned pages consume OS-available RAM ~1:1 and
        # (on Windows/WDDM) count against the GPU's shared-memory budget --
''', '''        # Pinning stops weights being paged out and re-faulted on every fetch.
        # It does not create a second steady-state copy of the already-loaded
        # CPU weights, but on Windows/WDDM it does count against the GPU's
        # shared-memory budget --
'''),
('''        Measured on real weights: pinning also consumes OS-available RAM ~1:1,
        so "the weights are already in RAM, pinning is free" does not hold.
        Caps applied: total RAM minus a floor, available RAM minus a floor,
        and the process-wide pinned-bytes ledger's headroom (a fraction of
''', '''        The useful cap is the process-wide pinned-bytes ledger's headroom (a fraction of
'''),
('''        total_floor = int(
            float(_env("AI_TOOLKIT_PINNED_WEIGHT_RAM_FLOOR_GIB", "8.0")) * gib
        )
        avail_floor = int(
            float(_env("AI_TOOLKIT_PINNED_WEIGHT_AVAILABLE_FLOOR_GIB", "6.0")) * gib
        )
        ledger_headroom = bounce_pool.pinned_bytes_headroom()
        caps = [
            budget,
            max(0, int(vm.total) - total_floor - reserve_bytes),
            max(0, int(vm.available) - avail_floor - reserve_bytes),
        ]
''', '''        total_floor = int(
            float(_env("AI_TOOLKIT_PINNED_WEIGHT_RAM_FLOOR_GIB", "8.0")) * gib
        )
        ledger_headroom = bounce_pool.pinned_bytes_headroom()
        caps = [
            budget,
            max(0, int(vm.total) - total_floor - reserve_bytes),
        ]
'''),
('''                f"(available={vm.available / gib:.2f} GiB total={vm.total / gib:.2f} GiB;"
''', '''                f"(reported_available={vm.available / gib:.2f} GiB total={vm.total / gib:.2f} GiB;"
'''),
('''        # the host can actually give (see _cap_auto_pin_budget -- pinning costs
        # available RAM ~1:1 and WDDM shared-budget commit; it is NOT free).
''', '''        # the WDDM shared pinned-memory proxy can actually give (see
        # _cap_auto_pin_budget). Pinning is not free, but psutil available RAM is
        # not the relevant budget for already-loaded model weights.
'''),
('''            # Cap by real host headroom: pinning consumes available RAM ~1:1
            # and commits against the WDDM shared GPU budget -- sizing this by
            # total RAM alone crashed startup with a raw cudaErrorMemoryAllocation
            # once the pins exhausted the shared budget (LA Jinx Krea 7).
''', '''            # Cap by the WDDM shared pinned-memory proxy. psutil available RAM is
            # not authoritative for already-loaded weights; the failure mode is
            # exhausting shared pinned-memory commit, not ordinary pageable RAM.
'''),
]
for old, new in repls:
    if old not in text:
        raise SystemExit('manager target not found:\n' + old[:200])
    text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')

path = Path(r'toolkit/memory_management/manager_modules.py')
text = path.read_text(encoding='utf-8')
start = text.index('\ndef _pin_available_headroom()')
end = text.index('\ndef _move_params_to_cpu_and_pin', start)
text = text[:start] + '\n' + text[end:]
old = '''            if remaining:
                headroom = _pin_available_headroom()
                if headroom is not None:
                    remaining = min(remaining, headroom)
            cpu_data, pinned = _ensure_cpu_pinned(param.data, remaining)
'''
new = '''            cpu_data, pinned = _ensure_cpu_pinned(param.data, remaining)
'''
if old not in text:
    raise SystemExit('manager_modules pin guard block not found')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
