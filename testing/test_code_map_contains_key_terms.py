from pathlib import Path


def test_code_map_contains_core_terms():
    txt = Path('.claude/skills/training-lifecycle/references/CODE_MAP.md').read_text(encoding='utf-8').lower()
    assert 'preservation' in txt, 'CODE_MAP.md should reference preservation_loss or DOP'
    assert 'lora' in txt, 'CODE_MAP.md should reference LoRA entries'
    assert 'run.py' in txt, 'CODE_MAP.md should reference run.py/job loader'
