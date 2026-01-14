from pathlib import Path


def test_skill_primary_index():
    text = Path('.claude/skills/training-lifecycle/SKILL.md').read_text(encoding='utf-8').lower()
    assert ('primary index' in text) or ('must consult' in text) or ('consult this skill' in text), "SKILL.md should clearly state it is the primary index and that agents must consult it before broad searches or edits"
