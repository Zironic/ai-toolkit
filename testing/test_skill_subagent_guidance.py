from pathlib import Path


def test_skill_mentions_subagents():
    txt = Path('.claude/skills/training-lifecycle/SKILL.md').read_text(encoding='utf-8').lower()
    assert 'subagent' in txt or 'raptor mini' in txt.lower(), 'SKILL.md should instruct agents to prefer using subagents (e.g., Raptor Mini) for targeted searches'