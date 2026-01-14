from pathlib import Path


def test_agent_procedure_present():
    txt = Path('.claude/skills/training-lifecycle/SKILL.md').read_text(encoding='utf-8')
    assert 'Agent procedure' in txt or 'Agent Procedure' in txt, 'SKILL.md should include a clear Agent procedure section (MANDATORY)'
