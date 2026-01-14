from pathlib import Path


def parse_frontmatter(path: Path):
    txt = path.read_text(encoding='utf-8')
    assert txt.startswith('---')
    parts = txt.split('---')
    assert len(parts) >= 3
    front = parts[1]
    keys = []
    for line in front.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            keys.append(line.split(':', 1)[0].strip())
    return keys


def test_claude_skill_frontmatter_minimal():
    p = Path('.claude/skills/training-lifecycle/SKILL.md')
    keys = parse_frontmatter(p)
    assert set(keys) <= {'name', 'description'}, f"Frontmatter should contain only name and description, found: {keys}"


def test_skill_description_mentions_training():
    txt = Path('.claude/skills/training-lifecycle/SKILL.md').read_text(encoding='utf-8').lower()
    assert 'training' in txt, 'SKILL.md description/body should reference "training" as a trigger'