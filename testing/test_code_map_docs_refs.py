from pathlib import Path


def test_code_map_docs_exist():
    code_map = Path(".claude/skills/training-lifecycle/references/CODE_MAP.md").read_text()
    docs_dir = Path("docs/training_lifecycle")
    missing = []
    for line in code_map.splitlines():
        line = line.strip()
        if line.startswith("- docs:"):
            # extract path between backticks
            if "`" in line:
                path = line.split("`")[1]
                if not (Path(path).exists()):
                    missing.append(path)
    assert not missing, f"CODE_MAP references missing docs files: {missing}"
