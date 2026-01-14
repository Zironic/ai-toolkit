#!/usr/bin/env python3
"""Generate a machine-readable code index for the training-lifecycle skill.

Output: references/CODE_INDEX.json

For each file listed in `references/CODE_MAP.md` 'key files', record:
- path
- short description (first non-empty comment or top-level docstring)
- symbols (class and def) with line numbers
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve()
for _ in range(6):
    ROOT = ROOT.parent
    if (ROOT / "run.py").exists() or (ROOT / ".git").exists():
        break
CODE_MAP = Path(__file__).resolve().parents[1] / "references" / "CODE_MAP.md"
OUT = Path(__file__).resolve().parents[1] / "references" / "CODE_INDEX.json"

if not CODE_MAP.exists():
    raise SystemExit("CODE_MAP.md not found")

text = CODE_MAP.read_text()

# naive parser: find 'key files:' lines and capture the glob inside backticks
file_globs = []
for line in text.splitlines():
    m = re.search(r"- key files: `(.*?)`", line)
    if m:
        g = m.group(1)
        file_globs.append(g)

# Expand globs and deduplicate
files = set()
for g in file_globs:
    try:
        # If glob contains wildcard, use rglob, otherwise treat as path
        if '*' in g or g.endswith('/'):
            for p in ROOT.rglob(g):
                if p.is_file():
                    files.add(p)
        else:
            # may be a directory or a single file
            p = ROOT / g
            if p.exists():
                if p.is_dir():
                    for f in p.rglob('*.py'):
                        files.add(f)
                elif p.is_file():
                    files.add(p)
            else:
                # try rglob for basename
                for f in ROOT.rglob(g):
                    files.add(f)
    except Exception as ex:
        print(f"Warning: failed to expand glob '{g}': {ex}")

# fallback: include some known files if none matched
if not files:
    candidates = [
        "extensions_built_in/sd_trainer/SDTrainer.py",
        "toolkit/dataloader_mixins.py",
        "run.py",
    ]
    for c in candidates:
        p = ROOT / c
        if p.exists():
            files.add(p)

print(f"Found {len(files)} files to index")

index = {}
for f in sorted(files):
    try:
        txt = f.read_text(errors='ignore')
    except Exception:
        continue
    lines = txt.splitlines()
    # short description: first module docstring or first block comment
    description = None
    # module docstring
    m = re.match(r"\s*\"\"\"(.*?)\"\"\"", txt, re.S)
    if m:
        description = m.group(1).strip().splitlines()[0]
    else:
        # first top-level comment
        for line in lines[:40]:
            s = line.strip()
            if s.startswith('#'):
                description = s.lstrip('#').strip()
                break
    if not description:
        description = "No module docstring or top comment"

    symbols = []
    for i, line in enumerate(lines, start=1):
        c = re.match(r"\s*class\s+(\w+)\s*\(|\s*def\s+(\w+)\s*\(|async def\s+(\w+)\s*\(", line)
        if c:
            name = c.group(1) or c.group(2) or c.group(3)
            symbols.append({"name": name, "line": i, "snippet": line.strip()})
    index[str(f.relative_to(ROOT))] = {"description": description, "symbols": symbols[:200]}

try:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(index, indent=2))
    print(f"Wrote {OUT}")
except Exception as ex:
    print(f"Failed to write {OUT}: {ex}")
    raise

