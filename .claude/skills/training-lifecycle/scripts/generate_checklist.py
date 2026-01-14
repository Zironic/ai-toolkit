#!/usr/bin/env python3
"""Simple helper: generate a markdown checklist from references/REFERENCE.md"""

from pathlib import Path

REF = Path(__file__).resolve().parents[1] / "references" / "REFERENCE.md"

if not REF.exists():
    raise SystemExit(f"Missing references file: {REF}")

lines = REF.read_text().splitlines()

# Collect headings (## ...) and first action-item line after heading
checklist = []
current_section = None

for i, line in enumerate(lines):
    if line.startswith("## "):
        current_section = line[3:].strip()
    elif current_section and line.strip().startswith("- Action items:"):
        # next lines may list items
        j = i + 1
        while j < len(lines) and lines[j].strip().startswith("-"):
            checklist.append((current_section, lines[j].strip()[2:].strip()))
            j += 1
        current_section = None

# Fallback: if no action items found, create checklist from headings
if not checklist:
    for line in lines:
        if line.startswith("## "):
            checklist.append((line[3:].strip(), "Review chapter and add verification steps"))

out = ["# Training lifecycle checklist", ""]
for section, item in checklist:
    out.append(f"- [ ] **{section}** — {item}")

print("\n".join(out))
