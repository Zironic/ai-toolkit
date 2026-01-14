#!/usr/bin/env python3
"""Minimal validator for SKILL.md frontmatter and basic structure.

Usage: python scripts/validate_skill.py [path-to-skill-dir]
"""
import os
import re
import sys

SKILL_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(__file__))
SKILL_MD = os.path.join(SKILL_DIR, "SKILL.md")
SKILL_NAME = os.path.basename(SKILL_DIR)

if not os.path.exists(SKILL_MD):
    print(f"ERROR: SKILL.md not found in {SKILL_DIR}")
    sys.exit(2)

text = open(SKILL_MD, "r", encoding="utf-8").read()
if not text.startswith("---"):
    print("ERROR: SKILL.md missing YAML frontmatter start (---)")
    sys.exit(2)

# Extract frontmatter
parts = text.split("---")
if len(parts) < 3:
    print("ERROR: SKILL.md frontmatter seems malformed")
    sys.exit(2)
front = parts[1]

# Simple parse for name and description
name_match = re.search(r"^name:\s*(.+)$", front, re.MULTILINE)
desc_match = re.search(r"^description:\s*(.+)$", front, re.MULTILINE)

# Parse ALL frontmatter keys and ensure only 'name' and 'description' are present (Anthropic best practice)
keys = set()
for line in front.splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_-]*)\s*:', line)
    if m:
        keys.add(m.group(1))
allowed = {'name', 'description'}
if not keys.issubset(allowed):
    extra = sorted(keys - allowed)
    errors.append(f"frontmatter contains extra fields not allowed in minimal SKILL.md: {extra}")

errors = []
if not name_match:
    errors.append("missing 'name' field in frontmatter")
else:
    name = name_match.group(1).strip().strip('"')
    # name must match dir name
    if name != SKILL_NAME:
        errors.append(f"name '{name}' does not match directory name '{SKILL_NAME}'")
    if not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", name):
        errors.append("name must be lowercase alphanumeric and hyphens, no leading/trailing/duplicate hyphens")

if not desc_match:
    errors.append("missing 'description' field in frontmatter")
else:
    desc = desc_match.group(1).strip().strip('"')
    if len(desc) == 0 or len(desc) > 1024:
        errors.append("description must be 1-1024 characters")

if errors:
    print("SKILL validation failed:")
    for e in errors:
        print(" - ", e)
    sys.exit(2)

print("SKILL validation OK: frontmatter looks good")
sys.exit(0)
