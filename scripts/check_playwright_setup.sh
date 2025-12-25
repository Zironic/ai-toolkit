#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UI_DIR="$REPO_ROOT/ui"

echo "Repo root: $REPO_ROOT"
echo "UI folder: $UI_DIR"

if [ ! -d "$UI_DIR" ]; then
  echo "ERROR: UI folder not found at $UI_DIR" >&2
  exit 2
fi

echo "Checking Playwright version (using local ui package)..."
# Use npm --prefix so it resolves the ui package regardless of CWD
npm --prefix "$UI_DIR" exec -- playwright --version || echo "Playwright CLI failed"

echo "Ensuring Playwright browsers are installed..."
npm --prefix "$UI_DIR" exec -- playwright install --with-deps

echo "Playwright setup looks OK."