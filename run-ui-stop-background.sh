#!/usr/bin/env bash
# run-ui-stop-background.sh - stop the background UI server started by run-ui-start-background.sh
# Usage: ./run-ui-stop-background.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/ui"

if [ ! -f ui_start.pid ]; then
  echo "PID file ui/ui_start.pid not found"
  exit 1
fi
PID=$(cat ui_start.pid)
kill "$PID" || true
rm -f ui_start.pid
echo "Stopped UI server (pid=$PID) and removed ui/ui_start.pid"
