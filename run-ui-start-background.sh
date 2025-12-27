#!/usr/bin/env bash
# run-ui-start-background.sh - start the production UI server in the background (agent-safe)
# Usage: ./run-ui-start-background.sh
# Writes PID to ui/ui_start.pid
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/ui"

# Start build_and_start in background
nohup npm run build_and_start >/dev/null 2>&1 &
PID=$!
echo $PID > ui_start.pid
printf "Started UI server (pid=%s). PID written to ui/ui_start.pid\n" "$PID"
