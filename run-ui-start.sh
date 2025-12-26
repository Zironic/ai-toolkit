#!/usr/bin/env bash
# run-ui-start.sh - start the production UI server in background (agent-safe)
# Usage: ./run-ui-start.sh
# Writes PID to ui/ui_start.pid
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/ui"

# Ensure dependencies and DB
npm --silent install
npm --silent run update_db
npm --silent run build

# Start in background with nohup
nohup npm run start >/dev/null 2>&1 &
PID=$!
echo $PID > ui_start.pid
printf "Started UI server (pid=%s). PID written to ui/ui_start.pid\n" "$PID"
