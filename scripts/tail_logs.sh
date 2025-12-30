#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LATENTSCORE_LOG_DIR:-${HOME}/Library/Logs/LatentScore}"
MENUBAR_LOG="${LOG_DIR}/menubar-server.log"
TUI_LOG="${LOG_DIR}/textual-ui.log"

if command -v tmux >/dev/null 2>&1; then
  SESSION="latentscore-logs"
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux attach -t "${SESSION}"
    exit 0
  fi
  tmux new-session -d -s "${SESSION}" "tail -n 200 -F \"${MENUBAR_LOG}\""
  tmux split-window -h -t "${SESSION}" "tail -n 200 -F \"${TUI_LOG}\""
  tmux select-layout -t "${SESSION}" even-horizontal
  tmux attach -t "${SESSION}"
else
  echo "tmux not found; falling back to two background tails." >&2
  tail -n 200 -F "${MENUBAR_LOG}" &
  tail -n 200 -F "${TUI_LOG}" &
  wait
fi
