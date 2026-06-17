#!/usr/bin/env bash
# Launch the full VoxPopuli pipeline inside a detached tmux session.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="voxpopuli"
LOG_DIR="$ROOT/logs/runpod"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/voxpopuli_$(date +%Y%m%d_%H%M%S).log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "bash scripts/runpod/voxpopuli_pipeline.sh 2>&1 | tee -a '$LOG_FILE'"

echo "Started tmux session: $SESSION"
echo "Log file: $LOG_FILE"
echo "Attach:   tmux attach -t $SESSION"
echo "Detach:   Ctrl-b then d"
