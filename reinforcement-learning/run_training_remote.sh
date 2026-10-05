#!/usr/bin/env bash
# Syncs reinforcement-learning/ to a remote server and starts training there in tmux.
#
#   ./run_training_remote.sh          # CEM policy search (the policy the userscript ships)
#   ./run_training_remote.sh dqn      # DQN v2
#   ./run_training_remote.sh fetch    # copy checkpoints and logs back
#
# The server comes from ../.env (see ../.env.example) or the environment:
#   TETRIS_RL_REMOTE=user@host  TETRIS_RL_REMOTE_PORT=22  TETRIS_RL_REMOTE_DIR=tetris-rl  TETRIS_RL_WORKERS=12
set -euo pipefail

# Settings from ../.env, without overriding variables that are already set
load_env() {
  local file="$1" line key
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in '' | '#'*) continue ;; esac
    key="${line%%=*}"
    if [ -z "${!key+x}" ]; then export "$line"; fi
  done < "$file"
}

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
load_env "$LOCAL_DIR/../.env"
REMOTE="${TETRIS_RL_REMOTE:?Set TETRIS_RL_REMOTE=user@host (see .env.example)}"
PORT="${TETRIS_RL_REMOTE_PORT:-22}"
DIR="${TETRIS_RL_REMOTE_DIR:-tetris-rl}"   # relative to the remote home directory
WORKERS="${TETRIS_RL_WORKERS:-12}"
MODE="${1:-cem}"
SSH=(ssh -p "$PORT" "$REMOTE")

case "$MODE" in
  cem) SESSION="cem-train"; CMD="venv/bin/python3 -u src/cem_train.py --num-workers $WORKERS 2>&1 | tee logs/cem_train.log" ;;
  dqn) SESSION="dqn-train"; CMD="venv/bin/python3 -u src/train.py --episodes 5000 2>&1 | tee logs/dqn_train.log" ;;
  fetch)
    for d in cem_checkpoints checkpoints logs; do
      rsync -av -e "ssh -p $PORT" "$REMOTE:$DIR/$d/" "$LOCAL_DIR/$d/"
    done
    exit 0 ;;
  *) echo "usage: $0 [cem|dqn|fetch]" >&2; exit 1 ;;
esac

echo "Syncing code to $REMOTE:$DIR ..."
"${SSH[@]}" "mkdir -p $DIR/logs"
rsync -avz -e "ssh -p $PORT" \
  --exclude venv --exclude __pycache__ --exclude 'runs*' --exclude 'checkpoints*' \
  --exclude cem_checkpoints --exclude logs \
  "$LOCAL_DIR/" "$REMOTE:$DIR/"

echo "Starting tmux session '$SESSION' ..."
# ssh joins its arguments into one remote command line, so quote them for the remote shell
"${SSH[@]}" "bash -s -- $(printf '%q ' "$DIR" "$SESSION" "$CMD")" <<'EOF'
set -e
cd "$HOME/$1"
if [ ! -x venv/bin/python3 ]; then
  python3 -m venv venv
  venv/bin/pip install -r requirements.txt
fi
if tmux has-session -t "$2" 2>/dev/null; then
  echo "Session '$2' is already running."
else
  tmux new-session -d -s "$2" "cd $HOME/$1 && $3"
  echo "Started."
fi
EOF

echo
echo "Attach:  ssh -t -p $PORT $REMOTE tmux attach -t $SESSION   (Ctrl+B then D to detach)"
echo "Results: $0 fetch"
