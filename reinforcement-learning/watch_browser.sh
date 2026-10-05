#!/usr/bin/env bash
# Starts the DQN web viewer on the remote server (if needed), tunnels it to localhost and opens it.
# The server comes from ../.env (see ../.env.example) or the environment, like run_training_remote.sh.
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
REMOTE_PORT="${TETRIS_RL_REMOTE_PORT:-22}"
DIR="${TETRIS_RL_REMOTE_DIR:-tetris-rl}"
PORT="${VIEWER_PORT:-5000}"
URL="http://127.0.0.1:${PORT}/"

ssh -p "$REMOTE_PORT" "$REMOTE" \
  "tmux has-session -t tetris-web 2>/dev/null || tmux new-session -d -s tetris-web 'cd \$HOME/$DIR && venv/bin/python3 -u src/web_viewer.py --port $PORT'"

if ss -lnt | grep -q ":${PORT} "; then
  echo "Port ${PORT} is already forwarded."
else
  echo "Tunnelling ${REMOTE}:${PORT} to localhost:${PORT} ..."
  ssh -f -N -p "$REMOTE_PORT" -L "${PORT}:127.0.0.1:${PORT}" "$REMOTE"
  sleep 1
fi

if { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; } && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 &
fi
echo "Viewer: $URL"
