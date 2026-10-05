"""
Real-time Web Visualizer for Tetris Reinforcement Learning.
Serves an interactive HTML5/Canvas UI and streams live gameplay of the agent
using Server-Sent Events (SSE). Automatically reloads checkpoints as training progresses.
"""

import os
import sys
import time
import json
import queue
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tetris_engine import TetrisEngine, PIECES
from dqn_agent import DQNAgent

# Global simulation state shared with HTTP handler
game_state = {
    "board": [[0] * 10 for _ in range(20)],
    "current_piece": "T",
    "next_piece": "I",
    "ghost_piece": None,
    "score": 0,
    "lines_cleared": 0,
    "pieces_placed": 0,
    "action": {"rotation": 0, "col": 0},
    "model_loaded": False,
    "model_mtime": 0,
    "status": "Initializing...",
    "speed": 0.08,  # Seconds per move
    "paused": False
}

state_lock = threading.Lock()
subscribers = []
subscribers_lock = threading.Lock()


def get_piece_blocks(piece_name, rot_idx, x, y):
    """Return list of (r, c) board coordinates occupied by the piece."""
    orientations = PIECES[piece_name]
    shape = orientations[rot_idx]
    blocks = []
    for r in range(len(shape)):
        for c in range(len(shape[0])):
            if shape[r][c]:
                blocks.append([y + r, x + c])
    return blocks


def get_ghost_y(env, shape, x):
    """Find the landing row y for the ghost piece."""
    y = 0
    while env._is_valid_position(shape, x, y + 1, env.board):
        y += 1
    return y


def simulation_worker(model_path, checkpoint_dir):
    """Background thread that runs live test games with the latest model."""
    global game_state

    agent = DQNAgent(input_size=6, hidden_size=64)  # 6 features, same as tetris_engine.py and train.py
    env = TetrisEngine()
    last_mtime = 0

    while True:
        # Check if best_model.pt has updated
        target_path = model_path
        if not os.path.exists(target_path):
            alt_path = os.path.join(checkpoint_dir, "best_model.pt")
            if os.path.exists(alt_path):
                target_path = alt_path

        if os.path.exists(target_path):
            current_mtime = os.path.getmtime(target_path)
            if current_mtime != last_mtime:
                try:
                    agent.load(target_path)
                    last_mtime = current_mtime
                    with state_lock:
                        game_state["model_loaded"] = True
                        game_state["model_mtime"] = current_mtime
                        game_state["status"] = f"Loaded checkpoint: {os.path.basename(target_path)}"
                    print(f" Loaded updated model from {target_path}")
                except Exception as e:
                    print(f"⚠️ Error loading checkpoint: {e}")

        # Start a game
        env.reset()
        next_states = env.get_next_states()

        while not env.game_over:
            with state_lock:
                paused = game_state["paused"]
                speed = game_state["speed"]

            if paused:
                time.sleep(0.2)
                continue

            # Agent picks best action (greedy)
            if game_state["model_loaded"] and next_states:
                action, _ = agent.select_action(next_states, epsilon=0.0)
            elif next_states:
                # Random play if model not loaded yet
                action = list(next_states.keys())[0]
            else:
                action = None

            if action is None:
                break

            use_hold, rot_idx, target_x = action
            # with hold, the placed piece is the held one (or the next piece if nothing is held yet)
            placed_piece = env.current_piece
            if use_hold:
                placed_piece = env.hold_piece if env.hold_piece is not None else env.next_piece
            shape = PIECES[placed_piece][rot_idx]
            ghost_y = get_ghost_y(env, shape, target_x)
            ghost_blocks = get_piece_blocks(placed_piece, rot_idx, target_x, ghost_y)

            # Update state for broadcast with ghost piece
            with state_lock:
                game_state["board"] = env.board.tolist()
                game_state["current_piece"] = env.current_piece
                game_state["next_piece"] = env.next_piece
                game_state["ghost_piece"] = {
                    "blocks": ghost_blocks,
                    "piece": placed_piece
                }
                game_state["action"] = {"rotation": rot_idx, "col": target_x}
                game_state["score"] = env.score
                game_state["lines_cleared"] = env.lines_cleared
                game_state["pieces_placed"] = env.pieces_placed

            broadcast_state()
            time.sleep(speed)

            # Execute placement
            reward, done = env.step(action)

            # Update board immediately after placement
            with state_lock:
                game_state["board"] = env.board.tolist()
                game_state["score"] = env.score
                game_state["lines_cleared"] = env.lines_cleared
                game_state["pieces_placed"] = env.pieces_placed
                game_state["ghost_piece"] = None

            broadcast_state()

            if not done:
                next_states = env.get_next_states()

        # Game over delay before starting new game
        with state_lock:
            game_state["status"] = f"Game Over! Lines: {env.lines_cleared}, Score: {env.score}. Restarting..."
        broadcast_state()
        time.sleep(1.0)


def broadcast_state():
    """Enqueue current game state for all connected SSE clients."""
    with state_lock:
        data = json.dumps(game_state)
    msg = f"data: {data}\n\n"

    with subscribers_lock:
        for q in list(subscribers):
            try:
                if q.full():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                q.put_nowait(msg)
            except Exception:
                pass


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>🎮 Tetris RL - Live Browser Viewer</title>
  <style>
    :root {
      --bg: #0f111a;
      --card-bg: #1a1c29;
      --text: #ffffff;
      --text-dim: #8b92b2;
      --accent: #00d2ff;
      --accent-glow: rgba(0, 210, 255, 0.4);
      --border: #2a2e42;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 20px;
    }
    header {
      text-align: center;
      margin-bottom: 20px;
    }
    h1 {
      font-size: 2.2rem;
      letter-spacing: 2px;
      background: linear-gradient(45deg, #00d2ff, #9b51e0, #ff4081);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 6px;
    }
    .badge {
      display: inline-block;
      padding: 4px 12px;
      font-size: 0.8rem;
      border-radius: 20px;
      background: #232738;
      color: var(--accent);
      border: 1px solid var(--border);
    }
    .main-container {
      display: flex;
      gap: 30px;
      background: var(--card-bg);
      padding: 24px;
      border-radius: 16px;
      border: 1px solid var(--border);
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
      max-width: 900px;
      width: 100%;
    }
    .board-container {
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    canvas#tetris {
      background: #090a0f;
      border: 3px solid #2e334d;
      border-radius: 8px;
      box-shadow: 0 0 20px rgba(0,0,0,0.8), inset 0 0 10px rgba(0,0,0,0.6);
    }
    .sidebar {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .stat-card {
      background: #12141f;
      padding: 14px 18px;
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    .stat-label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-dim);
      margin-bottom: 4px;
    }
    .stat-value {
      font-size: 1.6rem;
      font-weight: 700;
      color: #00ffcc;
      font-family: monospace;
    }
    .next-box {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 90px;
    }
    .controls {
      background: #12141f;
      padding: 16px;
      border-radius: 10px;
      border: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .slider-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.85rem;
    }
    input[type=range] {
      width: 100%;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .btn-row {
      display: flex;
      gap: 10px;
    }
    button {
      flex: 1;
      padding: 10px;
      border: none;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    button.btn-primary {
      background: var(--accent);
      color: #000;
    }
    button.btn-primary:hover {
      box-shadow: 0 0 12px var(--accent-glow);
    }
    button.btn-secondary {
      background: #2e334d;
      color: #fff;
    }
    button.btn-secondary:hover {
      background: #3e4466;
    }
    .status-bar {
      font-size: 0.8rem;
      color: var(--text-dim);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #00ff88;
      box-shadow: 0 0 8px #00ff88;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0% { opacity: 0.4; }
      50% { opacity: 1; }
      100% { opacity: 0.4; }
    }
  </style>
</head>
<body>
  <header>
    <h1>TETRIS REINFORCEMENT LEARNING</h1>
    <div class="badge">🚀 Live playback • Deep Q-Network (DQN v2)</div>
  </header>

  <div class="main-container">
    <div class="board-container">
      <canvas id="tetris" width="280" height="560"></canvas>
    </div>

    <div class="sidebar">
      <div class="stat-card">
        <div class="stat-label">Next Piece</div>
        <div class="next-box">
          <canvas id="next-canvas" width="120" height="70"></canvas>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-label">Score</div>
        <div class="stat-value" id="val-score">0</div>
      </div>

      <div class="stat-card">
        <div class="stat-label">Lines Cleared</div>
        <div class="stat-value" id="val-lines">0</div>
      </div>

      <div class="stat-card">
        <div class="stat-label">Pieces Placed</div>
        <div class="stat-value" id="val-pieces">0</div>
      </div>

      <div class="controls">
        <div class="slider-row">
          <span>Animation Speed:</span>
          <span id="speed-label">Normal</span>
        </div>
        <input type="range" id="speed-slider" min="10" max="300" value="80" step="10">

        <div class="btn-row">
          <button class="btn-primary" id="btn-pause">Pause</button>
          <button class="btn-secondary" id="btn-reload">Reload Model</button>
        </div>
      </div>

      <div class="status-bar">
        <div class="pulse-dot"></div>
        <span id="status-text">Connected to model stream</span>
      </div>
    </div>
  </div>

  <script>
    const canvas = document.getElementById('tetris');
    const ctx = canvas.getContext('2d');
    const nextCanvas = document.getElementById('next-canvas');
    const nextCtx = nextCanvas.getContext('2d');

    const BLOCK_SIZE = 28;
    const COLS = 10;
    const ROWS = 20;

    // Piece colors (Standard Tetris Colors)
    const COLORS = {
      1: '#00ffff', // I - Cyan
      2: '#0000ff', // J - Blue
      3: '#ffaa00', // L - Orange
      4: '#ffff00', // O - Yellow
      5: '#00ff00', // S - Green
      6: '#aa00ff', // T - Purple
      7: '#ff0000', // Z - Red
      'I': '#00ffff',
      'J': '#0055ff',
      'L': '#ff8800',
      'O': '#ffee00',
      'S': '#00ee44',
      'T': '#aa00ee',
      'Z': '#ee2222'
    };

    const PIECE_SHAPES = {
      'I': [[1, 1, 1, 1]],
      'O': [[1, 1], [1, 1]],
      'T': [[0, 1, 0], [1, 1, 1]],
      'S': [[0, 1, 1], [1, 1, 0]],
      'Z': [[1, 1, 0], [0, 1, 1]],
      'J': [[1, 0, 0], [1, 1, 1]],
      'L': [[0, 0, 1], [1, 1, 1]]
    };

    function drawBlock(c, x, y, color, isGhost = false) {
      const px = x * BLOCK_SIZE;
      const py = y * BLOCK_SIZE;

      if (isGhost) {
        c.strokeStyle = color;
        c.lineWidth = 2;
        c.strokeRect(px + 2, py + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4);
        c.fillStyle = color + '22';
        c.fillRect(px + 2, py + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4);
        return;
      }

      // Filled block with bevel
      c.fillStyle = color;
      c.fillRect(px, py, BLOCK_SIZE, BLOCK_SIZE);

      c.fillStyle = 'rgba(255, 255, 255, 0.35)';
      c.fillRect(px, py, BLOCK_SIZE, 3);
      c.fillRect(px, py, 3, BLOCK_SIZE);

      c.fillStyle = 'rgba(0, 0, 0, 0.35)';
      c.fillRect(px, py + BLOCK_SIZE - 3, BLOCK_SIZE, 3);
      c.fillRect(px + BLOCK_SIZE - 3, py, 3, BLOCK_SIZE);
    }

    function render(state) {
      // Clear board
      ctx.fillStyle = '#090a0f';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Grid lines
      ctx.strokeStyle = '#181b28';
      ctx.lineWidth = 1;
      for (let x = 0; x <= canvas.width; x += BLOCK_SIZE) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = 0; y <= canvas.height; y += BLOCK_SIZE) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }

      // Draw locked board cells
      if (state.board) {
        for (let r = 0; r < ROWS; r++) {
          for (let c = 0; c < COLS; c++) {
            const val = state.board[r][c];
            if (val > 0) {
              const col = COLORS[val] || '#00d2ff';
              drawBlock(ctx, c, r, col);
            }
          }
        }
      }

      // Draw ghost piece (planned placement)
      if (state.ghost_piece && state.ghost_piece.blocks) {
        const color = COLORS[state.ghost_piece.piece] || '#ffffff';
        state.ghost_piece.blocks.forEach(([r, c]) => {
          drawBlock(ctx, c, r, color, true);
        });
      }

      // Draw next piece preview
      nextCtx.fillStyle = '#12141f';
      nextCtx.fillRect(0, 0, nextCanvas.width, nextCanvas.height);
      if (state.next_piece && PIECE_SHAPES[state.next_piece]) {
        const shape = PIECE_SHAPES[state.next_piece];
        const color = COLORS[state.next_piece];
        const h = shape.length;
        const w = shape[0].length;
        const offX = Math.floor((nextCanvas.width - w * 18) / 2);
        const offY = Math.floor((nextCanvas.height - h * 18) / 2);

        for (let r = 0; r < h; r++) {
          for (let c = 0; c < w; c++) {
            if (shape[r][c]) {
              nextCtx.fillStyle = color;
              nextCtx.fillRect(offX + c * 18, offY + r * 18, 16, 16);
            }
          }
        }
      }

      // Update text
      document.getElementById('val-score').textContent = state.score.toLocaleString();
      document.getElementById('val-lines').textContent = state.lines_cleared;
      document.getElementById('val-pieces').textContent = state.pieces_placed;
      if (state.status) {
        document.getElementById('status-text').textContent = state.status;
      }
    }

    let lastStateTime = Date.now();
    function fetchState() {
      fetch('/state')
        .then(r => r.json())
        .then(data => {
          lastStateTime = Date.now();
          render(data);
        })
        .catch(() => {});
    }
    // Fetch immediately on load
    fetchState();

    // Connect to SSE stream
    let evtSource = null;
    function connectSSE() {
      if (evtSource) evtSource.close();
      evtSource = new EventSource('/stream');
      evtSource.onmessage = (e) => {
        lastStateTime = Date.now();
        const data = JSON.parse(e.data);
        render(data);
      };
      evtSource.onerror = () => {
        document.getElementById('status-text').textContent = 'Connecting...';
      };
    }
    connectSSE();

    // Heartbeat / Fallback polling watchdog in case SSE is throttled by browser
    setInterval(() => {
      if (Date.now() - lastStateTime > 800) {
        fetchState();
      }
    }, 600);

    // Controls
    const speedSlider = document.getElementById('speed-slider');
    const speedLabel = document.getElementById('speed-label');
    speedSlider.oninput = (e) => {
      const delayMs = parseInt(e.target.value);
      speedLabel.textContent = delayMs < 50 ? 'Turbo' : (delayMs < 120 ? 'Normal' : 'Slow');
      fetch('/control?speed=' + (delayMs / 1000.0));
    };

    let isPaused = false;
    const btnPause = document.getElementById('btn-pause');
    btnPause.onclick = () => {
      isPaused = !isPaused;
      btnPause.textContent = isPaused ? 'Resume' : 'Pause';
      fetch('/control?paused=' + isPaused);
    };

    const btnReload = document.getElementById('btn-reload');
    btnReload.onclick = () => {
      fetch('/control?reload=true');
    };
  </script>
</body>
</html>
"""


class TetrisHTTPHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        global game_state

        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path == "/state":
            with state_lock:
                data = json.dumps(game_state)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data.encode("utf-8"))

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q = queue.Queue(maxsize=20)
            with subscribers_lock:
                subscribers.append(q)

            # Send current state immediately on connect
            with state_lock:
                init_msg = f"data: {json.dumps(game_state)}\n\n"
            try:
                self.wfile.write(init_msg.encode("utf-8"))
                self.wfile.flush()
                while True:
                    msg = q.get(timeout=10)
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
            except (queue.Empty, BrokenPipeError, ConnectionResetError, Exception):
                pass
            finally:
                with subscribers_lock:
                    if q in subscribers:
                        subscribers.remove(q)

        elif self.path.startswith("/control"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)

            with state_lock:
                if "speed" in query:
                    try:
                        game_state["speed"] = max(0.005, float(query["speed"][0]))
                    except ValueError:
                        pass
                if "paused" in query:
                    game_state["paused"] = query["paused"][0].lower() == "true"
                if "reload" in query:
                    game_state["model_mtime"] = 0  # Trigger reload

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence default request logging to avoid terminal clutter
        pass


def run_server(port=5000, model_path="checkpoints/best_model.pt", checkpoint_dir="checkpoints"):
    # Start simulation worker in background daemon thread
    worker = threading.Thread(
        target=simulation_worker,
        args=(model_path, checkpoint_dir),
        daemon=True
    )
    worker.start()

    server = ThreadingHTTPServer(("0.0.0.0", port), TetrisHTTPHandler)
    server.daemon_threads = True
    print("=" * 65)
    print(f" 🎮 Tetris RL Web Visualizer is running (ThreadingHTTPServer)!")
    print(f" 🌐 Access URL: http://localhost:{port}/")
    print(f" Monitoring checkpoints in: {checkpoint_dir}")
    print("=" * 65)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web visualizer...")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tetris RL Web Visualizer")
    parser.add_argument("--port", type=int, default=5000, help="HTTP port to serve visualizer")
    parser.add_argument("--model-path", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoints directory")

    args = parser.parse_args()
    run_server(port=args.port, model_path=args.model_path, checkpoint_dir=args.checkpoint_dir)
