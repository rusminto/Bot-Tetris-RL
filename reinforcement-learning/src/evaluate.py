"""
Evaluation and inference script for trained Tetris DQN agent.
Plays test games with epsilon=0 and displays terminal board visualization.
"""

import os
import argparse
import time
import numpy as np

from tetris_engine import TetrisEngine
from dqn_agent import DQNAgent


def render_board(board, lines_cleared, score, pieces):
    os.system('clear' if os.name == 'posix' else 'cls')
    print("+" + "--" * TetrisEngine.WIDTH + "+")
    for r in range(TetrisEngine.HEIGHT):
        line = "|"
        for c in range(TetrisEngine.WIDTH):
            line += "[]" if board[r, c] > 0 else "  "
        line += "|"
        print(line)
    print("+" + "--" * TetrisEngine.WIDTH + "+")
    print(f" Score: {score:6d} | Lines: {lines_cleared:4d} | Pieces: {pieces:4d}")


def evaluate(args):
    agent = DQNAgent(input_size=6, hidden_size=args.hidden_size)
    if not os.path.exists(args.model_path):
        print(f"❌ Model checkpoint not found at: {args.model_path}")
        return

    agent.load(args.model_path)
    print(f" Loaded model from: {args.model_path}")

    env = TetrisEngine()
    scores = []
    lines_list = []
    pieces_list = []

    for ep in range(1, args.episodes + 1):
        env.reset()
        next_states = env.get_next_states()

        while not env.game_over:
            action, _ = agent.select_action(next_states, epsilon=0.0)
            if action is None:
                break

            reward, done = env.step(action)

            if args.visualize:
                render_board(env.board, env.lines_cleared, env.score, env.pieces_placed)
                time.sleep(args.delay)

            if not done:
                next_states = env.get_next_states()

        scores.append(env.score)
        lines_list.append(env.lines_cleared)
        pieces_list.append(env.pieces_placed)

        print(
            f"Test Game {ep:2d}/{args.episodes} -> "
            f"Score: {env.score:6d} | "
            f"Lines Cleared: {env.lines_cleared:4d} | "
            f"Pieces Placed: {env.pieces_placed:4d}"
        )

    print("\n" + "=" * 50)
    print("📊 Evaluation Summary:")
    print(f" Average Score:         {np.mean(scores):.1f}")
    print(f" Max Score:             {np.max(scores):.1f}")
    print(f" Average Lines Cleared: {np.mean(lines_list):.1f}")
    print(f" Max Lines Cleared:     {np.max(lines_list):.1f}")
    print(f" Average Pieces Placed: {np.mean(pieces_list):.1f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Tetris DQN Agent")
    parser.add_argument("--model-path", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden dimension of MLP")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test games")
    parser.add_argument("--visualize", action="store_true", help="Display ASCII terminal animation")
    parser.add_argument("--delay", type=float, default=0.05, help="Frame delay in seconds for animation")

    args = parser.parse_args()
    evaluate(args)
