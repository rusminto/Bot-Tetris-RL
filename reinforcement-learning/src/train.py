"""
Training loop for Tetris RL with Deep Q-Networks.
Logs metrics to TensorBoard and saves checkpoints.
"""

import os
import argparse
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tetris_engine import TetrisEngine
from dqn_agent import DQNAgent


def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.log_dir)
    env = TetrisEngine()
    agent = DQNAgent(
        input_size=6,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        replay_size=args.replay_size
    )

    epsilon = args.epsilon_start
    best_score = 0
    best_lines = 0

    print("=" * 65, flush=True)
    print(" 🚀 Starting Tetris Reinforcement Learning (DQN)", flush=True)
    print(f" Device:         {agent.device}", flush=True)
    print(f" Episodes:       {args.episodes}", flush=True)
    print(f" Batch Size:     {args.batch_size}", flush=True)
    print(f" Replay Buffer:  {args.replay_size}", flush=True)
    print(f" Checkpoint Dir: {args.checkpoint_dir}", flush=True)
    print(f" Log Dir:        {args.log_dir}", flush=True)
    print("=" * 65, flush=True)

    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        env.reset()
        episode_reward = 0
        losses = []

        # Get initial next states for first piece
        next_states = env.get_next_states()
        current_state_features = None

        while not env.game_over:
            # Agent picks best action and associated state features
            action, chosen_state_features = agent.select_action(next_states, epsilon=epsilon)
            if action is None:
                break

            # Execute action in environment
            reward, done = env.step(action)
            episode_reward += reward

            # If not done, compute candidate states for next piece
            if not done:
                next_states = env.get_next_states()
                # Peak into greedy next best state for target Q estimation
                best_next_action, next_best_features = agent.select_action(next_states, epsilon=0.0)
            else:
                next_best_features = None

            # Store transition in replay buffer
            agent.store_transition(chosen_state_features, reward, next_best_features, done)

            # Perform a training step
            loss = agent.train_step(batch_size=args.batch_size)
            if loss is not None:
                losses.append(loss)

        # Decay exploration rate epsilon
        if epsilon > args.epsilon_end:
            epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        # Update target network periodically
        if episode % args.target_update == 0:
            agent.update_target_network()

        # Log episode metrics
        avg_loss = np.mean(losses) if losses else 0.0
        writer.add_scalar("Score", env.score, episode)
        writer.add_scalar("Lines_Cleared", env.lines_cleared, episode)
        writer.add_scalar("Level", env.level, episode)
        writer.add_scalar("Pieces_Placed", env.pieces_placed, episode)
        writer.add_scalar("Epsilon", epsilon, episode)
        writer.add_scalar("Loss", avg_loss, episode)

        # Track best performance
        if env.lines_cleared > best_lines or env.score > best_score:
            best_score = max(best_score, env.score)
            best_lines = max(best_lines, env.lines_cleared)
            agent.save(os.path.join(args.checkpoint_dir, "best_model.pt"))

        # Periodic checkpoint
        if episode % args.save_interval == 0:
            agent.save(os.path.join(args.checkpoint_dir, f"model_ep{episode}.pt"))

        # Console logging
        if episode % args.log_interval == 0 or episode == 1:
            elapsed = time.time() - start_time
            print(
                f"Ep {episode:5d}/{args.episodes} | "
                f"Score: {env.score:6d} | "
                f"Lines: {env.lines_cleared:4d} | "
                f"Lvl: {env.level:2d} | "
                f"Pieces: {env.pieces_placed:4d} | "
                f"Eps: {epsilon:.3f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Best Lines: {best_lines} | "
                f"Elapsed: {elapsed:.1f}s",
                flush=True
            )

    # Save final model
    agent.save(os.path.join(args.checkpoint_dir, "final_model.pt"))
    writer.close()
    print("=" * 65)
    print(f" Training complete! Best Lines Cleared: {best_lines}, Best Score: {best_score}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tetris DQN Agent")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--replay-size", type=int, default=30000, help="Experience replay capacity")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden dimension of MLP")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Final exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay factor per episode")
    parser.add_argument("--target-update", type=int, default=10, help="Frequency of target net updates")
    parser.add_argument("--save-interval", type=int, default=500, help="Episodes between checkpoints")
    parser.add_argument("--log-interval", type=int, default=20, help="Episodes between console logs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="runs/tetris_dqn", help="Tensorboard log directory")

    args = parser.parse_args()
    train(args)
