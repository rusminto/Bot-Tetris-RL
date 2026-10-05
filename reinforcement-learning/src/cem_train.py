"""
Cross-Entropy Method Reinforcement Learning (CEM-RL) for Tetris.
Based on the seminal work of Thiery & Scherrer (2009) and Szita & Lörincz (2006).

Learns linear policy weights over 8 structural board features using parallel Monte Carlo
rollouts in tetris_sim.py, which models the rules the browser bot faces on play.tetris.com
(SRS kicks, 20G reachability from Level 20, hold once per piece, guideline top out).

Training defaults are deliberately harder than the real game (20G from the first piece, no hold,
10-row board): on the full board with hold, decent policies never top out within any affordable
piece cap, so every candidate scores the same and CEM has nothing to select on. The final
weights are validated on the full 20-row board.
"""

import os
import json
import time
import random
import argparse
import multiprocessing as mp
import numpy as np

from tetris_sim import play_game

FEATURE_NAMES = [
    "landing_height",
    "eroded_piece_cells",
    "row_transitions",
    "col_transitions",
    "holes",
    "cumulative_wells",
    "hole_depth",
    "rows_with_holes"
]
NUM_FEATURES = len(FEATURE_NAMES)

# Thiery & Scherrer (2009) BCTS weights, same feature order (used as a validation baseline)
BCTS_WEIGHTS = [-12.63, 6.60, -9.22, -19.77, -13.08, -10.49, -1.61, -24.04]


def play_worker(args):
    weights, seed, max_pieces, env = args
    lines, _, topped_out = play_game(weights, seed=seed, max_pieces=max_pieces, **env)
    return lines, topped_out


def evaluate_population(pool, population, seeds, max_pieces, env):
    """Mean lines per candidate over the same games, and how many candidates never topped out."""
    tasks = [(list(map(float, w)), s, max_pieces, env) for w in population for s in seeds]
    results = pool.map(play_worker, tasks, chunksize=1)
    n = len(seeds)
    scores = np.array([np.mean([r[0] for r in results[i * n:(i + 1) * n]]) for i in range(len(population))])
    survivors = sum(1 for i in range(len(population)) if not any(r[1] for r in results[i * n:(i + 1) * n]))
    return scores, survivors


def normalized(w):
    w = np.asarray(w, dtype=np.float64)
    return w / np.linalg.norm(w)


def train_cem(args):
    os.makedirs(args.output_dir, exist_ok=True)
    num_cpus = args.num_workers or mp.cpu_count()
    train_env = {
        "start_level": args.start_level,
        "visible_height": args.board_height,
        "use_hold": args.hold,
    }
    print("=" * 65)
    print(" 🧠 Cross-Entropy Method (CEM-RL) Tetris Policy Search")
    print(f" Workers:     {num_cpus} parallel processes")
    print(f" Population:  {args.population_size}")
    print(f" Elite Size:  {args.elite_size}")
    print(f" Generations: {args.generations}")
    print(f" Games/eval:  {args.games_per_eval} (max {args.max_pieces} pieces each)")
    print(f" Environment: start level {args.start_level}, {args.board_height} rows, hold {'on' if args.hold else 'off'}")
    print(f" Output Dir:  {args.output_dir}")
    print("=" * 65)

    # Initialize with sensible domain signs:
    # landing_height(-), eroded(+), row_trans(-), col_trans(-), holes(-), wells(-), hole_depth(-), rows_with_holes(-)
    mu = normalized([-5.0, 5.0, -5.0, -10.0, -10.0, -5.0, -2.0, -15.0])
    sigma = np.ones(NUM_FEATURES) * args.initial_sigma

    best_global_score = -1.0
    best_global_weights = mu.copy()
    seed_rng = random.Random(args.seed)
    np.random.seed(args.seed)
    start_time = time.time()

    with mp.Pool(processes=num_cpus) as pool:
        for gen in range(1, args.generations + 1):
            gen_start = time.time()

            # 1. Sample candidate population (direction defines the policy, so normalize)
            population = [normalized(np.random.normal(mu, sigma)) for _ in range(args.population_size)]
            population[0] = mu.copy()  # always re-evaluate the current mean

            # 2. Evaluate every candidate on the same games (common random numbers)
            seeds = [seed_rng.randint(1, 10**9) for _ in range(args.games_per_eval)]
            scores, survivors = evaluate_population(pool, population, seeds, args.max_pieces, train_env)

            # 3. Select elite candidates
            elite_indices = np.argsort(scores)[::-1][:args.elite_size]
            elite_samples = np.array([population[i] for i in elite_indices])
            elite_scores = scores[elite_indices]

            # 4. Update distribution (noisy CEM update)
            new_mu = normalized(np.mean(elite_samples, axis=0))
            new_sigma = np.std(elite_samples, axis=0)
            noise = max(0.01, args.noise_factor / (gen + 1.0))
            mu = normalized(args.alpha * mu + (1.0 - args.alpha) * new_mu)
            sigma = args.alpha * sigma + (1.0 - args.alpha) * new_sigma + noise

            if elite_scores[0] > best_global_score:
                best_global_score = float(elite_scores[0])
                best_global_weights = elite_samples[0].copy()

            print(
                f"Gen {gen:3d}/{args.generations} | "
                f"Best: {elite_scores[0]:7.1f} lines | "
                f"Elite Avg: {np.mean(elite_scores):7.1f} | "
                f"Mean policy: {scores[0]:7.1f} | "
                f"Pop median: {np.median(scores):7.1f} | "
                f"Never topped out: {survivors}/{len(scores)} | "
                f"Time: {time.time() - gen_start:5.1f}s | "
                f"Total: {(time.time() - start_time) / 60:5.1f}m",
                flush=True
            )
            save_checkpoint(args.output_dir, "cem_progress.json", gen, best_global_score, mu, sigma, best_global_weights)

        # 5. Validate on fresh games: the final mean vs the best single candidate, on the training
        #    environment and on the real 20-row board at 20G
        print("=" * 65)
        print(" Validating on held-out games...")
        val_seeds = [seed_rng.randint(1, 10**9) for _ in range(args.validation_games)]
        candidates = {"final_mean": mu, "best_candidate": best_global_weights, "bcts_baseline": normalized(BCTS_WEIGHTS)}
        full_env = {"start_level": 20, "visible_height": 20, "use_hold": False}
        results = {}
        for name, w in candidates.items():
            train_score = evaluate_population(pool, [w], val_seeds, args.max_pieces, train_env)[0][0]
            full_score = evaluate_population(pool, [w], val_seeds[:max(2, len(val_seeds) // 2)], args.validation_max_pieces, full_env)[0][0]
            results[name] = (train_score, full_score)
            print(f"   {name:15s}: training env {train_score:8.1f} lines | 20 rows @20G no hold {full_score:8.1f} lines")

    learned = {k: v for k, v in results.items() if k != "bcts_baseline"}
    chosen = max(learned, key=lambda k: learned[k])
    chosen_weights = candidates[chosen]
    print("=" * 65)
    print(f" 🎉 CEM-RL Training Finished! Selected: {chosen}")
    for name, w in zip(FEATURE_NAMES, chosen_weights):
        print(f"   {name:20s}: {w:+.6f}")
    path = save_checkpoint(args.output_dir, "best_cem_weights.json", args.generations, results[chosen][0], mu, sigma,
                           chosen_weights, validation={k: {"training_env_lines": v[0], "full_board_lines": v[1]}
                                                       for k, v in results.items()},
                           environment=train_env)
    print(f" Saved to: {path}")
    print("=" * 65)


def save_checkpoint(output_dir, filename, gen, score, mu, sigma, weights, **extra):
    checkpoint = {
        "generation": gen,
        "best_score": float(score),
        "weights": {FEATURE_NAMES[i]: float(weights[i]) for i in range(NUM_FEATURES)},
        "feature_names": FEATURE_NAMES,
        "weights_list": [float(w) for w in weights],
        "mean": [float(v) for v in mu],
        "sigma": [float(v) for v in sigma],
    }
    checkpoint.update(extra)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Entropy Method RL for Tetris")
    parser.add_argument("--population-size", type=int, default=32, help="Number of candidate policies per generation")
    parser.add_argument("--elite-size", type=int, default=6, help="Number of elite candidates to select")
    parser.add_argument("--generations", type=int, default=30, help="Number of CEM generations")
    parser.add_argument("--games-per-eval", type=int, default=12, help="Number of games to average per candidate")
    parser.add_argument("--max-pieces", type=int, default=3000, help="Cap pieces per training game")
    parser.add_argument("--start-level", type=int, default=20, help="Starting level (20+ means 20G gravity)")
    parser.add_argument("--board-height", type=int, default=10, help="Visible rows during training (real game: 20)")
    parser.add_argument("--hold", action="store_true", help="Allow hold during training (makes games much longer)")
    parser.add_argument("--validation-games", type=int, default=24, help="Held-out games for the final comparison")
    parser.add_argument("--validation-max-pieces", type=int, default=5000, help="Piece cap for full-board validation")
    parser.add_argument("--alpha", type=float, default=0.2, help="Smoothing parameter for distribution update")
    parser.add_argument("--initial-sigma", type=float, default=0.5, help="Initial standard deviation")
    parser.add_argument("--noise-factor", type=float, default=0.2, help="Exploration noise factor")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling and game seeds")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cem_checkpoints")),
                        help="Output directory (default: reinforcement-learning/cem_checkpoints)")

    args = parser.parse_args()
    train_cem(args)
