"""
Cross-Entropy Method Reinforcement Learning (CEM-RL) for Tetris.
Based on the seminal work of Thiery & Scherrer (2009) and Szita & Lörincz (2006).

Learns linear policy weights over board features using parallel Monte Carlo rollouts in
tetris_sim.py, which models the rules the browser bot faces on play.tetris.com (SRS kicks, 20G
reachability from Level 20, hold once per piece, guideline top out, the game's scoring).

Two objectives:
- score (default): play.tetris.com's Marathon (Level 1 -> 300 lines, hold on, 20 rows) and maximize the
  final score, with a large penalty for topping out. Uses 11 features: the 8 Thiery & Scherrer ones plus
  the base points of the clear, how many lines a vertical I would clear (so the policy can plan
  Tetrises) and whether the placement clears a Tetris.
- lines (the v2 policy): survival only. The Marathon is too easy to tell survival policies apart, so it
  trains at 20G from the first piece on a 10-row board without hold and counts lines cleared.
"""

import os
import json
import time
import random
import argparse
import multiprocessing as mp
import numpy as np

from tetris_sim import MARATHON_LINES, play_game

FEATURE_NAMES = [
    "landing_height",
    "eroded_piece_cells",
    "row_transitions",
    "col_transitions",
    "holes",
    "cumulative_wells",
    "hole_depth",
    "rows_with_holes",
    "clear_points",   # base points of the clear / 100: 0, 1, 3, 5, 8
    "ready_lines",    # lines a vertical I would clear in the lowest column (0-4)
    "tetris",         # 8 if the placement clears a Tetris, else 0
]

# Thiery & Scherrer (2009) BCTS weights, first 8 features (used as a validation baseline)
BCTS_WEIGHTS = [-12.63, 6.60, -9.22, -19.77, -13.08, -10.49, -1.61, -24.04]
# Hand-set starting point (the lines objective starts here): landing_height(-), eroded(+), row_trans(-),
# col_trans(-), holes(-), wells(-), hole_depth(-), rows_with_holes(-)
LINES_INIT = [-5.0, 5.0, -5.0, -10.0, -10.0, -5.0, -2.0, -15.0]
# The score objective warm-starts from the v3 weights with ready_lines raised to 0.5 and the Tetris feature at
# 0.5: neither change alone helps, together they scored 884k against v3's 848k (64 simulated Marathons). v4 was
# trained from here with the defaults. v3 itself started from the v2 survival weights plus 0.3 on clear_points and ready_lines; starting from the
# hand-set signs instead only reached ~726k points instead of ~836k
SCORE_INIT = [-0.522041, -0.20335, -0.179347, -0.498333, -0.361269, -0.015675, -0.055044, -0.410601, -0.001968,
              0.5, 0.5]

PRESETS = {
    "score": dict(features=11, start_level=1, board_height=20, hold=True, max_lines=MARATHON_LINES, max_pieces=1500,
                  generations=40, init_weights=SCORE_INIT),
    "lines": dict(features=8, start_level=20, board_height=10, hold=False, max_lines=None, max_pieces=3000,
                  generations=30, init_weights=LINES_INIT),
}


def play_worker(args):
    weights, seed, env = args
    r = play_game(weights, seed=seed, **env)
    return r.lines, r.score, r.topped_out


def evaluate(pool, population, seeds, env, objective, topout_penalty):
    """Per candidate: fitness, mean lines, mean score, games topped out (all on the same games)."""
    tasks = [(list(map(float, w)), s, env) for w in population for s in seeds]
    results = pool.map(play_worker, tasks, chunksize=1)
    n = len(seeds)
    out = []
    for i in range(len(population)):
        games = results[i * n:(i + 1) * n]
        lines = float(np.mean([g[0] for g in games]))
        score = float(np.mean([g[1] for g in games]))
        topped = sum(g[2] for g in games)
        fitness = lines if objective == "lines" else score - topout_penalty * topped / n
        out.append((fitness, lines, score, topped))
    return out


def normalized(w):
    w = np.asarray(w, dtype=np.float64)
    return w / np.linalg.norm(w)


def train_cem(args):
    os.makedirs(args.output_dir, exist_ok=True)
    num_cpus = args.num_workers or mp.cpu_count()
    names = FEATURE_NAMES[:args.features]
    env = dict(max_pieces=args.max_pieces, start_level=args.start_level, visible_height=args.board_height,
               use_hold=args.hold, max_lines=args.max_lines)
    unit = "lines" if args.objective == "lines" else "points"
    print("=" * 65)
    print(" 🧠 Cross-Entropy Method (CEM-RL) Tetris Policy Search")
    print(f" Objective:   {args.objective}" + (f" (top-out penalty {args.topout_penalty:,.0f})" if args.objective == "score" else ""))
    print(f" Features:    {len(names)}")
    print(f" Workers:     {num_cpus} parallel processes")
    print(f" Population:  {args.population_size}, elite {args.elite_size}, {args.generations} generations")
    print(f" Games/eval:  {args.games_per_eval} (max {args.max_pieces} pieces each)")
    print(f" Environment: start level {args.start_level}, {args.board_height} rows, hold {'on' if args.hold else 'off'}"
          + (f", ends at {args.max_lines} lines" if args.max_lines else ""))
    print(f" Output Dir:  {args.output_dir}")
    print("=" * 65)

    init = args.init_weights
    if isinstance(init, str):
        init = [float(v) for v in init.split(",")]
    if len(init) < len(names):
        raise SystemExit(f"--init-weights needs {len(names)} values")
    mu = normalized(init[:len(names)])
    sigma = np.ones(len(names)) * args.initial_sigma

    best_fitness = -np.inf
    best_weights = mu.copy()
    seed_rng = random.Random(args.seed)
    np.random.seed(args.seed)
    start_time = time.time()

    with mp.Pool(processes=num_cpus) as pool:
        for gen in range(1, args.generations + 1):
            gen_start = time.time()

            # 1. Sample candidates (only the direction of w matters, so normalize); slot 0 is the mean
            population = [normalized(np.random.normal(mu, sigma)) for _ in range(args.population_size)]
            population[0] = mu.copy()

            # 2. Every candidate plays the same games (common random numbers)
            seeds = [seed_rng.randint(1, 10**9) for _ in range(args.games_per_eval)]
            stats = evaluate(pool, population, seeds, env, args.objective, args.topout_penalty)
            fitness = np.array([s[0] for s in stats])

            # 3. Elites and the smoothed, noisy CEM update
            elite_idx = np.argsort(fitness)[::-1][:args.elite_size]
            elites = np.array([population[i] for i in elite_idx])
            new_mu = normalized(np.mean(elites, axis=0))
            noise = max(0.01, args.noise_factor / (gen + 1.0))
            mu = normalized(args.alpha * mu + (1.0 - args.alpha) * new_mu)
            sigma = args.alpha * sigma + (1.0 - args.alpha) * np.std(elites, axis=0) + noise

            if fitness[elite_idx[0]] > best_fitness:
                best_fitness = float(fitness[elite_idx[0]])
                best_weights = elites[0].copy()

            never_topped = sum(1 for s in stats if s[3] == 0)
            print(
                f"Gen {gen:3d}/{args.generations} | "
                f"Best: {fitness[elite_idx[0]]:10,.1f} | "
                f"Elite avg: {np.mean(fitness[elite_idx]):10,.1f} | "
                f"Mean policy: {stats[0][0]:10,.1f} {unit} ({stats[0][1]:.0f} lines, {stats[0][3]} top-outs) | "
                f"Pop median: {np.median(fitness):10,.1f} | "
                f"Never topped out: {never_topped}/{len(stats)} | "
                f"{time.time() - gen_start:5.1f}s | total {(time.time() - start_time) / 60:5.1f}m",
                flush=True
            )
            save_checkpoint(args.output_dir, "cem_progress.json", names, gen, best_fitness, mu, sigma, best_weights)

        # 4. Validate on held-out games: the final mean and the best single candidate, next to BCTS
        print("=" * 65)
        print(f" Validating on {args.validation_games} held-out games...")
        val_seeds = [seed_rng.randint(1, 10**9) for _ in range(args.validation_games)]
        bcts = normalized(BCTS_WEIGHTS + [0.0] * (len(names) - 8))
        candidates = {"final_mean": mu, "best_candidate": best_weights, "bcts_baseline": bcts}
        results = {}
        for name, w in candidates.items():
            fit, lines, score, topped = evaluate(pool, [w], val_seeds, env, args.objective, args.topout_penalty)[0]
            results[name] = {"fitness": fit, "mean_lines": lines, "mean_score": score, "topped_out": topped}
            print(f"   {name:15s}: fitness {fit:12,.1f} | {lines:7.1f} lines | score {score:11,.0f} | "
                  f"topped out {topped}/{len(val_seeds)}")

    chosen = max(("final_mean", "best_candidate"), key=lambda k: results[k]["fitness"])
    chosen_weights = candidates[chosen]
    print("=" * 65)
    print(f" 🎉 CEM-RL Training Finished! Selected: {chosen}")
    for name, w in zip(names, chosen_weights):
        print(f"   {name:20s}: {w:+.6f}")
    path = save_checkpoint(args.output_dir, "best_cem_weights.json", names, args.generations,
                           results[chosen]["fitness"], mu, sigma, chosen_weights,
                           objective=args.objective, validation=results, environment=env)
    print(f" Saved to: {path}")
    print("=" * 65)


def save_checkpoint(output_dir, filename, names, gen, score, mu, sigma, weights, **extra):
    checkpoint = {
        "generation": gen,
        "best_score": float(score),
        "weights": {names[i]: float(weights[i]) for i in range(len(names))},
        "feature_names": names,
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
    parser.add_argument("--objective", choices=sorted(PRESETS), default="score",
                        help="score: Marathon score (11 features); lines: survival at 20G on a short board (8 features). "
                             "Sets the defaults of the environment options below.")
    parser.add_argument("--features", type=int, choices=(8, 10, 11), help="Number of features")
    parser.add_argument("--start-level", type=int, help="Starting level (20+ means 20G gravity)")
    parser.add_argument("--board-height", type=int, help="Visible rows (real game: 20)")
    parser.add_argument("--hold", action=argparse.BooleanOptionalAction, default=None, help="Allow hold")
    parser.add_argument("--max-lines", type=int, help="End the game at this many lines (Marathon: 300)")
    parser.add_argument("--max-pieces", type=int, help="Cap pieces per game")
    parser.add_argument("--topout-penalty", type=float, default=500_000, help="Score objective: points lost per top out")
    parser.add_argument("--init-weights", type=str, help="Comma-separated starting mean (default: per objective)")
    parser.add_argument("--population-size", type=int, default=32, help="Number of candidate policies per generation")
    parser.add_argument("--elite-size", type=int, default=6, help="Number of elite candidates to select")
    parser.add_argument("--generations", type=int, help="Number of CEM generations (score: 40, lines: 30)")
    parser.add_argument("--games-per-eval", type=int, default=12, help="Number of games to average per candidate")
    parser.add_argument("--validation-games", type=int, default=24, help="Held-out games for the final comparison")
    parser.add_argument("--alpha", type=float, default=0.2, help="Smoothing parameter for distribution update")
    parser.add_argument("--initial-sigma", type=float, default=0.5, help="Initial standard deviation")
    parser.add_argument("--noise-factor", type=float, default=0.2, help="Exploration noise factor")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling and game seeds")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cem_checkpoints")),
                        help="Output directory (default: reinforcement-learning/cem_checkpoints)")

    args = parser.parse_args()
    for key, value in PRESETS[args.objective].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    train_cem(args)
