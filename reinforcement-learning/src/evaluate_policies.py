"""
Compares policies in tetris_sim.py on the same games.

    # play.tetris.com's Marathon (Level 1 -> 300 lines, hold on): score, survival, clear types
    python3 src/evaluate_policies.py --marathon cem_checkpoints/best_cem_weights.json --dqn-v1 --dqn-v2 --bcts

    # the CEM training setting (20G, 10 rows, no hold)
    python3 src/evaluate_policies.py cem_checkpoints/best_cem_weights.json --bcts
"""

import os
import json
import argparse
import multiprocessing as mp

from tetris_sim import MARATHON_LINES, LinearPolicy, MLPPolicy, play_game

RL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BCTS_WEIGHTS = [-12.63, 6.60, -9.22, -19.77, -13.08, -10.49, -1.61, -24.04]


def load_policy(spec):
    kind, arg = spec
    if kind == "linear":
        return LinearPolicy(arg)
    return MLPPolicy(json.load(open(os.path.join(RL_DIR, f"weights_v{arg}.json"))), version=arg)


def run(task):
    name, spec, seed, env = task
    r = play_game(load_policy(spec), seed, **env)
    return name, r.lines, r.score, r.topped_out, r.clears


def main():
    parser = argparse.ArgumentParser(description="Evaluate policies in tetris_sim.py")
    parser.add_argument("weights", nargs="*", help="CEM weight files (best_cem_weights.json format)")
    parser.add_argument("--bcts", action="store_true", help="Published Thiery & Scherrer weights")
    parser.add_argument("--dqn-v1", action="store_true", help="weights_v1.json")
    parser.add_argument("--dqn-v2", action="store_true", help="weights_v2.json")
    parser.add_argument("--marathon", action="store_true",
                        help="Marathon: start level 1, 20 rows, hold, ends at 300 lines (overrides the options below)")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--max-pieces", type=int, default=3000)
    parser.add_argument("--start-level", type=int, default=20, help="20+ means 20G gravity")
    parser.add_argument("--board-height", type=int, default=10)
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--seed", type=int, default=1_000_000, help="First game seed (games use seed..seed+games-1)")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    # label weight files by folder/file so cem_checkpoints*/best_cem_weights.json don't collide
    policies = {os.path.join(os.path.basename(os.path.dirname(os.path.abspath(p))), os.path.basename(p)):
                ("linear", json.load(open(p))["weights_list"]) for p in args.weights}
    if len(policies) != len(args.weights):
        parser.error("two weight files have the same folder/file name")
    if args.bcts:
        policies["bcts"] = ("linear", BCTS_WEIGHTS)
    if args.dqn_v1:
        policies["dqn_v1"] = ("mlp", 1)
    if args.dqn_v2:
        policies["dqn_v2"] = ("mlp", 2)
    if not policies:
        parser.error("give at least one weights file or --bcts / --dqn-v1 / --dqn-v2")

    if args.marathon:
        env = dict(max_pieces=2000, start_level=1, visible_height=20, use_hold=True, max_lines=MARATHON_LINES)
    else:
        env = dict(max_pieces=args.max_pieces, start_level=args.start_level, visible_height=args.board_height,
                   use_hold=args.hold)

    tasks = [(name, spec, args.seed + g, env) for name, spec in policies.items() for g in range(args.games)]
    with mp.Pool(args.num_workers) as pool:
        results = pool.map(run, tasks, chunksize=1)

    print(f"{args.games} games: start level {env['start_level']}, {env['visible_height']} rows, "
          f"hold {'on' if env['use_hold'] else 'off'}, "
          + (f"ends at {env['max_lines']} lines" if env.get("max_lines") else f"cap {env['max_pieces']} pieces"))
    for name in policies:
        rs = [r for r in results if r[0] == name]
        n = len(rs)
        lines = sorted(r[1] for r in rs)
        scores = sorted(r[2] for r in rs)
        topped = sum(r[3] for r in rs)
        clears = [sum(r[4][k] for r in rs) / n for k in range(1, 5)]
        mean = sum(scores) / n
        sd = (sum((x - mean) ** 2 for x in scores) / (n - 1)) ** 0.5 if n > 1 else 0.0
        print(f"  {name:36s} score mean {mean:9,.0f} ± {sd:7,.0f} (sd) median {scores[n // 2]:9,} | "
              f"lines mean {sum(lines) / n:7.1f} | topped out {topped}/{n} | "
              f"per game: {clears[0]:.0f} singles, {clears[1]:.0f} doubles, {clears[2]:.1f} triples, {clears[3]:.1f} tetrises")


if __name__ == "__main__":
    main()
