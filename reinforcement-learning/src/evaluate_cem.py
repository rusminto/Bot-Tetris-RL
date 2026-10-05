"""
Compares CEM weight files in tetris_sim.py on the same games.

    python3 src/evaluate_cem.py cem_checkpoints/best_cem_weights.json cem_checkpoints/best_cem_weights_v1_legacy.json --bcts
    python3 src/evaluate_cem.py cem_checkpoints/best_cem_weights.json --board-height 20 --max-pieces 5000 --games 16
"""

import os
import json
import argparse
import multiprocessing as mp

from tetris_sim import play_game

BCTS_WEIGHTS = [-12.63, 6.60, -9.22, -19.77, -13.08, -10.49, -1.61, -24.04]


def run(task):
    name, weights, seed, args = task
    lines, pieces, topped_out = play_game(weights, seed, args.max_pieces, start_level=args.start_level,
                                          visible_height=args.board_height, use_hold=args.hold)
    return name, lines, topped_out


def main():
    parser = argparse.ArgumentParser(description="Evaluate CEM policy weights in tetris_sim.py")
    parser.add_argument("weights", nargs="*", help="best_cem_weights.json-style files")
    parser.add_argument("--bcts", action="store_true", help="Also evaluate the published Thiery & Scherrer weights")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--max-pieces", type=int, default=3000)
    parser.add_argument("--start-level", type=int, default=20, help="20+ means 20G gravity")
    parser.add_argument("--board-height", type=int, default=10)
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--seed", type=int, default=1_000_000, help="First game seed (games use seed..seed+games-1)")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    policies = {os.path.basename(p): json.load(open(p))["weights_list"] for p in args.weights}
    if args.bcts:
        policies["bcts"] = BCTS_WEIGHTS
    if not policies:
        parser.error("give at least one weights file or --bcts")

    tasks = [(name, w, args.seed + g, args) for name, w in policies.items() for g in range(args.games)]
    with mp.Pool(args.num_workers) as pool:
        results = pool.map(run, tasks, chunksize=1)

    print(f"{args.games} games, start level {args.start_level}, {args.board_height} rows, "
          f"hold {'on' if args.hold else 'off'}, cap {args.max_pieces} pieces")
    for name in policies:
        lines = sorted(l for n, l, _ in results if n == name)
        topped = sum(t for n, _, t in results if n == name)
        print(f"  {name:36s} mean {sum(lines) / len(lines):8.1f} lines | median {lines[len(lines) // 2]:6d} | "
              f"topped out {topped}/{len(lines)}")


if __name__ == "__main__":
    main()
