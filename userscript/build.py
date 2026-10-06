"""
Builds the Tampermonkey/Violentmonkey userscript: concatenates the modules in userscript/src/
and embeds the trained policy weights.

    python3 userscript/build.py            # -> dist/tetris_bot.user.js
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "userscript" / "src"
RL = ROOT / "reinforcement-learning"

VERSION = "3.2"  # bump on every release so userscript managers pick up the update

# Concatenated in this order inside one IIFE (header.js opens it, hud.js closes it)
MODULES = [
    "header.js",             # userscript metadata, init guard
    "policies.js",           # weights, piece shapes, CEM + DQN evaluation, straight-drop planners
    "placement_search.js",   # SRS/20G model of the BPS engine and reachable placement search
    "game_hooks.js",         # SystemJS/Cocos hooks, live stats, player/model lookup
    "keystrokes.js",         # synthetic key input (keystroke mode)
    "instant_placement.js",  # synchronous placement from the piece-activation hook
    "bot_loop.js",           # 20 ms polling loop: fallback + keystroke mode
    "messaging.js",          # iframe <-> top window bridge
    "hud.js",                # floating HUD
]

# Thiery & Scherrer (2009) BCTS weights, used if no trained CEM checkpoint exists
BCTS_WEIGHTS = [-12.63, 6.60, -9.22, -19.77, -13.08, -10.49, -1.61, -24.04]


def load_cem_weights(path):
    raw = json.loads(path.read_text())["weights_list"] if path.exists() else BCTS_WEIGHTS
    norm = sum(w * w for w in raw) ** 0.5
    return [round(w / norm, 6) for w in raw] if norm > 1e-6 else raw


def build(output, cem_weights):
    tokens = {
        "__CEM_WEIGHTS__": json.dumps(load_cem_weights(cem_weights)),
        "__DQN_V1_WEIGHTS__": json.dumps(json.loads((RL / "weights_v1.json").read_text())),
        "__DQN_V2_WEIGHTS__": json.dumps(json.loads((RL / "weights_v2.json").read_text())),
    }
    script = "".join((SRC / name).read_text() for name in MODULES).replace("__VERSION__", VERSION)
    for token, value in tokens.items():
        if script.count(token) != 1:
            raise ValueError(f"expected exactly one {token} in userscript/src")
        script = script.replace(token, value)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script)
    print(f"Generated {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Tetris RL userscript")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "tetris_bot.user.js")
    parser.add_argument("--cem-weights", type=Path, default=RL / "cem_checkpoints" / "best_cem_weights.json",
                        help="CEM weights to embed (e.g. a candidate from a new training run)")
    args = parser.parse_args()
    build(args.output.resolve(), args.cem_weights.resolve())
