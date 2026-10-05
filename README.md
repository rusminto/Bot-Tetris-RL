# Bot-Tetris-RL

A userscript bot that plays the official [play.tetris.com](https://play.tetris.com/) Marathon with a policy learned by reinforcement learning. It reads the game engine's state directly, plans only the placements the piece can really reach (including at 20G, where pieces drop instantly), and plays all the way to the end of the Marathon at **Level 30**.

![The bot finishing the Marathon on play.tetris.com: Level 30, 304 lines, 600,230 points](screenshots/marathon-level-30.png)

*The bot finishing the Marathon on play.tetris.com: Level 30, 304 lines, 600,230 points.*

## Results

| Run | Result |
| :--- | :--- |
| play.tetris.com, CEM policy | Marathon completed: Level 30, 304 lines, 600,230 points |
| Offline copy of the game, headless Firefox, CEM policy (2 games) | Marathon completed both times (301 and 302 lines, stack ≤ 2 rows at the end) |
| Same, with 0 ms gravity forced from the first piece | Marathon completed (300 lines) |
| Previous build (before the reachable placement search) | Topped out at Level 20–22 |

How the Level 20 wall was found and fixed: [The 20G Gravity Breakthrough](docs/THE_20G_BREAKTHROUGH.md).

## How it works

1. **Hook the engine, not the screen.** The userscript wraps SystemJS module registration to get the game's `Player` and `Model` objects, so it reads the board, the live piece, hold and queue directly and drives the piece through the engine's own control actions.
2. **Search reachable placements.** When a piece activates, the bot runs a breadth-first search over left/right moves and SRS rotations with wall kicks. From Level 20 the fall speed is 0 ms, so the piece drops onto the stack after every action; the search models that, so the bot never plans a column the piece can't get to.
3. **Score them with a learned policy.** The default policy is linear over 8 classic board features (Thiery & Scherrer), trained with the **Cross-Entropy Method** in [`tetris_sim.py`](reinforcement-learning/src/tetris_sim.py), a simulator with the same rules as the browser search. Two DQN networks are available as alternatives.
4. **Execute and verify.** Actions are applied one at a time; after each one the bot checks where the engine put the piece and re-plans if it differs, then hard-drops.

Details: [Architecture](docs/ARCHITECTURE.md) · [RL algorithms & benchmarks](docs/RL_ALGORITHMS.md).

## Quick start

### Play on play.tetris.com

1. Install [Tampermonkey](https://www.tampermonkey.net/) or [Violentmonkey](https://violentmonkey.github.io/).
2. Open [`dist/tetris_bot.user.js`](dist/tetris_bot.user.js), click **Raw**, and install it (or paste it into a new userscript).
3. Go to <https://play.tetris.com/>, keep **Direct Snapping** on in the HUD, and press **Press Start**.

### Run the offline copy of the game

The game files are not part of this repository; they live in a separate repository, included as the `tetris` submodule (cloning it needs access to that repository).

```bash
git clone --recurse-submodules git@github.com:rusminto/Bot-Tetris-RL.git
cd Bot-Tetris-RL
python3 userscript/build.py            # -> dist/tetris_bot.user.js
python3 userscript/serve_offline.py    # http://localhost:8000/, bot injected
```

### Train the policy

```bash
cd reinforcement-learning
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 src/test_tetris_sim.py
python3 src/cem_train.py --num-workers 12    # ~12 min on 12 cores, writes cem_checkpoints/best_cem_weights.json
cd .. && python3 userscript/build.py           # embed the new weights
```

To train on a remote server, copy `.env.example` to `.env`, fill in the server, and run `reinforcement-learning/run_training_remote.sh`. See the [User Guide](docs/USER_GUIDE.md) for DQN training, the end-to-end test and troubleshooting.

## Repository layout

```
├── dist/tetris_bot.user.js        built userscript (install this)
├── userscript/
│   ├── src/                       userscript modules (placement search, hooks, policies, HUD, ...)
│   ├── build.py                   concatenates src/ and embeds the weights
│   ├── serve_offline.py           serves the offline game with the bot injected
│   └── tests/e2e.js               headless Firefox end-to-end test (Playwright)
├── reinforcement-learning/
│   ├── src/                       simulators, CEM and DQN training, evaluation
│   ├── cem_checkpoints/           CEM weights used by the userscript (+ v1 legacy)
│   ├── checkpoints*/              DQN checkpoints, weights_v1.json / weights_v2.json exports
│   ├── logs/                      training logs
│   └── run_training_remote.sh     sync + train on a remote server (configured by .env)
├── docs/                          architecture, algorithms, the 20G write-up, user guide
├── screenshots/                   screenshots and recordings
└── tetris/                        submodule: offline copy of the game (separate repository)
```

## Documentation

| Document | Contents |
| :--- | :--- |
| [User Guide](docs/USER_GUIDE.md) | Installing, the HUD, offline game, building, testing, training, troubleshooting |
| [Architecture](docs/ARCHITECTURE.md) | Engine hooks, engine facts (actions, gravity, lock delay), placement search, simulators, build |
| [RL Algorithms & Benchmarks](docs/RL_ALGORITHMS.md) | CEM and DQN, features, training setup, measured results |
| [The 20G Gravity Breakthrough](docs/THE_20G_BREAKTHROUGH.md) | Why the bot died at Level 20 and how it was fixed |

## Disclaimer

Tetris® is a registered trademark of The Tetris Company. This is a personal research project, not affiliated with or endorsed by The Tetris Company or Blue Planet Software, and it does not include their game files.
