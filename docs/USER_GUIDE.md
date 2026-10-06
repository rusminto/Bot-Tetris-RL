# User Guide

## 1. Install the userscript

1. Install a userscript manager: [Tampermonkey](https://www.tampermonkey.net/) (recommended) or [Violentmonkey](https://violentmonkey.github.io/).
2. Open [`dist/tetris_bot.user.js`](../dist/tetris_bot.user.js) on GitHub and click **Raw**; the manager offers to install it. (Or create a new script and paste the file's contents.)
3. Go to <https://play.tetris.com/>. The HUD appears in the top-right corner with the status **STANDBY**.
4. Press **Press Start** in the HUD (or start the game yourself).

The script also matches `http://localhost:*` and `http://127.0.0.1:*`, so it works on the offline copy of the game too.

## 2. The HUD

```text
+-----------------------------------------------------+
| 🤖 Tetris RL  script v3.5                [PLAYING]  |
+-----------------------------------------------------+
| Score: 1,103,430                  Level: 30         |
| Lines: 302                        Pieces: 765       |
| Action: O -> Rot 0, Col 6                           |
| Fitness/Q: -14.53                                   |
+-----------------------------------------------------+
| RL Engine:        [ 🧠 V5 CEM-RL (13 features) ▾ ]  |
+-----------------------------------------------------+
| [ Pause Bot ]                    [ Press Start ]    |
+-----------------------------------------------------+
| ⚡ Direct Snapping                             (ON) |
+-----------------------------------------------------+
| Key Delay:  [----o---------]  15ms                  |
+-----------------------------------------------------+
| Default. Keeps the well at the left or right wall   |
| and saves I pieces in hold: ~85% Tetrises ...       |
+-----------------------------------------------------+
```

- **Header**: drag to move the HUD.
- **Telemetry**: score, level, lines and pieces read from the game; the last placement (piece, rotation, leftmost column); the policy's score for it.
- **RL Engine**: switch policies at any time; the next piece uses the new one. The text at the bottom of the HUD describes the selected one. Mean scores are from 6 offline games each; see [RL Algorithms](RL_ALGORITHMS.md) for the details.

  | Option | Policy | Offline game |
  | :--- | :--- | :--- |
  | V1 DQN (4 features) | Neural network; finishes the Marathon, mostly with doubles and triples | ~760k |
  | V2 DQN (6 features) | Neural network, not trained for 20G; tops out around Level 17–18 | — |
  | V3 CEM-RL (10 features) | Linear policy trained for Marathon score; about half of its lines are Tetrises | ~830k |
  | V4 CEM-RL (11 features) | Adds a Tetris feature; ~70% Tetrises, with the well in any column | ~970k |
  | **V5 CEM-RL (13 features)** | **Default.** Keeps the well at a wall and saves I pieces in hold; ~85% Tetrises, but tops out a little more often than V4 | **~1.12M** |
- **Pause Bot / Press Start**: pause to play yourself; Press Start sends Enter, then Space.
- **Direct Snapping**: places each piece instantly from the engine's piece-activation hook, using only placements the piece can actually reach. **Keep it on**; it is the only mode that survives 20G (Level 20+).
- **Key Delay**: only used with Direct Snapping off (keystroke mode, 10–60 ms between key presses).

## 3. Offline copy of the game

The game files are in a separate repository, included as the `tetris` submodule (you need access to it):

```bash
git clone --recurse-submodules git@github.com:rusminto/Bot-Tetris-RL.git
# or, in an existing clone:
git submodule update --init
```

Then build the userscript and serve the game with the bot injected:

```bash
python3 userscript/build.py
python3 userscript/serve_offline.py          # http://localhost:8000/  (--port to change)
```

`serve_offline.py` adds the bot to the game page (`game.html`) and to the page around it (`index.html`, which shows the HUD) while serving them, so the game files stay untouched. If you already use Tampermonkey, run it with `--no-bot` so the bot isn't loaded twice.

## 4. Building the userscript

```bash
python3 userscript/build.py
```

It concatenates the modules in `userscript/src/` and embeds:
- the CEM weights in `reinforcement-learning/cem_checkpoints/`: `best_cem_weights.json` (V5), `best_cem_weights_v4.json` (V4) and `best_cem_weights_v3.json` (V3),
- `reinforcement-learning/weights_v2.json` and `weights_v1.json` (DQN),

into `dist/tetris_bot.user.js`. Bump `VERSION` in `userscript/build.py` for every release so userscript managers pick up the update.

To try other CEM weights without replacing the shipped ones, build a candidate elsewhere (`--cem-weights` replaces the V5 engine's weights) and test it with `--bot`:

```bash
python3 userscript/build.py --cem-weights path/to/best_cem_weights.json --output /tmp/candidate.user.js
cd userscript && node tests/e2e.js --bot /tmp/candidate.user.js
```

The older policies are in the HUD's RL Engine list. To play with the survival-trained CEM v2, which isn't, build with `--cem-weights reinforcement-learning/cem_checkpoints/best_cem_weights_v2_survival.json`; it then runs as the V5 option.

## 5. Tests

```bash
python3 reinforcement-learning/src/test_tetris_sim.py   # simulator: features, reachability, scoring, Marathon
node userscript/tests/features_parity.js                # userscript and simulator compute identical features
```

### End-to-end test (headless Firefox)

Plays the offline game with the built userscript and prints progress every 5 seconds, then a summary (level, lines, score, gravity/lock delay per level, the bot's placement statistics):

```bash
cd userscript
npm install
npx playwright install firefox
node tests/e2e.js                              # normal game until the Marathon ends (~3 min)
node tests/e2e.js --force-20g --seconds 180    # 0 ms gravity and 150 ms lock delay from the first piece
node tests/e2e.js --engine cem_v4              # cem_v5 (default), cem_v4, cem_v3, dqn_v2 or dqn_v1
```

The exit code is 0 if the game is still running or reached 300 lines, and 2 if it topped out.

## 6. Training

```bash
cd reinforcement-learning
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt    # CEM only needs numpy; DQN needs torch and tensorboard
```

### CEM (the shipped policy)

```bash
python3 src/test_tetris_sim.py                     # simulator sanity checks, a few seconds
python3 src/cem_train.py --num-workers 12          # v5: Marathon score (~40 min on 12 cores)
python3 src/cem_train.py --objective lines --num-workers 12   # v2: survival (~12 min)
```

- Games run in `src/tetris_sim.py`, which follows the same rules as the browser search (SRS kicks, 20G reachability from Level 20, hold once per piece, guideline top out) and play.tetris.com's scoring.
- `--objective score` (default) plays full Marathons and maximizes the score, minus 500,000 per top-out (`--topout-penalty`), with 13 features, as a fine-tune of a warm start (small `--initial-sigma`, 24 games per candidate). `--objective lines` counts lines before topping out at 20G on a 10-row board without hold, with 8 features. Each objective sets its own defaults for `--features`, `--start-level`, `--board-height`, `--hold`, `--max-lines`, `--max-pieces`, `--generations`, `--games-per-eval`, `--initial-sigma`, `--noise-factor` and `--init-weights`; any of them can be overridden.
- At the end, the final mean and the best candidate are validated on held-out games next to the published Thiery & Scherrer weights, and the better learned one is written to `cem_checkpoints/best_cem_weights.json` (progress goes to `cem_progress.json` every generation). Runs are deterministic for a given `--seed`.
- Compare policies on the same games: `python3 src/evaluate_policies.py --marathon cem_checkpoints/best_cem_weights.json --dqn-v1 --dqn-v2 --games 64`.

Then rebuild the userscript (`python3 userscript/build.py` from the repository root).

### DQN

```bash
python3 src/train.py --episodes 5000                   # checkpoints/ and TensorBoard logs in runs/
python3 src/evaluate.py --episodes 10 --visualize      # terminal playback
python3 src/web_viewer.py --port 5000                  # live browser view of the latest checkpoint
python3 src/export_weights.py                          # checkpoints/best_model.pt -> weights_v2.json
tensorboard --logdir runs
```

### On a remote server

```bash
cp .env.example .env      # set TETRIS_RL_REMOTE=user@your-server, port, remote directory, workers
reinforcement-learning/run_training_remote.sh          # sync the code and start CEM training in tmux
reinforcement-learning/run_training_remote.sh cem --output-dir cem_checkpoints_v4   # extra args go to cem_train.py
reinforcement-learning/run_training_remote.sh dqn      # or DQN
reinforcement-learning/run_training_remote.sh fetch    # copy cem_checkpoints*/, checkpoints/, logs/ back (never overwrites newer local files)
reinforcement-learning/watch_browser.sh                # tunnel the DQN web viewer to localhost:5000
```

The first run creates a virtualenv on the server and installs `requirements.txt`. Variables already set in your environment override `.env`.

## 7. Troubleshooting

**The HUD doesn't appear on play.tetris.com.** Check that the script is enabled in the userscript manager and the site is allowed, then hard-refresh (Ctrl+F5).

**The bot dies at Level 20.** Make sure Direct Snapping is on, and that you have the current build (v3.1 or later). Older builds assumed every column was reachable; from Level 20 the fall speed is 0 ms and the engine drops each piece onto the stack before the bot sees it, so they piled pieces up around the spawn columns. See [The 20G Gravity Breakthrough](THE_20G_BREAKTHROUGH.md#stage-2-why-the-bot-still-died-at-level-20).

**Is the bot placing pieces where it planned?** In the game iframe's console, `window.__tetrisBotStats` counts placements, holds and mismatches; each mismatch also logs `[TetrisRL] Engine disagreed with the SRS model`.

**The game stops at Level 30.** That's the end of the Marathon (300 lines), not a bug.
