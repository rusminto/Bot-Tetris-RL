# reinforcement-learning

Training and evaluation for the bot's policies. The userscript embeds the CEM weights from `cem_checkpoints/best_cem_weights.json` and the DQN weights from `weights_v2.json` / `weights_v1.json` (see `userscript/build.py` in the repository root).

```
├── src/
│   ├── tetris_sim.py         simulator: SRS kicks, 20G reachability, hold/top out, scoring, Marathon (bitboards)
│   ├── test_tetris_sim.py    sanity checks for tetris_sim.py
│   ├── cem_train.py          Cross-Entropy Method policy search (the shipped policy)
│   ├── evaluate_policies.py  compare CEM weights and DQN networks on the same simulated games
│   ├── tetris_engine.py      simpler engine used for DQN
│   ├── dqn_agent.py          after-state value network, replay buffer, target network
│   ├── train.py              DQN v2 training with TensorBoard logging
│   ├── evaluate.py           play a DQN checkpoint, optional terminal animation
│   ├── web_viewer.py         live browser view of a DQN checkpoint (SSE)
│   └── export_weights.py     DQN checkpoint -> JSON for the userscript
├── cem_checkpoints/          best_cem_weights.json (v4, shipped), best_cem_weights_v3.json, best_cem_weights_v2_survival.json, best_cem_weights_v1_legacy.json, cem_progress.json (v4 run)
├── checkpoints/              DQN v2 checkpoints (best_model.pt, final_model.pt; per-500-episode ones are git-ignored)
├── checkpoints_v1_legacy/    DQN v1 checkpoints
├── weights_v1.json, weights_v2.json   exported DQN weights embedded in the userscript
├── logs/                     cem_v1_train.log … cem_v4_train.log, dqn_v2_train.log
├── run_training_remote.sh    sync + train on a remote server in tmux (configured by ../.env)
└── watch_browser.sh          start and tunnel web_viewer.py from the remote server
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt    # CEM only needs numpy; DQN needs torch and tensorboard
```

## CEM

```bash
python3 src/test_tetris_sim.py
python3 src/cem_train.py --num-workers 12                   # v4: Marathon score, ~35 min on 12 cores
python3 src/cem_train.py --objective lines --num-workers 12 # v2: survival at 20G, ~12 min
python3 src/evaluate_policies.py --marathon cem_checkpoints/best_cem_weights.json --dqn-v1 --games 64
```

See [RL Algorithms](../docs/RL_ALGORITHMS.md#d-objectives-and-training-settings) for the two objectives.

## DQN

```bash
python3 src/train.py --episodes 5000
python3 src/evaluate.py --episodes 10 --visualize
python3 src/web_viewer.py --port 5000
python3 src/export_weights.py                             # -> weights_v2.json
tensorboard --logdir runs
```

## Remote server

Copy `../.env.example` to `../.env` and set `TETRIS_RL_REMOTE=user@your-server` (plus port, remote directory and worker count), then:

```bash
./run_training_remote.sh          # CEM in tmux session cem-train
./run_training_remote.sh dqn      # DQN in tmux session dqn-train
./run_training_remote.sh cem --output-dir cem_checkpoints_v4   # extra arguments go to cem_train.py
./run_training_remote.sh fetch    # copy cem_checkpoints*/, checkpoints/ and logs/ back (keeps newer local files)
./watch_browser.sh                # web viewer on http://127.0.0.1:5000/
```
