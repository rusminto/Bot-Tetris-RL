"""
Exports a DQN checkpoint to the JSON layout the userscript embeds.

    python3 src/export_weights.py                                   # checkpoints/best_model.pt -> weights_v2.json
    python3 src/export_weights.py --checkpoint checkpoints_v1_legacy/best_model.pt --output weights_v1.json
"""

import os
import json
import argparse

import torch

RL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def export(checkpoint, output):
    print(f"Loading checkpoint from: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    weights = {}
    for key, val in state_dict.items():
        weights[key] = val.detach().cpu().numpy().tolist()
        print(f"Exported {key}: shape {tuple(val.shape)}")

    with open(output, "w") as f:
        json.dump(weights, f)
    print(f"Saved weights to {output} ({os.path.getsize(output)} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a DQN checkpoint for the userscript")
    parser.add_argument("--checkpoint", default=os.path.join(RL_DIR, "checkpoints", "best_model.pt"))
    parser.add_argument("--output", default=os.path.join(RL_DIR, "weights_v2.json"))
    args = parser.parse_args()
    export(args.checkpoint, args.output)
