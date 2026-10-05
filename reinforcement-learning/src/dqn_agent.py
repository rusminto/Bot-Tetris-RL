"""
Deep Q-Network (DQN) Agent for Tetris.
Evaluates candidate board states using a multi-layer neural network trained with MSE loss and experience replay.
"""

import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TetrisQNetwork(nn.Module):
    def __init__(self, input_size=6, hidden_size=64):
        super(TetrisQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        lr=1e-3,
        gamma=0.99,
        replay_size=30000,
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.replay_buffer = deque(maxlen=replay_size)

        self.model = TetrisQNetwork(input_size, hidden_size).to(self.device)
        self.target_model = TetrisQNetwork(input_size, hidden_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def select_action(self, next_states_dict, epsilon=0.0):
        """
        Given dict {(use_hold, rot, x): (features, next_board)}, select best action.
        Returns:
            best_action: (use_hold, rot, x)
            best_state_features: np.ndarray
        """
        if not next_states_dict:
            return None, None

        actions = list(next_states_dict.keys())

        # Exploration: pick random action
        if random.random() < epsilon:
            chosen_action = random.choice(actions)
            return chosen_action, next_states_dict[chosen_action][0]

        # Exploitation: evaluate all candidate states in one forward pass
        features_list = [next_states_dict[a][0] for a in actions]
        states_tensor = torch.tensor(np.array(features_list), dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            q_values = self.model(states_tensor).squeeze(-1)
        self.model.train()

        best_idx = torch.argmax(q_values).item()
        best_action = actions[best_idx]
        return best_action, features_list[best_idx]

    def store_transition(self, state, reward, next_state, done):
        self.replay_buffer.append((state, reward, next_state, done))

    def train_step(self, batch_size=512):
        if len(self.replay_buffer) < batch_size:
            return None

        batch = random.sample(self.replay_buffer, batch_size)
        states, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Handle terminal states vs non-terminal states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=self.device, dtype=torch.bool)
        non_final_next_states = [s for s in next_states if s is not None]

        # Predicted Q(s)
        current_q = self.model(states_tensor)

        # Target Q(s')
        next_q = torch.zeros(batch_size, 1, device=self.device)
        if non_final_next_states:
            non_final_tensor = torch.tensor(np.array(non_final_next_states), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                next_q[non_final_mask] = self.target_model(non_final_tensor)

        expected_q = rewards_tensor + (1.0 - dones_tensor) * self.gamma * next_q

        loss = self.criterion(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(self.model.state_dict())
        self.model.eval()
