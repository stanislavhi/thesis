import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque


class Hippocampus:
    """
    Episodic memory with salience-based storage.
    Stores high-salience experiences and allows replay ('dreaming') for world model training.
    """
    def __init__(self, capacity=10000, salience_threshold=1.5):
        self.buffer = deque(maxlen=capacity)
        self.salience_threshold = salience_threshold

    def remember(self, state, action, reward, next_state, done):
        salience = abs(reward) + float(not np.array_equal(state, next_state))

        if salience > self.salience_threshold:
            self.buffer.append((state, action, reward, next_state, done))

    def dream(self, batch_size=64):
        """Sample a batch of memories for consolidation."""
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )


class WorldModel(nn.Module):
    """
    Predicts next-state deltas given (state, action).
    Used for planning (simulating futures) and curiosity (prediction error = intrinsic reward).
    """
    ACTION_EMBED_DIM = 4

    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, self.ACTION_EMBED_DIM)
        self.net = nn.Sequential(
            nn.Linear(input_dim + self.ACTION_EMBED_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, state, action):
        action_emb = self.action_embed(action.long())
        x = torch.cat([state, action_emb], dim=1)
        return self.net(x)

    def get_prediction_error(self, state, action, next_state):
        """Curiosity signal: MSE between predicted and actual next state."""
        with torch.no_grad():
            pred_delta = self.forward(state, action)
            pred_next_state = state + pred_delta
            error = torch.mean((next_state - pred_next_state) ** 2, dim=1)
        return error


class HierarchicalController(nn.Module):
    """
    Manager-Worker feudal architecture.
    Manager: state -> goal vector (bounded by tanh).
    Worker: (state, goal) -> action logits.
    Exposes worker hidden activations for thermodynamic sigma measurement.
    """
    def __init__(self, input_dim, hidden_dim, action_dim, goal_dim=4):
        super().__init__()

        self.manager = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, goal_dim)
        )

        # Worker split into layers to expose hidden activations
        self.worker_hidden = nn.Linear(input_dim + goal_dim, hidden_dim)
        self.worker_activation = nn.ReLU()
        self.worker_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        goal = self.manager(state)
        x = torch.cat([state, goal], dim=1)

        h = self.worker_hidden(x)
        h = self.worker_activation(h)
        logits = self.worker_head(h)

        return logits, goal, h
