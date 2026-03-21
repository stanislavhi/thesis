import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

class Hippocampus:
    """
    Gap 1: Memory & Consolidation.
    Stores high-salience experiences and allows 'dreaming' (replay) to train the World Model.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def remember(self, state, action, reward, next_state, done):
        # Only store 'meaningful' experiences (non-zero reward or transition)
        # In a sparse maze, finding the goal is high salience.
        # Hitting a wall (state == next_state) is also salient (pain).
        salience = abs(reward) + float(not np.array_equal(state, next_state))
        
        if salience > 0.01:
            self.buffer.append((state, action, reward, next_state, done))
            
    def dream(self, batch_size=64):
        """Return a batch of memories for consolidation."""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        # Unzip
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
    Gap 3: World Model.
    Predicts the next state given current state and action.
    Used for:
    1. Planning (simulating futures).
    2. Curiosity (intrinsic reward = prediction error).
    """
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), # +1 for action index
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim) # Predict next state delta
        )
        
    def forward(self, state, action):
        # Action is an index, we treat it as a scalar feature for simplicity here
        # (In complex envs, use embeddings)
        action = action.float().unsqueeze(1)
        x = torch.cat([state, action], dim=1)
        return self.net(x)
    
    def get_prediction_error(self, state, action, next_state):
        """Calculates surprise (curiosity signal)."""
        with torch.no_grad():
            pred_delta = self.forward(state, action)
            pred_next_state = state + pred_delta
            error = torch.mean((next_state - pred_next_state) ** 2, dim=1)
        return error

class HierarchicalController(nn.Module):
    """
    Gap 2: Hierarchical Goals.
    Manager: Sets a 'Goal Vector' (Context).
    Worker: Takes State + Goal -> Action.
    """
    def __init__(self, input_dim, hidden_dim, action_dim, goal_dim=4):
        super().__init__()
        
        # Manager: Looks at state, outputs a latent goal
        self.manager = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), # Tanh to keep goals bounded [-1, 1]
            nn.Linear(hidden_dim, goal_dim)
        )
        
        # Worker: Looks at state AND goal, outputs action
        self.worker = nn.Sequential(
            nn.Linear(input_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        goal = self.manager(state)
        # Detach goal gradient? Usually yes for feudal RL, but for simple ES we can evolve both.
        # We'll treat them as one big brain for now.
        x = torch.cat([state, goal], dim=1)
        logits = self.worker(x)
        return logits, goal
