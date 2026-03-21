import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.base import EvolutionaryAgent
from agents.thermodynamic.thermo_injector import ThermodynamicInjector
from core.chaos import LorenzGenerator
from agi.components import Hippocampus, WorldModel, HierarchicalController

class AGIAgent(EvolutionaryAgent):
    """
    Thermodynamic AGI v1.0.
    Integrates Memory, World Modeling, Hierarchy, and Curiosity.
    """
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # 1. The Brain (Hierarchical)
        self.brain = HierarchicalController(input_dim, hidden_dim, action_dim)
        
        # 2. The Simulator (World Model)
        self.world_model = WorldModel(input_dim, action_dim, hidden_dim)
        
        # 3. The Memory (Hippocampus)
        self.memory = Hippocampus()
        
        # 4. The Drive (Thermodynamics)
        self.injector = ThermodynamicInjector(LorenzGenerator(), base_mutation_rate=0.02)
        self.sigma_history = []
        self.current_sigma = 0.0
        
        # Optimizers (Separate for Brain and World Model)
        self.brain_optimizer = optim.Adam(self.brain.parameters(), lr=0.01)
        self.wm_optimizer = optim.Adam(self.world_model.parameters(), lr=0.005)

    def forward(self, state):
        # Forward pass through the Hierarchical Brain
        logits, goal = self.brain(state)
        
        # Measure Sigma (Internal Work)
        # We use the variance of the goal vector as a proxy for "Managerial Thought"
        with torch.no_grad():
            if goal.shape[0] > 1:
                sigma = torch.var(goal, dim=1).mean().item()
            else:
                sigma = torch.var(goal).item()
            self.current_sigma = sigma
            self.sigma_history.append(sigma)
            
        return logits

    def act(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        logits = self(state_t)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action

    def get_intrinsic_reward(self, state, action, next_state):
        """
        Gap 5: Curiosity.
        Reward = Prediction Error (Surprise).
        If the World Model can't predict this transition, it's interesting!
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        action_t = torch.tensor([action])
        
        error = self.world_model.get_prediction_error(state_t, action_t, next_state_t)
        return error.item()

    def sleep(self, epochs=10):
        """
        Gap 1 & 3: Consolidation.
        Train the World Model on memories.
        """
        if len(self.memory.buffer) < 64:
            return 0.0
            
        total_loss = 0
        for _ in range(epochs):
            batch = self.memory.dream(batch_size=64)
            if not batch: break
            
            states, actions, _, next_states, _ = batch
            
            self.wm_optimizer.zero_grad()
            
            # Predict next state delta
            pred_delta = self.world_model(states, actions)
            target_delta = next_states - states
            
            loss = nn.MSELoss()(pred_delta, target_delta)
            loss.backward()
            self.wm_optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / epochs

    def get_thermodynamic_status(self):
        if len(self.sigma_history) < 10: return 'healthy'
        avg_sigma = np.mean(self.sigma_history[-10:])
        if avg_sigma < 0.01: return 'frozen' # Goal vector collapsed
        return 'healthy'

    def mutate(self):
        """Apply thermodynamic mutation to the BRAIN (not the world model)."""
        self.brain = self.injector.mutate(self.brain)
        return self

    def get_topology_info(self):
        return {"sigma": self.current_sigma}
