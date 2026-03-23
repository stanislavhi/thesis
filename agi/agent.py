import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.base import EvolutionaryAgent
from agents.thermodynamic.thermo_injector import ThermodynamicInjector
from core.chaos import LorenzGenerator
from agi.components import Hippocampus, WorldModel, HierarchicalController


class AGIAgent(EvolutionaryAgent):
    """
    Thermodynamic AGI agent integrating:
    1. HierarchicalController (Manager-Worker brain)
    2. WorldModel (prediction + curiosity)
    3. Hippocampus (salience-filtered episodic memory)
    4. ThermodynamicInjector (chaos-driven mutation via operator selection)

    Sigma is measured from worker hidden-layer activation variance,
    consistent with ThermodynamicAgent's neuron-health diagnostic.
    """
    def __init__(self, input_dim, action_dim, hidden_dim=64,
                 brain_lr=0.01, wm_lr=0.005, base_mutation_rate=0.02,
                 memory_capacity=10000, salience_threshold=1.5):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.brain = HierarchicalController(input_dim, hidden_dim, action_dim)
        self.world_model = WorldModel(input_dim, action_dim, hidden_dim)
        self.memory = Hippocampus(capacity=memory_capacity,
                                  salience_threshold=salience_threshold)

        self.injector = ThermodynamicInjector(LorenzGenerator(),
                                              base_mutation_rate=base_mutation_rate)
        self.sigma_history = []
        self.current_sigma = 0.0
        self.last_wm_loss = 0.0

        self.brain_optimizer = optim.Adam(self.brain.parameters(), lr=brain_lr)
        self.wm_optimizer = optim.Adam(self.world_model.parameters(), lr=wm_lr)

    def forward(self, state):
        logits, goal, h_worker = self.brain(state)

        # Sigma from worker hidden activations (neuron health diagnostic)
        with torch.no_grad():
            if h_worker.shape[0] > 1:
                sigma = torch.var(h_worker, dim=1).mean().item()
            else:
                sigma = torch.var(h_worker).item()
            self.current_sigma = sigma
            self.sigma_history.append(sigma)
            if len(self.sigma_history) > 100:
                self.sigma_history.pop(0)

        return logits

    def act(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        logits = self(state_t)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action

    def get_intrinsic_reward(self, state, action, next_state):
        """Curiosity: prediction error from the world model."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        action_t = torch.tensor([action])
        error = self.world_model.get_prediction_error(state_t, action_t, next_state_t)
        return error.item()

    def sleep(self, epochs=10):
        """Train the world model on salient memories (consolidation)."""
        if len(self.memory.buffer) < 64:
            return 0.0

        total_loss = 0
        for _ in range(epochs):
            batch = self.memory.dream(batch_size=64)
            if not batch:
                break

            states, actions, _, next_states, _ = batch

            self.wm_optimizer.zero_grad()
            pred_delta = self.world_model(states, actions)
            target_delta = next_states - states
            loss = nn.MSELoss()(pred_delta, target_delta)
            loss.backward()
            self.wm_optimizer.step()
            total_loss += loss.item()

        self.last_wm_loss = total_loss / max(epochs, 1)
        return self.last_wm_loss

    def get_thermodynamic_status(self):
        """
        Diagnose agent health via sigma stagnation detection.

        Instead of absolute thresholds (which are architecture-dependent),
        we detect whether sigma is CHANGING. A flat sigma trace means the
        network's internal dynamics have stalled — regardless of the
        absolute value.

        - frozen:     sigma coefficient of variation < 5% over last 20 steps
        - overheated: sigma is diverging (recent >> baseline)
        - healthy:    sigma is varying normally
        """
        if len(self.sigma_history) < 20:
            return 'healthy'

        recent = np.array(self.sigma_history[-20:])
        mean_sigma = np.mean(recent)

        if mean_sigma < 1e-9:
            return 'frozen'

        cv = np.std(recent) / mean_sigma

        # Low coefficient of variation → sigma is flat → frozen
        if cv < 0.05:
            return 'frozen'

        # Check for divergence: recent mean >> earlier baseline
        if len(self.sigma_history) >= 40:
            baseline = np.mean(self.sigma_history[-40:-20])
            if baseline > 1e-9 and mean_sigma > baseline * 5.0:
                return 'overheated'

        return 'healthy'

    def mutate(self):
        """
        Apply thermodynamic mutation to the brain.

        Forces additive_noise operator: in pure ES (no gradient recovery),
        targeted_dropout permanently kills neurons since there's no optimizer
        to revive zeroed weights. Additive noise preserves full network capacity.
        """
        status = self.get_thermodynamic_status()
        self.brain = self.injector.mutate(self.brain, status=status,
                                          operator_override='additive_noise')
        return self

    def get_topology_info(self):
        return {
            "sigma": self.current_sigma,
            "status": self.get_thermodynamic_status(),
            "brain_params": sum(p.numel() for p in self.brain.parameters()),
            "wm_loss": self.last_wm_loss,
            "memory_size": len(self.memory.buffer),
        }
