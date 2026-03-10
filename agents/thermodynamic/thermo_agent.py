import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.base import EvolutionaryAgent

class ThermodynamicAgent(EvolutionaryAgent):
    """
    An agent that monitors its own internal entropy production (sigma).
    
    It extends a standard neural network policy with:
    1. Layer-wise activity monitoring (sigma).
    2. A 'thermodynamic state' that tracks if the system is 'freezing' (low sigma).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Standard MLP Policy
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
        # Physics Monitoring
        self.sigma_history = [] # Tracks entropy production over time
        self.current_sigma = 0.0
        
    def forward(self, x):
        # We need to capture the 'work' done by the hidden layer
        # x shape: (batch, input_dim)
        
        # Layer 1 transformation
        h_pre = self.layer1(x)
        h = self.activation(h_pre)
        
        # Measure Sigma: The magnitude of the transformation in the hidden layer
        # We approximate sigma as the variance of the hidden activations.
        # High variance = rich representation = high entropy production.
        # Low variance (dead neurons) = low entropy = freezing.
        with torch.no_grad():
            # Calculate variance across the hidden dimension
            if h.shape[0] > 1:
                sigma = torch.var(h, dim=1).mean().item()
            else:
                sigma = torch.var(h).item()
            
            self.current_sigma = sigma
            self.sigma_history.append(sigma)
            # Keep history manageable
            if len(self.sigma_history) > 100:
                self.sigma_history.pop(0)
        
        # Layer 2
        logits = self.layer2(h)
        return self.softmax(logits)

    def get_topology_info(self):
        return {
            "input": self.input_dim,
            "hidden": self.hidden_dim,
            "output": self.output_dim,
            "sigma": self.current_sigma
        }

    def get_thermodynamic_status(self):
        """
        Diagnose the agent's health based on internal physics.
        Returns: 'healthy', 'frozen', or 'overheated'
        """
        if len(self.sigma_history) < 10:
            return 'healthy'
            
        avg_sigma = np.mean(self.sigma_history[-10:])
        
        # Thresholds calibrated based on LunarLander experiments
        # Normal healthy sigma is > 0.2
        # Damaged sigma is ~0.06 - 0.12
        if avg_sigma < 0.15:
            return 'frozen'  # Agent is struggling (low variance/dead neurons)
        elif avg_sigma > 10.0:
            return 'overheated' # Exploding activations
        else:
            return 'healthy'
