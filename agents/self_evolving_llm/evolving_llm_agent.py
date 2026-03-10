import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.base import EvolutionaryAgent
# Correct the import path to use the self-contained test file
from qwen.tests.test_qwen_thermodynamic import QwenThermodynamicModel

class EvolvingLLMAgent(EvolutionaryAgent):
    """
    A Language Agent that monitors its own internal entropy production (sigma)
    and evolves its weights in real-time to overcome reasoning blocks.
    
    It wraps a QwenThermodynamicModel and adds:
    1. Layer-wise activity monitoring (sigma).
    2. A 'thermodynamic state' diagnosis.
    3. Integration with the ThermodynamicInjector.
    """
    def __init__(self, vocab_size=1000, hidden_dim=256, num_heads=4, num_layers=4, max_seq_len=128):
        super().__init__()
        
        # Initialize the core language model
        self.model = QwenThermodynamicModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len
        )
        
        # Physics Monitoring
        self.sigma_history = [] # Tracks entropy production over time
        self.current_sigma = 0.0
        self.layer_sigmas = [] # Detailed layer-wise sigma
        
    def forward(self, input_ids, attention_mask=None):
        # We need to capture the 'work' done by each layer to calculate sigma.
        # The QwenThermodynamicModel implementation returns logits and a diag dict,
        # but we need to hook into the internal states.
        
        # Hook to capture hidden states
        hidden_states = []
        def get_hook():
            def hook(module, input, output):
                hidden_states.append(output.detach())
            return hook

        hooks = []
        # Register hooks on attention modules
        for i in range(len(self.model.layers)):
            layer_dict = self.model.layers[i]
            hook = layer_dict['attention'].register_forward_hook(get_hook())
            hooks.append(hook)
            
        # Run forward pass
        logits, _ = self.model(input_ids, attention_mask)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        # Calculate Sigma (Entropy Production)
        # Sigma is the mean squared magnitude of the update vector across all layers.
        if hidden_states:
            layer_sigmas = []
            for state in hidden_states:
                # state shape: (batch, seq_len, hidden_dim)
                # We calculate the mean squared magnitude of the update vector
                sigma = torch.mean(state ** 2).item()
                layer_sigmas.append(sigma)
            
            self.layer_sigmas = layer_sigmas
            self.current_sigma = np.mean(layer_sigmas)
            self.sigma_history.append(self.current_sigma)
            
            # Keep history manageable
            if len(self.sigma_history) > 100:
                self.sigma_history.pop(0)
        
        return logits

    def get_topology_info(self):
        return {
            "vocab": self.model.vocab_size,
            "hidden": self.model.hidden_dim,
            "layers": self.model.num_layers,
            "sigma": self.current_sigma
        }

    def get_thermodynamic_status(self):
        """
        Diagnose the agent's health based on internal physics.
        Returns: 'healthy', 'frozen', or 'overheated'
        """
        if len(self.sigma_history) < 5:
            return 'healthy'
            
        avg_sigma = np.mean(self.sigma_history[-5:])
        
        # Thresholds calibrated based on previous experiments
        # Normal healthy sigma is > 0.2
        # Frozen sigma is < 0.15 (based on LunarLander stress test)
        if avg_sigma < 0.15:
            return 'frozen'  # Agent is struggling (low variance/dead neurons)
        elif avg_sigma > 10.0:
            return 'overheated' # Exploding activations
        else:
            return 'healthy'
