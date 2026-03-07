import torch
import torch.nn as nn
import numpy as np
from core.base import EvolutionaryAgent, Mutator
from core.chaos import LorenzGenerator

class SolidEvolvingPolicy(EvolutionaryAgent):
    """
    A SOLID implementation of the Evolving Policy.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.net(x)
        
    def get_topology_info(self):
        return {
            "hidden_size": self.net[0].out_features,
            "activation": str(self.net[1])
        }

class SolidChaosMutator(Mutator):
    def __init__(self):
        self.gen = LorenzGenerator()
        
    def mutate(self, agent: SolidEvolvingPolicy) -> SolidEvolvingPolicy:
        chaos = self.gen.get_perturbation()
        
        # Extract current structure
        first_layer = agent.net[0]
        last_layer = agent.net[2]
        
        input_dim = first_layer.in_features
        current_hidden = first_layer.out_features
        output_dim = last_layer.out_features
        
        # Determine new size
        change = int(chaos * 5)
        if change == 0: change = 1 if np.random.random() > 0.5 else -1
        new_hidden = max(4, min(current_hidden + change, 128))
        
        # Activation Shift
        current_act = agent.net[1]
        new_act = current_act
        if abs(chaos) > 1.5:
            if isinstance(current_act, nn.ReLU): new_act = nn.Tanh()
            elif isinstance(current_act, nn.Tanh): new_act = nn.GELU()
            else: new_act = nn.ReLU()
            
        # Create new network
        new_agent = SolidEvolvingPolicy(input_dim, new_hidden, output_dim)
        new_agent.net[1] = new_act
        
        # Transfer Weights
        self._transfer_weights(agent.net, new_agent.net)
        
        return new_agent

    def _transfer_weights(self, old, new):
        try:
            with torch.no_grad():
                n_in = min(old[0].in_features, new[0].in_features)
                n_out = min(old[0].out_features, new[0].out_features)
                new[0].weight[:n_out, :n_in] = old[0].weight[:n_out, :n_in]
                new[0].bias[:n_out] = old[0].bias[:n_out]
                
                n_in_2 = min(old[2].in_features, new[2].in_features)
                n_out_2 = min(old[2].out_features, new[2].out_features)
                new[2].weight[:n_out_2, :n_in_2] = old[2].weight[:n_out_2, :n_in_2]
                new[2].bias[:n_out_2] = old[2].bias[:n_out_2]
        except: pass
