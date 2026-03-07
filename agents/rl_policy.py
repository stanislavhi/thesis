import torch
import torch.nn as nn
import numpy as np

class EvolvingPolicy(nn.Module):
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

class RLChaosInjector:
    def __init__(self, generator):
        self.gen = generator
    
    def mutate(self, policy_net):
        chaos = self.gen.get_perturbation()
        print(f"\n>>> CHAOS INJECTION (Z={chaos:.2f}): Mutating Brain Topology...", flush=True)
        
        # Extract current structure
        first_layer = policy_net.net[0]
        last_layer = policy_net.net[2]
        
        input_dim = first_layer.in_features
        current_hidden = first_layer.out_features
        output_dim = last_layer.out_features
        
        # Determine new size
        change = int(chaos * 5)
        if change == 0: change = 1 if np.random.random() > 0.5 else -1
        new_hidden = max(4, min(current_hidden + change, 64))
        
        # Activation Shift
        current_act = policy_net.net[1]
        new_act = current_act
        act_name = "Unchanged"
        if abs(chaos) > 1.5:
            if isinstance(current_act, nn.ReLU): new_act, act_name = nn.Tanh(), "ReLU->Tanh"
            elif isinstance(current_act, nn.Tanh): new_act, act_name = nn.GELU(), "Tanh->GELU"
            else: new_act, act_name = nn.ReLU(), "GELU->ReLU"
            
        print(f"   -> Resizing: {current_hidden}->{new_hidden} | {act_name}", flush=True)
        
        # Create new network
        new_policy = EvolvingPolicy(input_dim, new_hidden, output_dim)
        new_policy.net[1] = new_act
        
        # Transfer Weights (Neuroplasticity)
        self._transfer_weights(policy_net.net, new_policy.net)
        
        return new_policy

    def _transfer_weights(self, old, new):
        try:
            with torch.no_grad():
                # Layer 1
                n_in = min(old[0].in_features, new[0].in_features)
                n_out = min(old[0].out_features, new[0].out_features)
                new[0].weight[:n_out, :n_in] = old[0].weight[:n_out, :n_in]
                new[0].bias[:n_out] = old[0].bias[:n_out]
                
                # Layer 2
                n_in_2 = min(old[2].in_features, new[2].in_features)
                n_out_2 = min(old[2].out_features, new[2].out_features)
                new[2].weight[:n_out_2, :n_in_2] = old[2].weight[:n_out_2, :n_in_2]
                new[2].bias[:n_out_2] = old[2].bias[:n_out_2]
        except: pass
