import torch
import torch.nn as nn
import numpy as np
import math

class SwarmAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, agent_id):
        super().__init__()
        self.id = agent_id
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # The "Thought Vector"
        )
        
    def forward(self, x):
        return self.net(x)

class HolographicChannel(nn.Module):
    """
    Simulates the physical constraints of communication.
    - Bandwidth is limited by the Bekenstein Bound.
    - Noise is proportional to the system's "temperature" (loss).
    """
    def __init__(self, bekenstein_bits):
        super().__init__()
        # Scale the immense Bekenstein Bound to a usable thought vector size
        self.thought_vector_size = int(4 + math.log10(bekenstein_bits / 1e68))
        print(f"   >> Channel bandwidth set to {self.thought_vector_size} based on Bekenstein Bound.")

    def forward(self, thoughts, current_loss):
        # 1. Concatenate thoughts from all agents
        combined = torch.cat(thoughts, dim=1)
        
        # 2. Add "Hawking Radiation" (Noise)
        # Noise is proportional to the system's temperature (loss)
        noise_std = current_loss * 0.05 # Heuristic scaling factor
        noise = torch.randn_like(combined) * noise_std
        
        return combined + noise

class HolographicAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Final Prediction
        )
        
    def forward(self, x):
        return self.net(x)

class ChaosInjector:
    def __init__(self, generator):
        self.gen = generator
    
    def mutate_swarm(self, agents, aggregator):
        """
        Mutates the entire swarm based on chaos.
        Some agents might grow, others shrink. The Aggregator might change topology.
        """
        chaos = self.gen.get_perturbation()
        print(f"\n>>> CHAOS INJECTION (Z={chaos:.2f}): Mutating Swarm Topology...", flush=True)
        
        # Mutate each agent
        new_agents = []
        for agent in agents:
            # Randomly decide to mutate this agent based on chaos
            if np.random.random() < 0.7: 
                new_agent = self._mutate_single_net(agent.net, agent.id)
                wrapped_agent = SwarmAgent(1, 1, 1, agent.id) # Dummy init
                wrapped_agent.net = new_agent
                new_agents.append(wrapped_agent)
            else:
                new_agents.append(agent)
                
        # Mutate Aggregator
        new_aggregator_net = self._mutate_single_net(aggregator.net, "Aggregator")
        new_aggregator = HolographicAggregator(1, 1) # Dummy init
        new_aggregator.net = new_aggregator_net
        
        return new_agents, new_aggregator

    def _mutate_single_net(self, model, name):
        if not isinstance(model, nn.Sequential): return model
        
        first = model[0]
        last = model[-1]
        input_dim = first.in_features
        current_hidden = first.out_features
        output_dim = last.out_features
        
        chaos = self.gen.get_perturbation()
        change = int(chaos * 5)
        if change == 0: change = 1 if np.random.random() > 0.5 else -1
        
        new_hidden = max(8, min(current_hidden + change, 128))
        
        # Activation Shift
        current_act = model[1]
        new_act = current_act
        act_name = "Unchanged"
        if abs(chaos) > 1.5:
            if isinstance(current_act, nn.ReLU): new_act, act_name = nn.Tanh(), "ReLU->Tanh"
            elif isinstance(current_act, nn.Tanh): new_act, act_name = nn.GELU(), "Tanh->GELU"
            else: new_act, act_name = nn.ReLU(), "GELU->ReLU"
            
        print(f"   -> Agent {name}: {current_hidden}->{new_hidden} | {act_name}", flush=True)
        
        new_model = nn.Sequential(
            nn.Linear(input_dim, new_hidden),
            new_act,
            nn.Linear(new_hidden, output_dim)
        )
        
        self._transfer_weights(model, new_model)
        return new_model

    def _transfer_weights(self, old, new):
        try:
            with torch.no_grad():
                # Layer 1
                n_in = min(old[0].in_features, new[0].in_features)
                n_out = min(old[0].out_features, new[0].out_features)
                new[0].weight[:n_out, :n_in] = old[0].weight[:n_out, :n_in]
                new[0].bias[:n_out] = old[0].bias[:n_out]
                
                # Layer 2 (Output)
                n_in_2 = min(old[2].in_features, new[2].in_features)
                n_out_2 = min(old[2].out_features, new[2].out_features)
                new[2].weight[:n_out_2, :n_in_2] = old[2].weight[:n_out_2, :n_in_2]
                new[2].bias[:n_out_2] = old[2].bias[:n_out_2]
        except Exception as e:
            print(f"Warning: weight transfer failed: {e}")
