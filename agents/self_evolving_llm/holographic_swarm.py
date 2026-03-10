import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from agents.self_evolving_llm.evolving_llm_agent import EvolvingLLMAgent

class LLMSwarmAgent(nn.Module):
    """
    A single agent within the Holographic Swarm.
    
    It's a wrapper around the EvolvingLLMAgent, designed to process a partial
    observation and output a compressed "Thought Vector".
    """
    def __init__(self, agent_id, vocab_size=1000, hidden_dim=64, num_layers=2, thought_vector_dim=32):
        super().__init__()
        self.agent_id = agent_id
        
        # The "brain" of the agent is a self-evolving LLM
        self.brain = EvolvingLLMAgent(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_seq_len=64
        )
        
        # A projection head to compress the final hidden state into a "Thought Vector"
        self.thought_projector = nn.Linear(hidden_dim, thought_vector_dim)
        
    def forward(self, partial_observation):
        """
        Processes a partial view of the world and outputs a thought.
        
        Args:
            partial_observation (Tensor): Shape (batch_size, seq_len)
            
        Returns:
            thought_vector (Tensor): Shape (batch_size, thought_vector_dim)
        """
        # Get the final hidden state from the brain
        # The EvolvingLLMAgent's forward returns logits, but we need the hidden states.
        # We'll re-implement a simplified forward pass here for that purpose.
        
        x = self.brain.model.embedding(partial_observation)
        for layer in self.brain.model.layers:
            attn_out = layer['attention'](layer['norm1'](x))
            x = x + attn_out
            mlp_out = layer['mlp'](layer['norm2'](x))
            x = x + mlp_out
            
        # Use the final hidden state of the [CLS] token (or average pooling)
        final_hidden_state = x.mean(dim=1) # Average pooling over the sequence
        
        # Project to a compressed thought vector
        thought_vector = self.thought_projector(final_hidden_state)
        
        return thought_vector

    def get_thermodynamic_status(self):
        """Pass through the brain's self-diagnosis."""
        return self.brain.get_thermodynamic_status()


class HolographicChannel(nn.Module):
    """
    The communication medium.
    
    Simulates physical constraints:
    1. Bandwidth Limit: Fixed thought vector size.
    2. Hawking Noise: Adds noise proportional to the system's 'temperature'.
    """
    def __init__(self, noise_level=0.1):
        super().__init__()
        self.base_noise = noise_level
        
    def forward(self, thoughts, system_temperature=1.0):
        """
        Args:
            thoughts (List[Tensor]): List of thought vectors from agents.
            system_temperature (float): Current stress/loss level of the system.
        """
        # Concatenate all thoughts into a single tensor
        combined_thoughts = torch.cat(thoughts, dim=-1)
        
        # Calculate noise magnitude
        # Higher temperature (stress) = More noise (communication breakdown)
        current_noise = self.base_noise * system_temperature
        
        # Add Gaussian noise
        noise = torch.randn_like(combined_thoughts) * current_noise
        
        return combined_thoughts + noise


class SwarmAggregator(nn.Module):
    """
    The central mind that fuses the noisy thoughts.
    """
    def __init__(self, num_agents, thought_vector_dim, output_dim):
        super().__init__()
        input_dim = num_agents * thought_vector_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, noisy_thoughts):
        return self.net(noisy_thoughts)
