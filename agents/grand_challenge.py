import torch
import torch.nn as nn
import torch.optim as optim
from agents.swarm import SwarmAgent, HolographicChannel, ChaosInjector
from core.chaos import LorenzGenerator


class SwarmAggregator(nn.Module):
    """
    Aggregates noisy thought vectors from multiple blind agents into action logits.
    Outputs action_dim logits for proper softmax.
    """
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class BlindAgent(nn.Module):
    """
    Wraps a SwarmAgent and ensures it only observes a specific slice of the environment.
    """
    def __init__(self, input_dim, hidden_dim, thought_size, name, indices):
        super().__init__()
        self.indices = indices
        self.agent = SwarmAgent(input_dim, hidden_dim, thought_size, name)

    def forward(self, full_state):
        agent_obs = torch.FloatTensor(full_state[self.indices]).unsqueeze(0)
        return self.agent(agent_obs)


class HolographicSwarm(nn.Module):
    """
    Orchestrates multiple BlindAgents, the HolographicChannel, and the SwarmAggregator.
    Includes built-in optimization and the ability to mutate upon stagnation.
    """
    def __init__(self, action_dim, thought_size, hidden_dim, agg_hidden, agent_configs, lr):
        super().__init__()
        self.action_dim = action_dim
        self.thought_size = thought_size
        self.hidden_dim = hidden_dim
        self.agg_hidden = agg_hidden
        self.agent_configs = agent_configs
        self.lr = lr
        
        # Initialize blind agents
        self.agents = nn.ModuleList([
            BlindAgent(len(indices), hidden_dim, thought_size, name, indices)
            for name, indices in agent_configs
        ])
        
        self.channel = HolographicChannel(bekenstein_bits=1e70)
        
        total_thought_dim = len(self.agents) * thought_size
        self.aggregator = SwarmAggregator(total_thought_dim, agg_hidden, action_dim)
        
        chaos_gen = LorenzGenerator()
        self.injector = ChaosInjector(chaos_gen)
        
        self._init_optimizer()
        
    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def predict(self, full_state, avg_score):
        """Generates an action probabilistically, passing thoughts through the noisy channel."""
        # 1. Individual localized observations yield thoughts
        thoughts = [ba(full_state) for ba in self.agents]
        
        # 2. Add Hawking noise inversely proportional to swarm performance
        noise_level = 10.0 / (avg_score + 1e-9)
        noisy_thoughts = self.channel(thoughts, noise_level)
        
        # 3. Global Aggregation
        probs = self.aggregator(noisy_thoughts)
        
        # 4. Action Selection
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob

    def update(self, policy_loss_list):
        """Steps the optimizer on the REINFORCE loss."""
        self.optimizer.zero_grad()
        if policy_loss_list:
            torch.stack(policy_loss_list).sum().backward()
            self.optimizer.step()

    def mutate(self):
        """Applies chaos injection to mutate stagnant internal SwarmAgents."""
        pure_agents = [ba.agent for ba in self.agents]
        new_agents_list, _ = self.injector.mutate_swarm(pure_agents, self.aggregator)
        
        # Replace internal neural agents
        for i, ba in enumerate(self.agents):
            ba.agent = new_agents_list[i]
            
        # Rebuild aggregator sizing based on mutations
        new_total_thought = sum(ba.agent.net[-1].out_features for ba in self.agents)
        self.aggregator = SwarmAggregator(new_total_thought, self.agg_hidden, self.action_dim)
        
        # Re-initialize optimizer because parameters changed
        self._init_optimizer()

    def get_agent_sizes_string(self):
        return "|".join(str(ba.agent.net[0].out_features) for ba in self.agents)
