import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.chaos import LorenzGenerator
from agents.swarm import SwarmAgent, HolographicChannel, ChaosInjector


class SwarmAggregator(nn.Module):
    """
    Aggregates noisy thought vectors from multiple blind agents into action logits.
    Unlike HolographicAggregator, this outputs action_dim logits for proper softmax.
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


# Environment-specific agent configurations
# Each entry: list of (name, state_indices) per agent
ENV_CONFIGS = {
    "CartPole-v1": {
        "agents": [
            ("Position", [0, 1]),   # Cart position & velocity
            ("Angle", [2, 3]),      # Pole angle & angular velocity
        ],
        "thought_size": 3,
        "hidden_dim": 16,
        "agg_hidden": 32,
        "episodes": 1000,
        "solved_threshold": 195,
        "lr": 0.01,
    },
    "LunarLander-v3": {
        "agents": [
            ("Navigator", [0, 1, 2, 3]),  # x, y, vx, vy
            ("Pilot", [4, 5]),             # angle, angular_vel
            ("Landing", [6, 7]),           # left_leg, right_leg
        ],
        "thought_size": 3,
        "hidden_dim": 16,
        "agg_hidden": 32,
        "episodes": 2000,
        "solved_threshold": 200,
        "lr": 0.005,
    },
}


def train_holographic_swarm(env_name, episodes=None):
    config = ENV_CONFIGS.get(env_name)
    if config is None:
        print(f"Error: No agent configuration defined for '{env_name}'.")
        print(f"Supported environments: {list(ENV_CONFIGS.keys())}")
        return

    if episodes is None:
        episodes = config["episodes"]

    env = gym.make(env_name)
    action_dim = env.action_space.n
    thought_size = config["thought_size"]

    # Initialize agents — each sees only its assigned state indices
    agents = []
    agent_configs = config["agents"]
    for name, indices in agent_configs:
        input_dim = len(indices)
        agent = SwarmAgent(input_dim, config["hidden_dim"], thought_size, name)
        agents.append(agent)

    # Channel adds Hawking noise proportional to inverse performance
    channel = HolographicChannel(bekenstein_bits=1e70)

    # Aggregator: takes all thought vectors, outputs action probabilities
    total_thought_dim = len(agents) * thought_size
    aggregator = SwarmAggregator(total_thought_dim, config["agg_hidden"], action_dim)

    # Optimizer over all components
    params = list(aggregator.parameters()) + list(channel.parameters())
    for a in agents:
        params += list(a.parameters())
    optimizer = optim.Adam(params, lr=config["lr"])

    # Chaos engine
    chaos_gen = LorenzGenerator()
    injector = ChaosInjector(chaos_gen)

    scores = deque(maxlen=50)
    log_data = []

    # Header
    agent_desc = " | ".join(
        f"{name} sees {indices}" for name, indices in agent_configs
    )
    print(f"--- HOLOGRAPHIC SWARM: BLIND {env_name} ---")
    print(f"Agents: {agent_desc}")
    print(f"Thought size: {thought_size} | Aggregator hidden: {config['agg_hidden']}")
    print(f"Episodes: {episodes} | Solved: {config['solved_threshold']}")
    print("They must communicate to survive.\n", flush=True)

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        while True:
            # Each agent observes only its assigned state slice
            thoughts = []
            for i, (name, indices) in enumerate(agent_configs):
                agent_obs = torch.FloatTensor(state[indices]).unsqueeze(0)
                thought = agents[i](agent_obs)
                thoughts.append(thought)

            # Communicate via noisy channel
            avg_score = np.mean(scores) if len(scores) > 0 else 10.0
            noise_level = 10.0 / (avg_score + 1e-9)
            noisy_thoughts = channel(thoughts, noise_level)

            # Aggregator decides action
            probs = aggregator(noisy_thoughts)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, truncated, _ = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

            if done or truncated:
                break

        total_reward = sum(rewards)
        scores.append(total_reward)
        avg_score = np.mean(scores)

        # REINFORCE Update
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, R in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * R)

        optimizer.zero_grad()
        if policy_loss:
            torch.stack(policy_loss).sum().backward()
            optimizer.step()

        # Log data
        agent_sizes = "|".join(str(a.net[0].out_features) for a in agents)
        log_data.append(f"{episode},{total_reward},{avg_score},{agent_sizes}")

        # Chaos Injection on stagnation
        if episode > 20 and avg_score < config["solved_threshold"] * 0.5 and np.std(scores) < 5.0:
            print(f"\n[Episode {episode}] Swarm Stagnation (Avg: {avg_score:.1f}). Mutating...", flush=True)

            # Mutate: use ChaosInjector but rebuild with correct aggregator type
            new_agents_list, _ = injector.mutate_swarm(agents, aggregator)
            agents = new_agents_list

            # Rebuild aggregator with correct output dim
            new_total_thought = sum(a.net[-1].out_features for a in agents)
            aggregator = SwarmAggregator(new_total_thought, config["agg_hidden"], action_dim)

            # Re-init optimizer
            params = list(aggregator.parameters()) + list(channel.parameters())
            for a in agents:
                params += list(a.parameters())
            optimizer = optim.Adam(params, lr=config["lr"])
            scores.clear()

        if episode % 20 == 0:
            print(f"Episode {episode} | Score: {total_reward:.0f} | Avg: {avg_score:.1f} | "
                  f"Agents: [{agent_sizes}]", flush=True)

        if avg_score > config["solved_threshold"]:
            print(f"\n>>> HOLOGRAPHIC SWARM SOLVED {env_name}! "
                  f"Avg: {avg_score:.1f}", flush=True)
            break

    env.close()

    # Save log
    env_tag = env_name.replace("-", "_").lower()
    log_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f'../../logs/holographic_swarm_{env_tag}_log.csv'
    ))
    with open(log_path, "w") as f:
        f.write("episode,score,avg_score,agent_hidden_sizes\n")
        f.write("\n".join(log_data))
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holographic Swarm — Blind Agents Challenge")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gymnasium environment (default: CartPole-v1)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes (default: from preset)")
    args = parser.parse_args()

    train_holographic_swarm(args.env, args.episodes)
