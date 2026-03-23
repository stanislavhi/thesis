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
from agents.grand_challenge import HolographicSwarm
from core.config_manager import ConfigManager

def train_holographic_swarm(env_name, episodes=None):
    full_config = ConfigManager.load_config()
    config = full_config.get("holographic_swarm", {}).get(env_name)
    if config is None:
        print(f"Error: No agent configuration defined for '{env_name}'.")
        print(f"Supported environments: {list(full_config.get('holographic_swarm', {}).keys())}")
        return

    if episodes is None:
        episodes = config["episodes"]

    env = gym.make(env_name)
    
    swarm = HolographicSwarm(
        action_dim=env.action_space.n,
        thought_size=config["thought_size"],
        hidden_dim=config["hidden_dim"],
        agg_hidden=config["agg_hidden"],
        agent_configs=config["agents"],
        lr=config["lr"]
    )

    scores = deque(maxlen=50)
    log_data = []

    agent_desc = " | ".join(f"{name} sees {indices}" for name, indices in config["agents"])
    print(f"--- HOLOGRAPHIC SWARM: BLIND {env_name} ---")
    print(f"Agents: {agent_desc}")
    print(f"Thought size: {config['thought_size']} | Aggregator hidden: {config['agg_hidden']}")
    print(f"Episodes: {episodes} | Solved: {config['solved_threshold']}")
    print("They must communicate to survive.\n", flush=True)

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        while True:
            avg_score = np.mean(scores) if len(scores) > 0 else 10.0
            action, log_prob = swarm.predict(state, avg_score)

            next_state, reward, done, truncated, _ = env.step(action)
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
            
        discounted_tensor = torch.tensor(discounted_rewards)
        if len(discounted_tensor) > 1:
            discounted_tensor = (discounted_tensor - discounted_tensor.mean()) / (discounted_tensor.std() + 1e-9)

        policy_loss = [-lp * R for lp, R in zip(log_probs, discounted_tensor)]
        swarm.update(policy_loss)

        # Logging
        agent_sizes = swarm.get_agent_sizes_string()
        log_data.append(f"{episode},{total_reward},{avg_score},{agent_sizes}")

        # Chaos Injection / Mutation logic
        if episode > 20 and avg_score < config["solved_threshold"] * 0.5 and np.std(scores) < 5.0:
            print(f"\n[Episode {episode}] Swarm Stagnation (Avg: {avg_score:.1f}). Mutating...", flush=True)
            swarm.mutate()
            scores.clear()

        if episode % 20 == 0:
            print(f"Episode {episode} | Score: {total_reward:.0f} | Avg: {avg_score:.1f} | Agents: [{agent_sizes}]", flush=True)

        if avg_score > config["solved_threshold"]:
            print(f"\n>>> HOLOGRAPHIC SWARM SOLVED {env_name}! Avg: {avg_score:.1f}", flush=True)
            break

    env.close()

    env_tag = env_name.replace("-", "_").lower()
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../logs/holographic_swarm_{env_tag}_log.csv'))
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
