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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.chaos import LorenzGenerator
from agents.rl_policy import EvolvingPolicy, RLChaosInjector

# Environment-specific presets
ENV_PRESETS = {
    "CartPole-v1": {
        "episodes": 500,
        "initial_hidden": 4,
        "solved_threshold": 195,
        "stagnation_score": 100,
        "stagnation_std": 5.0,
        "lr": 0.01,
    },
    "LunarLander-v3": {
        "episodes": 1500,
        "initial_hidden": 32,
        "solved_threshold": 200,
        "stagnation_score": -100,
        "stagnation_std": 20.0,
        "lr": 0.005,
    },
}

DEFAULT_PRESET = {
    "episodes": 1000,
    "initial_hidden": 16,
    "solved_threshold": 999999,  # Never auto-stop
    "stagnation_score": 0,
    "stagnation_std": 10.0,
    "lr": 0.01,
}


def train_rl_agent(env_name, episodes=None):
    # Get preset or use defaults
    preset = ENV_PRESETS.get(env_name, DEFAULT_PRESET)
    if episodes is None:
        episodes = preset["episodes"]

    env = gym.make(env_name)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Start with a small brain to force evolution
    policy = EvolvingPolicy(input_dim, preset["initial_hidden"], output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=preset["lr"])

    chaos_gen = LorenzGenerator()
    injector = RLChaosInjector(chaos_gen)

    # Monitor
    scores = deque(maxlen=50)
    best_avg = -float('inf')

    log_data = []

    print(f"--- THERMODYNAMIC RL: {env_name} ---")
    print(f"Config: hidden={preset['initial_hidden']}, lr={preset['lr']}, "
          f"solved={preset['solved_threshold']}, episodes={episodes}")
    print(f"Starting Evolution...\n", flush=True)

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        # Run Episode
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)

            # Sample action
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

        # Update Policy (REINFORCE)
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
        grad_norm = 0.0
        if policy_loss:
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()

            # Calculate Gradient Norm (Entropy Production Proxy)
            total_norm = 0.0
            for p in policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5

            optimizer.step()

        # Log: Episode, Score, AvgScore, HiddenSize, EntropyProduction (GradNorm)
        log_data.append(f"{episode},{total_reward},{avg_score},{policy.net[0].out_features},{grad_norm}")

        # Track best
        if avg_score > best_avg:
            best_avg = avg_score

        # Thermodynamic Check — mutate if stagnant
        if episode > 20 and avg_score < preset["stagnation_score"] and np.std(scores) < preset["stagnation_std"]:
            print(f"\n[Episode {episode}] Stagnation Detected (Avg Score: {avg_score:.1f}).", flush=True)
            policy = injector.mutate(policy)
            optimizer = optim.Adam(policy.parameters(), lr=preset["lr"])
            scores.clear()

        if episode % 20 == 0:
            print(f"Episode {episode} | Score: {total_reward:.0f} | Avg: {avg_score:.1f} | "
                  f"Hidden: {policy.net[0].out_features} | Heat: {grad_norm:.4f}", flush=True)

        if avg_score > preset["solved_threshold"]:
            print(f"\n>>> SOLVED! Avg score {avg_score:.1f} > {preset['solved_threshold']}", flush=True)
            break

    env.close()

    # Save Log
    env_tag = env_name.replace("-", "_").lower()
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../logs/rl_{env_tag}_log.csv'))
    with open(log_path, "w") as f:
        f.write("episode,score,avg_score,hidden_size,entropy_production\n")
        f.write("\n".join(log_data))
    print(f"\nBest avg score: {best_avg:.1f}")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thermodynamic RL Agent")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gymnasium environment (default: CartPole-v1)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes (default: from preset)")
    args = parser.parse_args()

    train_rl_agent(args.env, args.episodes)
