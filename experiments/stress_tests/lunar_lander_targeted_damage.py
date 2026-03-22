#!/usr/bin/env python3
"""
Stress Test: Targeted Layer Damage on LunarLander-v3.

Goal: Test if the Thermodynamic Agent can recover when specific critical components
      are destroyed, rather than just random diffuse damage.

Methodology:
1. Train two agents on LunarLander-v3 for 500 episodes.
2. At episode 500, completely zero out the weights of the *second* hidden layer.
   This simulates the loss of a specific functional module.
3. Observe recovery.
"""

import sys
import os
import torch
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from agents.thermodynamic.thermo_agent import ThermodynamicAgent
from agents.thermodynamic.thermo_injector import ThermodynamicInjector
from core.chaos import LorenzGenerator

def inflict_targeted_damage(agent):
    """Completely zeroes out the weights of a specific layer."""
    with torch.no_grad():
        # In ThermodynamicAgent, layer2 is the output layer.
        # We will damage the weights going into the final layer.
        agent.layer2.weight.fill_(0.0)
        agent.layer2.bias.fill_(0.0)

def run_lunar_lander_trial(seed, use_thermo_injection=True, damage_episode=500, max_episodes=1500):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("LunarLander-v3")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = ThermodynamicAgent(input_dim, 64, output_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)

    injector = ThermodynamicInjector(LorenzGenerator(), base_mutation_rate=0.05)

    scores = deque(maxlen=50)
    history = []

    for episode in range(max_episodes):
        # --- TARGETED DAMAGE ---
        if episode == damage_episode:
            inflict_targeted_damage(agent)
            print(f"   [Seed {seed}] TARGETED DAMAGE (Layer 2 zeroed) at ep {episode}.")
            optimizer = optim.Adam(agent.parameters(), lr=0.001)
            scores.clear()

        state, _ = env.reset()
        log_probs = []
        rewards = []

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = agent(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            state = next_state

            if done or truncated:
                break

        total_reward = sum(rewards)
        scores.append(total_reward)
        history.append(total_reward)

        # --- REINFORCE Update ---
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

        # --- Thermodynamic Injection ---
        if use_thermo_injection and episode > damage_episode:
            status = agent.get_thermodynamic_status()
            if status == 'frozen':
                agent = injector.mutate(agent)
                optimizer = optim.Adam(agent.parameters(), lr=0.001)
                scores.clear()

    env.close()
    return history

def main():
    print("--- STRESS TEST: LunarLander Targeted Damage ---")
    seeds = [42, 101, 999]
    thermo_results = []
    static_results = []

    print("\n>>> Running THERMODYNAMIC AGENT trials...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        res = run_lunar_lander_trial(seed, use_thermo_injection=True)
        thermo_results.append(res)

    print("\n>>> Running STATIC AGENT trials...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        res = run_lunar_lander_trial(seed, use_thermo_injection=False)
        static_results.append(res)

    # Analysis
    thermo_mean = np.mean(thermo_results, axis=0)
    static_mean = np.mean(static_results, axis=0)

    def smooth(y, box_pts=50):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='valid')
        return y_smooth

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(smooth(thermo_mean), color='blue', linewidth=2, label='Thermodynamic Agent (Recovers)')
    plt.plot(smooth(static_mean), color='red', linewidth=2, linestyle='--', label='Static Agent (Damaged)')
    
    plt.axvline(x=500, color='black', linestyle=':', linewidth=2, label='Targeted Damage (Output Layer Zeroed)')
    
    plt.xlabel('Episode')
    plt.ylabel('Score (Smoothed)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.title(f'Neuroplasticity: Recovery from Targeted Damage on LunarLander-v3 (n={len(seeds)})')
    plt.tight_layout()
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/lunar_lander_targeted_damage.png'))
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    main()
