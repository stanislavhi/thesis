#!/usr/bin/env python3
"""
Ablation Study: Why does LunarLander fail with Additive Chaos?

Goal: Test alternative, less destructive recovery mechanisms for sensitive environments.

Methodology:
1. Train agents on LunarLander-v3.
2. Inflict 50% brain damage at episode 500.
3. Compare three recovery strategies:
   A. Static (Baseline): No intervention.
   B. Additive Chaos (Current): Add scaled random noise to all weights when frozen.
   C. Targeted Dropout (New): Zero out the lowest-variance neurons when frozen,
      forcing the network to re-route around the "dead wood" without destroying
      healthy weights.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from agents.thermodynamic.thermo_agent import ThermodynamicAgent
from experiments.utils import reinforce_update, inflict_brain_damage, smooth, AblationInjector

def run_trial(seed, strategy, damage_episode=500, max_episodes=1500):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("LunarLander-v3")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = ThermodynamicAgent(input_dim, 64, output_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    
    # Initialize appropriate injector
    injector = None
    if strategy != "static":
        injector = AblationInjector(strategy=strategy)

    scores = deque(maxlen=50)
    history = []

    for episode in range(max_episodes):
        if episode == damage_episode:
            inflict_brain_damage(agent, damage_ratio=0.5)
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

            if done or truncated: break

        total_reward = sum(rewards)
        scores.append(total_reward)
        history.append(total_reward)

        # REINFORCE Update
        reinforce_update(log_probs, rewards, optimizer)

        # Thermodynamic Intervention
        if injector and episode > damage_episode:
            # We only intervene if we aren't recovering naturally
            avg_score = np.mean(scores) if scores else -1000
            if avg_score < -150: # LunarLander failure threshold
                agent = injector.mutate(agent)
                # Note: We do NOT reset the optimizer here. We want to keep momentum
                # and let the network route around the targeted dropout.

    env.close()
    return history

def main():
    print("--- ABLATION STUDY: LunarLander Recovery Mechanisms ---")
    seeds = [42, 101, 999] # Reduced seeds for speed
    
    strategies = {
        "static": [],
        "additive": [],
        "dropout": []
    }

    for strategy in strategies.keys():
        print(f"\n>>> Running Strategy: {strategy.upper()}...")
        for seed in seeds:
            print(f"   Seed {seed}...", flush=True)
            res = run_trial(seed, strategy=strategy)
            strategies[strategy].append(res)

    # Plotting
    plt.figure(figsize=(14, 7))
    
    colors = {"static": "red", "additive": "blue", "dropout": "green"}
    labels = {
        "static": "Static (No Intervention)",
        "additive": "Additive Noise (Current Sledgehammer)",
        "dropout": "Targeted Dropout (Pruning)"
    }
    
    for strategy, results in strategies.items():
        mean_res = np.mean(results, axis=0)
        plt.plot(smooth(mean_res, 50), color=colors[strategy], linewidth=2, label=labels[strategy])

    plt.axvline(x=500, color='black', linestyle=':', linewidth=2, label='50% Brain Damage')
    
    plt.xlabel('Episode')
    plt.ylabel('Score (Smoothed)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.title(f'Ablation: Identifying the Right Thermodynamic Operator for Sensitive Control')
    plt.tight_layout()
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/ablation_lunar_lander.png'))
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    main()
