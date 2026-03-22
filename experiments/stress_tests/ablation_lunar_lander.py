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

# We need a custom Mutator for this ablation
class AblationInjector:
    def __init__(self, strategy="additive", base_rate=0.05):
        self.strategy = strategy
        self.base_rate = base_rate
        
    def mutate(self, agent: ThermodynamicAgent) -> ThermodynamicAgent:
        status = agent.get_thermodynamic_status()
        
        if status == 'frozen':
            # print(f"   [{self.strategy.upper()}] Triggered at ep.")
            
            if self.strategy == "additive":
                # The old "Sledgehammer" approach
                magnitude = self.base_rate * 5.0
                with torch.no_grad():
                    for param in agent.parameters():
                        noise = torch.randn_like(param) * magnitude
                        param.add_(noise)
                        
            elif self.strategy == "dropout":
                # The "Pruning/Re-routing" approach
                # Find the neurons with the lowest variance and zero their incoming weights
                with torch.no_grad():
                    # Look at Layer 1 weights (hidden_dim, input_dim)
                    weights = agent.layer1.weight
                    # Calculate variance of weights going into each hidden neuron
                    variances = torch.var(weights, dim=1)
                    
                    # Find bottom 20% of neurons
                    k = int(weights.shape[0] * 0.2)
                    _, indices = torch.topk(variances, k, largest=False)
                    
                    # Zero out the weights for these "dead" neurons to force new connections
                    weights[indices] = 0.0
                    if agent.layer1.bias is not None:
                        agent.layer1.bias[indices] = 0.0
        
        return agent

def inflict_damage(policy, damage_ratio=0.5):
    with torch.no_grad():
        for param in policy.parameters():
            mask = torch.rand_like(param) > damage_ratio
            param.mul_(mask.float())

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
            inflict_damage(agent, damage_ratio=0.5)
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
    
    def smooth(y, box_pts=50):
        box = np.ones(box_pts)/box_pts
        return np.convolve(y, box, mode='valid')

    colors = {"static": "red", "additive": "blue", "dropout": "green"}
    labels = {
        "static": "Static (No Intervention)",
        "additive": "Additive Noise (Current Sledgehammer)",
        "dropout": "Targeted Dropout (Pruning)"
    }
    
    for strategy, results in strategies.items():
        mean_res = np.mean(results, axis=0)
        plt.plot(smooth(mean_res), color=colors[strategy], linewidth=2, label=labels[strategy])

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
