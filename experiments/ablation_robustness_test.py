#!/usr/bin/env python3
"""
Ablation Study: Why does CartPole succeed with Additive Chaos?

Goal: Run the same targeted dropout vs additive noise ablation on the CartPole
      environmental shift test to confirm our hypothesis about operator sensitivity.

Methodology:
1. Train agents on CartPole.
2. Invert environment at episode 150.
3. Compare three recovery strategies:
   A. Static (Baseline): No intervention.
   B. Additive Chaos (Sledgehammer): Current working mechanism for CartPole.
   C. Targeted Dropout (Pruning): The surgical mechanism that worked for LunarLander.
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agents.thermodynamic.thermo_agent import ThermodynamicAgent
from experiments.utils import reinforce_update, smooth, InvertibleEnv, AblationInjector

def run_trial(seed, strategy, env_name="CartPole-v1", max_episodes=400):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    base_env = gym.make(env_name)
    env = InvertibleEnv(base_env)
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    # CartPole needs a smaller brain
    agent = ThermodynamicAgent(input_dim, 16, output_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.01)
    
    injector = None
    if strategy != "static":
        # Additive needs higher base rate for this specific network
        rate = 0.1 if strategy == "additive" else 0.05
        injector = AblationInjector(strategy=strategy, base_rate=rate)
    
    scores = deque(maxlen=20)
    history = []
    
    for episode in range(max_episodes):
        if episode == 150:
            env.invert()
            scores.clear()
            # Force a status reset check
            if hasattr(agent, 'sigma_history'):
                agent.sigma_history = []
        
        state, _ = env.reset(seed=seed + episode)
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
        avg_score = np.mean(scores)
        history.append(total_reward)
        
        # REINFORCE Update
        reinforce_update(log_probs, rewards, optimizer)
            
        # Thermodynamic Intervention
        if injector and episode > 20:
            status = agent.get_thermodynamic_status()
            
            # CartPole specific: sometimes it learns a bad policy that has variance
            # but gets 9 reward. We force a 'frozen' state if score is very low and stable.
            if avg_score < 50 and np.std(scores) < 10.0 and len(scores) == 20:
                status = 'frozen'
                
            if status == 'frozen':
                agent = injector.mutate(agent)
                if strategy == "additive":
                    optimizer = optim.Adam(agent.parameters(), lr=0.01)
                scores.clear()
            
    env.close()
    return history

def main():
    print("--- ABLATION STUDY: CartPole Environment Shift ---")
    seeds = [42, 101, 999, 123, 777]
    
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
        "additive": "Additive Noise (Works on CartPole)",
        "dropout": "Targeted Dropout (New)"
    }
    
    for strategy, results in strategies.items():
        mean_res = np.mean(results, axis=0)
        plt.plot(smooth(mean_res), color=colors[strategy], linewidth=2, label=labels[strategy])

    plt.axvline(x=150, color='black', linestyle=':', linewidth=2, label='Environment Shift (Inverted)')
    
    plt.xlabel('Episode')
    plt.ylabel('Score (Smoothed)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.title(f'Ablation: Thermodynamic Operator Sensitivity on CartPole (n={len(seeds)})')
    plt.tight_layout()
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/ablation_cartpole.png'))
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    main()
