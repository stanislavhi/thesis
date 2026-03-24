#!/usr/bin/env python3
"""
Prospective Test of the Thermodynamic Operator Selection Rule.

Hypothesis formulated BEFORE the experiment:
- Environment: Acrobot-v1 (Discrete action, momentum-based).
- Architecture: 16 hidden neurons.
- Classification: Low C_V (Heat Capacity).
- Prediction: The Operator Selection Rule predicts that Additive Noise (Global Heat) 
  will successfully induce a phase transition to escape a local minimum (induced by
  an environment shift), whereas Targeted Dropout (Localized Annealing) will fail 
  due to the lack of redundant capacity.
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
from experiments.utils import reinforce_update, smooth, InvertibleEnv

class InvertibleAcrobot(InvertibleEnv):
    """Acrobot-specific alias — InvertibleEnv handles 3-action mapping automatically."""
    pass

class ProspectiveInjector:
    def __init__(self, strategy="additive", base_rate=0.1):
        self.strategy = strategy
        self.base_rate = base_rate
        
    def mutate(self, agent: ThermodynamicAgent) -> ThermodynamicAgent:
        status = agent.get_thermodynamic_status()
        
        if status == 'frozen':
            if self.strategy == "additive":
                # Global Phase Transition for low C_V
                magnitude = self.base_rate * 2.0
                with torch.no_grad():
                    for param in agent.parameters():
                        noise = torch.randn_like(param) * magnitude
                        param.add_(noise)
                        
            elif self.strategy == "dropout":
                # Localized Annealing
                with torch.no_grad():
                    weights = agent.layer1.weight
                    variances = torch.var(weights, dim=1)
                    k = max(1, int(weights.shape[0] * 0.2)) # Drop bottom 20%
                    _, indices = torch.topk(variances, k, largest=False)
                    weights[indices] = 0.0
                    if agent.layer1.bias is not None:
                        agent.layer1.bias[indices] = 0.0
        
        return agent

def run_trial(seed, strategy, max_episodes=800, shift_episode=250):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    base_env = gym.make("Acrobot-v1")
    env = InvertibleAcrobot(base_env)
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    # 16 Neurons -> Low C_V
    agent = ThermodynamicAgent(input_dim, 16, output_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.01)
    
    injector = None
    if strategy != "static":
        injector = ProspectiveInjector(strategy=strategy, base_rate=0.1)
    
    scores = deque(maxlen=20)
    history = []
    last_injection_ep = -100
    injection_cooldown = 40  # Give gradient descent time to learn from the perturbation

    for episode in range(max_episodes):
        if episode == shift_episode:
            env.invert()
            scores.clear()
            agent.sigma_history = []
            last_injection_ep = episode  # Treat shift as a disruption
            
        state, _ = env.reset(seed=seed + episode)
        log_probs = []
        rewards = []
        
        crashed = False
        try:
            while True:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                probs = agent(state_tensor)
                
                if torch.isnan(probs).any():
                    crashed = True
                    break 
                    
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                
                next_state, reward, done, truncated, _ = env.step(action.item())
                
                log_probs.append(dist.log_prob(action))
                rewards.append(reward)
                state = next_state
                
                if done or truncated: break
        except Exception as e:
            crashed = True
                
        total_reward = sum(rewards)
        
        if crashed:
            total_reward = -500
            # Gentle re-init to escape NaN death spiral
            optimizer = optim.Adam(agent.parameters(), lr=0.01)
            with torch.no_grad():
                for param in agent.parameters():
                    param.normal_(0, 0.1)
            
        scores.append(total_reward)
        avg_score = np.mean(scores)
        history.append(total_reward)
        
        if not log_probs or crashed:
            continue

        # REINFORCE Update (with gradient clipping to prevent explosion)
        reinforce_update(log_probs, rewards, optimizer, clip_grad=1.0)
            
        # Thermodynamic Intervention
        if injector and episode > 20 and (episode - last_injection_ep) >= injection_cooldown:
            status = agent.get_thermodynamic_status()

            # Acrobot specific: if stuck at bottom, reward is -500.
            if avg_score <= -490 and np.std(scores) < 10.0 and len(scores) == 20:
                status = 'frozen'

            if status == 'frozen':
                agent = injector.mutate(agent)
                if strategy == "additive":
                    optimizer = optim.Adam(agent.parameters(), lr=0.02)
                scores.clear()
                last_injection_ep = episode
            elif avg_score > -400:
                # Agent is recovering — back off and let gradient descent consolidate
                last_injection_ep = episode
            
    env.close()
    return history

def main():
    print("--- PROSPECTIVE TEST: Acrobot-v1 (Low C_V System) ---")
    print("Prediction: Additive Noise will succeed. Targeted Dropout will fail.")
    
    seeds = [42, 101, 999]
    
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
    plt.figure(figsize=(12, 6))
    
    styles = {
        "static":   {"color": "red",   "linestyle": "--", "linewidth": 2.5, "alpha": 0.9},
        "additive": {"color": "blue",  "linestyle": "-",  "linewidth": 2.5, "alpha": 0.9},
        "dropout":  {"color": "green", "linestyle": ":",  "linewidth": 2.5, "alpha": 0.9},
    }
    labels = {
        "static": "Static (Baseline)",
        "additive": "Additive Noise (Predicted to Succeed)",
        "dropout": "Targeted Dropout (Predicted to Fail)"
    }

    for strategy, results in strategies.items():
        if not results: continue

        min_len = min(len(r) for r in results)
        truncated_results = [r[:min_len] for r in results]

        mean_res = np.mean(truncated_results, axis=0)

        plt.plot(smooth(mean_res), label=labels[strategy], **styles[strategy])

    plt.axvline(x=250, color='black', linestyle=':', linewidth=2, label='Environment Shift (Actions Inverted)')

    plt.xlabel('Episode')
    plt.ylabel('Score (Smoothed) - Higher is better')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.title(f'Prospective Operator Test: Acrobot-v1 (Low $C_V$)')
    plt.tight_layout()
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/prospective_operator_test.png'))
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    main()
