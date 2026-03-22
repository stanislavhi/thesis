#!/usr/bin/env python3
"""
Prospective Test of the Thermodynamic Operator Selection Rule.

Hypothesis formulated BEFORE the experiment:
- Environment: Acrobot-v1 (Discrete action, momentum-based).
- Architecture: 16 hidden neurons.
- Classification: Low C_V (Heat Capacity).
- Prediction: The Operator Selection Rule predicts that Additive Noise (Global Heat) 
  will successfully induce a phase transition to escape a local minimum, whereas 
  Targeted Dropout (Localized Annealing) will fail due to the lack of redundant capacity.
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

class ProspectiveInjector:
    def __init__(self, strategy="additive", base_rate=0.05):
        self.strategy = strategy
        self.base_rate = base_rate
        
    def mutate(self, agent: ThermodynamicAgent) -> ThermodynamicAgent:
        status = agent.get_thermodynamic_status()
        
        if status == 'frozen':
            if self.strategy == "additive":
                # Global Phase Transition
                magnitude = self.base_rate * 5.0
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

def run_trial(seed, strategy, max_episodes=500):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Acrobot-v1: Reward is -1 per step until reaching the goal. Max steps 500.
    env = gym.make("Acrobot-v1")
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
    
    for episode in range(max_episodes):
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
        # We trigger if the agent is stuck at the max penalty (-500)
        if injector and episode > 50:
            status = agent.get_thermodynamic_status()
            
            if avg_score <= -490 and np.std(scores) < 10.0:
                status = 'frozen'
                
            if status == 'frozen':
                agent = injector.mutate(agent)
                if strategy == "additive":
                    optimizer = optim.Adam(agent.parameters(), lr=0.01) # Reset momentum
                scores.clear()
            
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
    
    def smooth(y, box_pts=20):
        box = np.ones(box_pts)/box_pts
        return np.convolve(y, box, mode='valid')

    colors = {"static": "red", "additive": "blue", "dropout": "green"}
    labels = {
        "static": "Static (Baseline)",
        "additive": "Additive Noise (Predicted to Succeed)",
        "dropout": "Targeted Dropout (Predicted to Fail)"
    }
    
    for strategy, results in strategies.items():
        mean_res = np.mean(results, axis=0)
        plt.plot(smooth(mean_res), color=colors[strategy], linewidth=2, label=labels[strategy])

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
