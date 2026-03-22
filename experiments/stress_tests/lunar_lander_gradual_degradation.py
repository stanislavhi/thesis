#!/usr/bin/env python3
"""
Stress Test: Gradual Degradation on LunarLander-v3.

This experiment tests the hypothesis that a Thermodynamic Agent can
recover from gradual, continuous damage over time, acting as a homeostatic
mechanism to resist entropy.

Methodology:
1. Train two agents on LunarLander-v3.
2. From episode 300 onwards, gradually degrade the weights of both agents.
3. Compare the performance:
   - Static Agent: Should slowly succumb to the damage.
   - Thermodynamic Agent: Should detect the "freezing" trend and periodically
     inject chaos to restore functionality, demonstrating homeostasis.
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

def apply_gradual_degradation(policy, decay_rate=0.995):
    """Gradually scales down all weights in the policy network."""
    with torch.no_grad():
        for param in policy.parameters():
            param.mul_(decay_rate)

def run_lunar_lander_trial(seed, use_thermo_injection=True, degrade_start_ep=300, max_episodes=1500):
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
    sigma_history = []

    for episode in range(max_episodes):
        # --- GRADUAL DEGRADATION ---
        if episode >= degrade_start_ep:
            apply_gradual_degradation(agent, decay_rate=0.99) # 1% decay every episode

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
        sigma_history.append(agent.current_sigma)

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
        if use_thermo_injection and episode > 100:
            status = agent.get_thermodynamic_status()
            if status == 'frozen':
                # print(f"   [!] THERMO TRIGGER at ep {episode} (sigma={agent.current_sigma:.4f})")
                agent = injector.mutate(agent)
                optimizer = optim.Adam(agent.parameters(), lr=0.001)
                scores.clear()

    env.close()
    return history, sigma_history

def main():
    print("--- STRESS TEST: LunarLander Gradual Degradation ---")
    seeds = [42, 101, 999]
    thermo_results = []
    static_results = []

    print("\n>>> Running THERMODYNAMIC AGENT trials...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        res, _ = run_lunar_lander_trial(seed, use_thermo_injection=True)
        thermo_results.append(res)

    print("\n>>> Running STATIC AGENT trials...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        res, _ = run_lunar_lander_trial(seed, use_thermo_injection=False)
        static_results.append(res)

    # Analysis
    thermo_mean = np.mean(thermo_results, axis=0)
    static_mean = np.mean(static_results, axis=0)

    # Smooth the curves
    def smooth(y, box_pts=50):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='valid')
        return y_smooth

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # We plot the smoothed versions for clarity
    plt.plot(smooth(thermo_mean), color='blue', linewidth=2, label='Thermodynamic Agent (Homeostatic)')
    plt.plot(smooth(static_mean), color='red', linewidth=2, linestyle='--', label='Static Agent (Decays)')
    
    plt.axvline(x=300, color='black', linestyle=':', linewidth=2, label='Start Gradual Degradation (1%/ep)')
    
    plt.xlabel('Episode')
    plt.ylabel('Score (Smoothed)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.title(f'Homeostasis: Resisting Gradual Entropy on LunarLander-v3 (n={len(seeds)})')
    plt.tight_layout()
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/lunar_lander_gradual_degradation.png'))
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    main()
