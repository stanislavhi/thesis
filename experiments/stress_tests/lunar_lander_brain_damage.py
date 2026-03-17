#!/usr/bin/env python3
"""
Stress Test: Neuroplasticity and Resilience on LunarLander-v3.

This experiment tests the core hypothesis: a Thermodynamic Agent, by monitoring
its own internal state (sigma), can recover from catastrophic damage far more
effectively than a standard agent that only relies on external reward signals.

Methodology:
1. Train two agents on LunarLander-v3 for 500 episodes.
   - Agent A: ThermodynamicAgent with physics-driven chaos injection.
   - Agent B: A standard (static) agent with the same architecture.
2. At episode 500, inflict "brain damage" by zeroing out 50% of the weights
   in both agents' networks.
3. Continue training for another 1000 episodes.
4. Plot the performance curves to compare recovery speed and final performance.
   The internal sigma of the ThermodynamicAgent is also plotted to show the
   physical mechanism of its recovery.
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

def inflict_brain_damage(policy, damage_ratio=0.5):
    """Randomly zeroes out a fraction of all weights in the policy network."""
    with torch.no_grad():
        for param in policy.parameters():
            mask = torch.rand_like(param) > damage_ratio
            param.mul_(mask.float())

def run_lunar_lander_trial(seed, use_thermo_injection=True, damage_episode=500, max_episodes=1500):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("LunarLander-v3")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Use the ThermodynamicAgent for both trials for a fair comparison
    agent = ThermodynamicAgent(input_dim, 64, output_dim) # Larger brain for a harder problem
    optimizer = optim.Adam(agent.parameters(), lr=0.001)

    injector = ThermodynamicInjector(LorenzGenerator(), base_mutation_rate=0.05)

    scores = deque(maxlen=50)
    history = []
    sigma_history = []

    for episode in range(max_episodes):
        # --- BRAIN DAMAGE ---
        if episode == damage_episode:
            inflict_brain_damage(agent, damage_ratio=0.5)
            print(f"   [Seed {seed}] BRAIN DAMAGE inflicted at ep {episode}.")
            print(f"   [Debug] Current Sigma: {agent.current_sigma:.6f}") # Debug print
            optimizer = optim.Adam(agent.parameters(), lr=0.001)
            scores.clear()

        state, _ = env.reset()
        log_probs = []
        rewards = []

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = agent(state_tensor) # Forward pass calculates sigma
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
        avg_score = np.mean(scores)
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
        if use_thermo_injection and episode > damage_episode:
            status = agent.get_thermodynamic_status()
            # If the agent is frozen internally, inject chaos to force recovery.
            if status == 'frozen':
                print(f"   [!!!] THERMODYNAMIC TRIGGER at ep {episode}: Agent is FROZEN (sigma={agent.current_sigma:.6f}). Injecting chaos.")
                agent = injector.mutate(agent)
                optimizer = optim.Adam(agent.parameters(), lr=0.001)
                scores.clear()

    env.close()
    return history, sigma_history

def main():
    print("--- STRESS TEST: LunarLander-v3 Brain Damage Resilience ---")
    seeds = [42, 101, 999]
    thermo_results = []
    static_results = []
    thermo_sigmas = []

    print("\n>>> Running THERMODYNAMIC AGENT trials (Physics-Driven Recovery)...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        res, sigmas = run_lunar_lander_trial(seed, use_thermo_injection=True)
        thermo_results.append(res)
        thermo_sigmas.append(sigmas)

    print("\n>>> Running STATIC AGENT trials (No Recovery Mechanism)...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        res, _ = run_lunar_lander_trial(seed, use_thermo_injection=False)
        static_results.append(res)

    # Analysis
    thermo_mean = np.mean(thermo_results, axis=0)
    static_mean = np.mean(static_results, axis=0)
    avg_sigma = np.mean(thermo_sigmas, axis=0)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(14, 7))
    x = np.arange(len(thermo_mean))

    # Plot Scores
    ax1.plot(x, thermo_mean, color='blue', linewidth=2, label='Thermodynamic Agent (Recovers)')
    ax1.plot(x, static_mean, color='red', linewidth=2, label='Static Agent (Damaged)')
    ax1.axvline(x=500, color='black', linestyle='--', linewidth=2, label='Brain Damage (50% zeroed)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot Sigma on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, avg_sigma, color='green', alpha=0.4, label='Thermo Agent Avg. Sigma')
    ax2.set_ylabel('Internal Sigma (Work)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='lower right')

    plt.title(f'Neuroplasticity: Recovery from Brain Damage on LunarLander-v3 (n={len(seeds)} seeds)')
    plt.tight_layout()
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/lunar_lander_brain_damage.png'))
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    main()
