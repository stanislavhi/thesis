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

from core.chaos import LorenzGenerator
from agents.rl_policy import EvolvingPolicy, RLChaosInjector
from experiments.utils import reinforce_update, InvertibleEnv

def run_trial(seed, use_chaos=True, env_name="CartPole-v1", max_episodes=400):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    base_env = gym.make(env_name)
    env = InvertibleEnv(base_env)
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    policy = EvolvingPolicy(input_dim, 16, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    
    chaos_gen = LorenzGenerator()
    injector = RLChaosInjector(chaos_gen)
    
    scores = deque(maxlen=20)
    history = []
    
    for episode in range(max_episodes):
        if episode == 150:
            env.invert()
            scores.clear()
        
        state, _ = env.reset(seed=seed + episode)
        log_probs = []
        rewards = []
        
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
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
        history.append(total_reward)
        
        # Update Policy
        reinforce_update(log_probs, rewards, optimizer)
            
        # Chaos Injection
        if use_chaos and episode > 20 and avg_score < 50:
            if np.std(scores) < 10.0 or episode == 155: 
                policy = injector.mutate(policy)
                optimizer = optim.Adam(policy.parameters(), lr=0.01)
                scores.clear()
            
    env.close()
    return history

def test_robustness():
    print("--- ROBUSTNESS TEST: ENVIRONMENTAL SHIFT (5 SEEDS) ---")
    
    seeds = [42, 101, 999, 123, 777] 
    chaos_results = []
    static_results = []
    
    print("\n>>> Running CHAOS Trials...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        res = run_trial(seed, use_chaos=True)
        chaos_results.append(res)

    print("\n>>> Running STATIC Trials...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        res = run_trial(seed, use_chaos=False)
        static_results.append(res)

    # Statistical Analysis
    chaos_matrix = np.array(chaos_results)
    static_matrix = np.array(static_results)
    
    chaos_mean = np.mean(chaos_matrix, axis=0)
    chaos_std = np.std(chaos_matrix, axis=0)
    
    static_mean = np.mean(static_matrix, axis=0)
    static_std = np.std(static_matrix, axis=0)
    
    # Recovery Analysis (Post-Shift)
    post_shift_chaos = chaos_matrix[:, 150:]
    post_shift_static = static_matrix[:, 150:]
    
    chaos_recovery = np.mean(post_shift_chaos)
    static_recovery = np.mean(post_shift_static)
    
    print("\n--- RESULTS ---")
    print(f"Chaos Avg Recovery Score: {chaos_recovery:.2f}")
    print(f"Static Avg Recovery Score: {static_recovery:.2f}")
    
    # Plotting with Error Bands
    plt.figure(figsize=(12, 6))
    x = np.arange(len(chaos_mean))
    
    plt.plot(x, chaos_mean, color='blue', label='Chaos (Adaptive)')
    plt.fill_between(x, chaos_mean - chaos_std, chaos_mean + chaos_std, color='blue', alpha=0.2)
    
    plt.plot(x, static_mean, color='red', label='Static (Fixed)')
    plt.fill_between(x, static_mean - static_std, static_mean + static_std, color='red', alpha=0.2)
    
    plt.axvline(x=150, color='black', linestyle='--', label='Environment Shift')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'Thermodynamic Adaptability (n={len(seeds)})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/robustness_test.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    test_robustness()
