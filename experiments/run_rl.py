import sys
import os
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

def train_rl_agent():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    # Start with a TINY brain to force evolution
    policy = EvolvingPolicy(input_dim, 4, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    
    chaos_gen = LorenzGenerator()
    injector = RLChaosInjector(chaos_gen)
    
    # Monitor
    scores = deque(maxlen=50)
    best_score = 0
    
    episodes = 500
    log_data = []
    
    print(f"Starting Evolution on {env_name}...", flush=True)
    
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
        
        # Thermodynamic Check
        # If score is low and stagnant, mutate
        if episode > 20 and avg_score < 100 and np.std(scores) < 5.0:
            print(f"\n[Episode {episode}] Stagnation Detected (Avg Score: {avg_score:.1f}).", flush=True)
            policy = injector.mutate(policy)
            optimizer = optim.Adam(policy.parameters(), lr=0.01)
            scores.clear() # Reset monitor
            
        if episode % 20 == 0:
            print(f"Episode {episode} | Score: {total_reward:.0f} | Avg: {avg_score:.1f} | Hidden: {policy.net[0].out_features} | Heat: {grad_norm:.4f}", flush=True)
            
        if avg_score > 195: # CartPole solved
            print(f"\n>>> SOLVED! The swarm has evolved to mastery.", flush=True)
            break

    env.close()
    
    # Save Log
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/rl_training_log.csv'))
    with open(log_path, "w") as f:
        f.write("episode,score,avg_score,hidden_size,entropy_production\n")
        f.write("\n".join(log_data))
    print(f"Log saved to {log_path}")

if __name__ == "__main__":
    train_rl_agent()
