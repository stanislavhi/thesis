import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.chaos import LorenzGenerator
from core.monitor import ArchitectureMonitor
from agents.swarm import SwarmAgent, HolographicChannel, HolographicAggregator, ChaosInjector

def train_holographic_swarm():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    
    # CartPole State: [Cart Pos, Cart Vel, Pole Angle, Pole Vel]
    # Agent A sees [0, 1]
    # Agent B sees [2, 3]
    
    # Initialize Swarm
    # We use a small thought vector (size 2) to force compression
    thought_size = 2
    
    agent_a = SwarmAgent(2, 8, thought_size, "Position_Specialist")
    agent_b = SwarmAgent(2, 8, thought_size, "Angle_Specialist")
    
    # Channel & Aggregator
    # Channel adds noise based on loss
    channel = HolographicChannel(bekenstein_bits=1e70) # Sets bandwidth
    
    # Aggregator takes 2 agents * thought_size
    aggregator = HolographicAggregator(2 * thought_size, 16)
    
    # Optimizer
    params = list(agent_a.parameters()) + list(agent_b.parameters()) + \
             list(aggregator.parameters()) + list(channel.parameters())
    optimizer = optim.Adam(params, lr=0.01)
    
    # Chaos
    chaos_gen = LorenzGenerator()
    injector = ChaosInjector(chaos_gen)
    
    scores = deque(maxlen=50)
    episodes = 1000
    
    print("--- HOLOGRAPHIC SWARM: BLIND CARTPOLE ---")
    print("Agent A sees Position. Agent B sees Angle.")
    print("They must communicate to survive.\n")
    
    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        while True:
            # Split State
            state_a = torch.FloatTensor(state[0:2]).unsqueeze(0) # Pos, Vel
            state_b = torch.FloatTensor(state[2:4]).unsqueeze(0) # Angle, AngVel
            
            # Agents Think
            thought_a = agent_a(state_a)
            thought_b = agent_b(state_b)
            
            # Communicate via Noisy Channel
            # Noise scales with inverse performance (heuristic: 100/avg_score)
            avg_score = np.mean(scores) if len(scores) > 0 else 10.0
            noise_level = 10.0 / (avg_score + 1e-9)
            
            noisy_thoughts = channel([thought_a, thought_b], noise_level)
            
            # Aggregator Decides
            # Aggregator outputs a single value (logit) for Left/Right
            # We need to map this to a probability distribution
            logits = aggregator(noisy_thoughts)
            
            # Since Aggregator output is linear(hidden, 1), we need to adapt it for 2 actions
            # Let's assume the Aggregator output is the log-odd of taking action 1
            # Or better, let's fix the Aggregator to output 2 values in the definition, 
            # but here we are using the generic one which outputs 1.
            # Hack: We'll use a small adapter layer here or just project it.
            # Actually, let's just use the output as a score for Action 1 vs Action 0
            
            # To fix this properly, we should update the Aggregator definition or wrap it.
            # For now, let's assume output is prob of Action 1 (Sigmoid)
            prob_1 = torch.sigmoid(logits)
            probs = torch.cat([1 - prob_1, prob_1], dim=1)
            
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
            
        # Chaos Injection
        if episode > 10 and avg_score < 100 and np.std(scores) < 5.0:
            print(f"\n[Episode {episode}] Swarm Stagnation. Mutating...", flush=True)
            # Mutate Agents
            agents = [agent_a, agent_b]
            new_agents, new_agg = injector.mutate_swarm(agents, aggregator)
            agent_a, agent_b = new_agents
            aggregator = new_agg
            
            # Re-init optimizer
            params = list(agent_a.parameters()) + list(agent_b.parameters()) + \
                     list(aggregator.parameters()) + list(channel.parameters())
            optimizer = optim.Adam(params, lr=0.01)
            scores.clear()
            
        if episode % 20 == 0:
            print(f"Episode {episode} | Score: {total_reward:.0f} | Avg: {avg_score:.1f}", flush=True)
            
        if avg_score > 195:
            print(f"\n>>> HOLOGRAPHIC SWARM SOLVED THE TASK!", flush=True)
            break
            
    env.close()

if __name__ == "__main__":
    train_holographic_swarm()
