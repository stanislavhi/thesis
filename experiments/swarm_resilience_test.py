#!/usr/bin/env python3
"""
Experiment: Holographic Swarm Resilience (Collective Self-Healing).

Goal: Demonstrate that a swarm can recover from catastrophic damage to specific
      individual agents within the collective, using thermodynamic self-regulation.

Methodology:
1. The Task: 3 agents (A, B, C) receive inputs X, Y, Z. Task is to predict (X+Y+Z)%10.
2. Training: Train the swarm until it achieves high accuracy.
3. The Catastrophe: At epoch 15, inflict 80% brain damage (weight zeroing) on Agents A and B.
4. Recovery:
   - Static Swarm: Continues with damaged agents, likely failing to recover.
   - Thermodynamic Swarm: Damaged agents detect "freezing" (sigma collapse),
     trigger their internal chaos injectors, and rapidly rewire to restore 
     communication with the aggregator.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agents.self_evolving_llm.holographic_swarm import (
    LLMSwarmAgent,
    HolographicChannel,
    SwarmAggregator
)
from agents.thermodynamic.thermo_injector import ThermodynamicInjector
from core.chaos import LorenzGenerator

def generate_3_agent_data(batch_size=64):
    for _ in range(100): # 100 batches per epoch
        x = torch.randint(0, 10, (batch_size,))
        y = torch.randint(0, 10, (batch_size,))
        z = torch.randint(0, 10, (batch_size,))
        target = (x + y + z) % 10
        
        yield x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1), target

def inflict_damage(agent, damage_ratio=0.8):
    """Zeros out a percentage of weights in the LLMSwarmAgent's brain."""
    with torch.no_grad():
        for param in agent.parameters():
            mask = torch.rand_like(param) > damage_ratio
            param.mul_(mask.float())

def run_swarm_resilience_trial(seed, use_thermo=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Initialize Swarm
    agent_a = LLMSwarmAgent(agent_id='A', vocab_size=10, hidden_dim=32, thought_vector_dim=16)
    agent_b = LLMSwarmAgent(agent_id='B', vocab_size=10, hidden_dim=32, thought_vector_dim=16)
    agent_c = LLMSwarmAgent(agent_id='C', vocab_size=10, hidden_dim=32, thought_vector_dim=16)
    
    agents = [agent_a, agent_b, agent_c]
    channel = HolographicChannel(noise_level=0.2) # Moderate noise
    aggregator = SwarmAggregator(num_agents=3, thought_vector_dim=16, output_dim=10)
    
    all_params = list(agent_a.parameters()) + list(agent_b.parameters()) + \
                 list(agent_c.parameters()) + list(aggregator.parameters())
    
    optimizer = optim.Adam(all_params, lr=0.002)
    criterion = nn.CrossEntropyLoss()
    injector = ThermodynamicInjector(LorenzGenerator(), base_mutation_rate=0.05)
    
    epochs = 40
    damage_epoch = 15
    acc_history = []
    
    for epoch in range(epochs):
        # --- CATASTROPHE ---
        if epoch == damage_epoch:
            print(f"   [Epoch {epoch}] INFLICTING 80% DAMAGE TO AGENTS A AND B...")
            inflict_damage(agent_a, 0.8)
            inflict_damage(agent_b, 0.8)
            # Reset optimizer momentum after massive damage
            optimizer = optim.Adam(all_params, lr=0.002)
            
        total_loss = 0
        correct = 0
        total = 0
        
        for obs_a, obs_b, obs_c, targets in generate_3_agent_data():
            optimizer.zero_grad()
            
            thoughts = [agent(obs) for agent, obs in zip(agents, [obs_a, obs_b, obs_c])]
            noisy_thoughts = channel(thoughts, system_temperature=1.0)
            logits = aggregator(noisy_thoughts)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        accuracy = correct / total
        acc_history.append(accuracy)
        
        print(f"   Epoch {epoch:02d} | Acc: {accuracy:.2%}")
        
        # --- THERMODYNAMIC RECOVERY ---
        if use_thermo and epoch >= damage_epoch:
            for i, agent in enumerate(agents):
                status = agent.get_thermodynamic_status()
                if status == 'frozen':
                    print(f"      [!] Agent {agent.agent_id} FROZEN. Injecting Chaos!")
                    agent.brain = injector.mutate(agent.brain, status='frozen')
                    # Rebuild param list since mutate may return a new module
                    all_params = list(agent_a.parameters()) + list(agent_b.parameters()) + \
                                 list(agent_c.parameters()) + list(aggregator.parameters())
                    optimizer = optim.Adam(all_params, lr=0.002)
                    
    return acc_history

def run_experiment():
    print("--- SWARM RESILIENCE TEST (Collective Healing) ---")
    
    seed = 123
    print("\n>>> Running THERMODYNAMIC SWARM (Self-Healing)...")
    thermo_acc = run_swarm_resilience_trial(seed, use_thermo=True)
    
    print("\n>>> Running STATIC SWARM (No Recovery)...")
    static_acc = run_swarm_resilience_trial(seed, use_thermo=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thermo_acc, label='Thermodynamic Swarm (Recovers)', color='blue', linewidth=2)
    plt.plot(static_acc, label='Static Swarm (Damaged)', color='red', linestyle='--', linewidth=2)
    
    plt.axvline(x=15, color='black', linestyle=':', label='80% Damage to 2/3 Agents')
    
    plt.title('Holographic Swarm: Resilience to Distributed Brain Damage')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (Sum Modulo 10)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/swarm_resilience_test.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    run_experiment()
