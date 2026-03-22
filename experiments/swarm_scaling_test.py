#!/usr/bin/env python3
"""
Experiment: Swarm Scaling Test (Smooth Gradient Task).

Goal: Test how the accuracy of the Holographic Swarm scales with the number of agents.

Methodology:
1. The Task: Distributed Average Threshold. N agents each receive a number (0-9). 
   Task is to predict if the global average > 4.5 (Binary Classification).
   * Unlike modulo arithmetic, this task has a smooth gradient. Partial communication
     yields partial success, allowing the swarm to actually learn via gradient descent.
2. Architecture: Variable N LLMSwarmAgents, passing thoughts through a HolographicChannel
   to a SwarmAggregator.
3. Test: Compare final accuracy for N = [2, 5, 10] agents.
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

def generate_n_agent_data(num_agents, batch_size=64):
    for _ in range(100): # 100 batches per epoch
        inputs = []
        total_sum = torch.zeros(batch_size, dtype=torch.float32)
        
        for _ in range(num_agents):
            x = torch.randint(0, 10, (batch_size,))
            inputs.append(x.unsqueeze(1))
            total_sum += x.float()
            
        # Target: 1 if average > 4.5, else 0
        average = total_sum / num_agents
        target = (average > 4.5).float().unsqueeze(1)
        
        yield inputs, target

def run_scaling_trial(num_agents, epochs=15):
    print(f"\n>>> Running Swarm with N={num_agents} Agents...")
    
    # Initialize Swarm
    agents = [LLMSwarmAgent(agent_id=f'A{i}', vocab_size=10, hidden_dim=32, thought_vector_dim=8) for i in range(num_agents)]
    channel = HolographicChannel(noise_level=0.1)
    
    # The aggregator input grows linearly with agents. Output is 1 (Binary)
    aggregator = SwarmAggregator(num_agents=num_agents, thought_vector_dim=8, output_dim=1)
    
    all_params = list(aggregator.parameters())
    for a in agents:
        all_params.extend(list(a.parameters()))
        
    optimizer = optim.Adam(all_params, lr=0.005) # Slightly higher LR for faster convergence
    criterion = nn.BCEWithLogitsLoss()
    
    acc_history = []
    
    for epoch in range(epochs):
        correct = 0
        total = 0
        
        for obs_list, targets in generate_n_agent_data(num_agents):
            optimizer.zero_grad()
            
            thoughts = [agent(obs) for agent, obs in zip(agents, obs_list)]
            noisy_thoughts = channel(thoughts, system_temperature=1.0)
            logits = aggregator(noisy_thoughts)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            # Accuracy calculation for binary classification
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        accuracy = correct / total
        acc_history.append(accuracy)
        print(f"   Epoch {epoch+1:02d} | Acc: {accuracy:.2%}")
        
    return acc_history

def run_experiment():
    print("--- SWARM SCALING TEST (SMOOTH TASK) ---")
    
    agent_counts = [2, 5, 10]
    results = {}
    
    for n in agent_counts:
        acc = run_scaling_trial(n)
        results[n] = acc
        
    # Plotting
    plt.figure(figsize=(10, 6))
    for n, acc in results.items():
        plt.plot(acc, label=f'N={n} Agents', linewidth=2)
        
    plt.title('Holographic Swarm Scaling: Accuracy vs Swarm Size\n(Task: Distributed Average > 4.5)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/swarm_scaling_test.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    run_experiment()
