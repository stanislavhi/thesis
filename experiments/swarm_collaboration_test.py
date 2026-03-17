#!/usr/bin/env python3
"""
Experiment: Holographic Swarm Collaboration Test (Hard Mode).

Goal: Demonstrate that a swarm of agents can collaborate through a noisy channel
      to solve a problem that is IMPOSSIBLE for any single agent to solve.

Methodology:
1. The Task: "Sum Modulo 10".
   - Agent A sees number X (0-9).
   - Agent B sees number Y (0-9).
   - Target: (X + Y) % 10.
   - Why it's hard: Knowing X gives 0 information about the result. The agents MUST
     communicate the actual value of their inputs to the aggregator.
2. The Architecture:
   - Two LLMSwarmAgents process their single-number inputs.
   - Their "thought vectors" pass through a noisy HolographicChannel.
   - A SwarmAggregator fuses the thoughts to predict the class (0-9).
3. The Test: Can the swarm learn an error-correcting communication protocol?
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

def generate_hard_task_data(batch_size=64):
    """Generates (X, Y) pairs and (X+Y)%10 targets."""
    for _ in range(200): # 200 batches per epoch
        x = torch.randint(0, 10, (batch_size,))
        y = torch.randint(0, 10, (batch_size,))
        target = (x + y) % 10
        
        # Agents expect sequence input (batch, seq_len). We fake seq_len=1.
        obs_a = x.unsqueeze(1)
        obs_b = y.unsqueeze(1)
        
        yield obs_a, obs_b, target

def run_swarm_test():
    print("--- HOLOGRAPHIC SWARM: SUM MODULO 10 TASK ---")
    
    # 1. Initialize Swarm Components
    # We use a small vocab size since inputs are just digits 0-9
    agent_a = LLMSwarmAgent(agent_id='A', vocab_size=10, hidden_dim=32, thought_vector_dim=16)
    agent_b = LLMSwarmAgent(agent_id='B', vocab_size=10, hidden_dim=32, thought_vector_dim=16)
    
    # Channel with significant noise to force robust communication
    channel = HolographicChannel(noise_level=0.5)
    
    # Aggregator outputs 10 classes (digits 0-9)
    aggregator = SwarmAggregator(num_agents=2, thought_vector_dim=16, output_dim=10)
    
    all_params = list(agent_a.parameters()) + list(agent_b.parameters()) + list(aggregator.parameters())
    optimizer = optim.Adam(all_params, lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    # 2. Training Loop
    epochs = 20 # More epochs needed for this harder task
    loss_history = []
    acc_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        for obs_a, obs_b, targets in generate_hard_task_data():
            optimizer.zero_grad()
            
            thought_a = agent_a(obs_a)
            thought_b = agent_b(obs_b)
            
            # Add noise
            noisy_thoughts = channel([thought_a, thought_b], system_temperature=1.0)
            
            # Predict
            logits = aggregator(noisy_thoughts)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        accuracy = correct / total
        loss_history.append(avg_loss)
        acc_history.append(accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}")
        
    # 3. Plot Results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(loss_history, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(acc_history, color=color, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Holographic Swarm: Sum Modulo 10 (Hard Collaboration)')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/swarm_collaboration_test.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
    print("\nInterpretation:")
    print("Chance accuracy is 10%. If the swarm reaches >90%, it has successfully")
    print("learned to transmit information through the noisy holographic channel.")

if __name__ == "__main__":
    run_swarm_test()
