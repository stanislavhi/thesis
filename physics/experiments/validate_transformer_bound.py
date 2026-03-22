#!/usr/bin/env python3
"""
Experiment: Validate the Thermodynamic Bound on a Transformer.

Goal: Test the falsifiable inequality σ² · ε ≥ C_phys on a Transformer model.

Methodology:
1.  ε (Self-Model Error): Measured as the cross-entropy loss of the model's prediction.
    This represents how "wrong" the model's current self-model is.

2.  σ (Entropy Production Rate): Proxied by the L2 norm of the gradients at each layer.
    Gradient norm represents the magnitude of the "work" done by the optimizer to 
    update the representation. High gradient norm = high heat dissipation.

3.  Bound Validation: We compute and plot σ² · ε for each layer.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from qwen.tests.test_qwen_thermodynamic import QwenThermodynamicModel

def validate_bound():
    print("--- THERMODYNAMIC BOUND VALIDATION (TRANSFORMER) ---")

    # 1. Setup Model
    device = torch.device('cpu')
    vocab_size = 1000
    hidden_dim = 256
    num_layers = 12 
    model = QwenThermodynamicModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=num_layers,
        max_seq_len=128
    ).to(device)
    
    # Must be in train mode to get gradients
    model.train() 

    # 2. Prepare Data
    input_ids = torch.randint(0, vocab_size, (4, 64)).to(device) # Batch of 4
    labels = torch.roll(input_ids, -1, dims=1)

    # 3. Measure ε (Self-Model Error)
    logits, _ = model(input_ids)
    loss_fn = nn.CrossEntropyLoss()
    epsilon = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
    
    print(f"Global Self-Model Error (ε): {epsilon.item():.4f}")

    # 4. Measure σ (Entropy Production) via Gradient Norms
    # Trigger backward pass to populate .grad attributes
    model.zero_grad()
    epsilon.backward()

    sigmas = []
    
    # Extract gradient norm for each layer's attention mechanism
    for i in range(num_layers):
        layer_dict = model.layers[i]
        attn_module = layer_dict['attention']
        
        # Sum the L2 norm of all parameters in this attention layer
        layer_grad_norm_sq = 0.0
        for param in attn_module.parameters():
            if param.grad is not None:
                layer_grad_norm_sq += param.grad.data.norm(2).item() ** 2
                
        # sigma = sqrt(sum(grad^2))
        sigma = np.sqrt(layer_grad_norm_sq)
        sigmas.append(sigma)

    print(f"Measured Entropy Production (σ gradient norm proxy) for {len(sigmas)} layers.")

    # 5. Compute and Plot the Bound
    sigmas = np.array(sigmas)
    epsilon_val = epsilon.item()
    bound_values = (sigmas ** 2) * epsilon_val

    # Print Results to Console
    print("\n--- Layer-wise Results ---")
    print(f"{'Layer':<6} | {'σ (Grad Norm)':<15} | {'σ² · ε (Bound)':<15}")
    print("-" * 45)
    for i, (sigma, bound) in enumerate(zip(sigmas, bound_values)):
        print(f"{i+1:<6} | {sigma:<15.4e} | {bound:<15.4e}")
    print("-" * 45)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 7))
    layer_indices = np.arange(1, len(bound_values) + 1)

    color = 'tab:red'
    ax1.set_xlabel('Transformer Layer Depth')
    ax1.set_ylabel('σ² · ε (Bound Value)', color=color)
    ax1.plot(layer_indices, bound_values, color=color, marker='o', linewidth=2, label='σ² · ε')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('σ (Gradient Norm - Work Done)', color=color)
    ax2.plot(layer_indices, sigmas, color=color, linestyle='--', marker='s', alpha=0.7, label='σ (Grad Norm)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Thermodynamic Bound via Gradient Norms (ε = {epsilon_val:.2f})')
    fig.legend(loc='upper left', bbox_to_anchor=(0.3, 0.9))
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/thermodynamic_bound_validation.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
    print("\nNote: σ is proxied by the L2 gradient norm, representing the thermodynamic work of the update.")

if __name__ == "__main__":
    validate_bound()
