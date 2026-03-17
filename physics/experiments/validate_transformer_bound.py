#!/usr/bin/env python3
"""
Experiment: Validate the Thermodynamic Bound on a Transformer.

Goal: Test the falsifiable inequality σ² · ε ≥ C_phys on a Transformer model.

Methodology:
1.  ε (Self-Model Error): Measured as the cross-entropy loss of the model's prediction.
    This represents how "wrong" the model's current self-model is.

2.  σ (Entropy Production Rate): Proxied by the change in representation between layers.
    We measure the Mean Squared Error (MSE) between the input and output of each
    Transformer block. A large change signifies high "work" or entropy production.

3.  Bound Validation: We compute and plot σ² · ε for each layer to see if it
    remains above a constant floor, as the theory predicts.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Fix Python path to allow imports from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the placeholder model from the test file
from qwen.tests.test_qwen_thermodynamic import QwenThermodynamicModel

def validate_bound():
    print("=" * 60)
    print("Running Thermodynamic Bound Validation on Transformer")
    print("=" * 60)

    # 1. Setup Model
    device = torch.device('cpu')
    vocab_size = 1000
    hidden_dim = 256
    num_layers = 12  # Use a deeper model to see the effect across layers
    model = QwenThermodynamicModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=num_layers,
        max_seq_len=128
    ).to(device)
    model.eval()

    # 2. Prepare Data
    # A single sequence is enough for this analysis
    input_ids = torch.randint(0, vocab_size, (1, 64)).to(device)
    
    # The target is the input sequence shifted by one
    labels = torch.roll(input_ids, -1, dims=1)

    # 3. Measure ε (Self-Model Error)
    with torch.no_grad():
        logits, _ = model(input_ids)
        loss_fn = nn.CrossEntropyLoss()
        # Calculate loss on the shifted targets
        epsilon = loss_fn(logits.view(-1, vocab_size), labels.view(-1)).item()
    
    print(f"Global Self-Model Error (ε): {epsilon:.4f}")

    # 4. Measure σ (Entropy Production) per layer
    # We need to capture the hidden state at each layer.
    # Since the model uses ModuleDict and doesn't call layer(), we hook into 'attention'.
    hidden_states = []

    def get_hook():
        def hook(module, input, output):
            # output is the result of the attention mechanism
            hidden_states.append(output.detach())
        return hook

    hooks = []
    # The model structure is model.layers[i]['attention']
    for i in range(num_layers):
        layer_dict = model.layers[i]
        # Hook into the attention module
        hook = layer_dict['attention'].register_forward_hook(get_hook())
        hooks.append(hook)

    # Run a forward pass to trigger hooks
    with torch.no_grad():
        model(input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    if not hidden_states:
        print("Could not measure hidden states. Aborting.")
        return

    print(f"Captured {len(hidden_states)} hidden states.")

    # Calculate sigma for each layer
    # σ is the magnitude of the update (work done) by the layer.
    # Here, we approximate it as the norm of the attention output.
    sigmas = []
    for state in hidden_states:
        # state shape: (batch, seq_len, hidden_dim)
        # We calculate the mean squared magnitude of the update vector
        sigma = torch.mean(state ** 2).item()
        sigmas.append(sigma)

    print(f"Measured Entropy Production (σ) for {len(sigmas)} layers.")

    # 5. Compute and Plot the Bound
    sigmas = np.array(sigmas)
    bound_values = (sigmas ** 2) * epsilon

    # Print Results to Console
    print("\n--- Layer-wise Results ---")
    print(f"{'Layer':<6} | {'σ (MSE)':<10} | {'σ² · ε (Bound)':<15}")
    print("-" * 35)
    for i, (sigma, bound) in enumerate(zip(sigmas, bound_values)):
        print(f"{i+1:<6} | {sigma:.6f}   | {bound:.6f}")
    print("-" * 35)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 7))
    layer_indices = np.arange(1, len(bound_values) + 1)

    # Plot σ² · ε
    color = 'tab:red'
    ax1.set_xlabel('Transformer Layer')
    ax1.set_ylabel('σ² · ε (Bound Value)', color=color)
    ax1.plot(layer_indices, bound_values, color=color, marker='o', label='σ² · ε')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # Plot σ on a secondary axis for context
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('σ (Entropy Production)', color=color)
    ax2.plot(layer_indices, sigmas, color=color, linestyle='--', marker='s', label='σ (MSE)')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add a horizontal line for the global error ε
    # We plot epsilon on the primary axis scale for comparison if it fits, 
    # otherwise just note it in the legend.
    # Here we just add it to the title/text.
    
    plt.title(f'Thermodynamic Bound Validation (ε = {epsilon:.2f})')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'thermodynamic_bound_validation.png'))
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")
    print("\nInterpretation:")
    print("The theory predicts that σ²·ε should remain above a near-constant floor (C_phys).")
    print("If the red curve (σ²·ε) stays flat or rises, the bound holds.")
    print("If it crashes to zero while ε is non-zero, the theory is challenged.")


if __name__ == "__main__":
    validate_bound()
