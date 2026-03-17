#!/usr/bin/env python3
"""
Experiment: Self-Evolving Language Agent Recovery Test.

Goal: Demonstrate that an LLM agent can detect its own reasoning block (low entropy/sigma)
      and trigger a chaotic mutation to recover, while a static agent remains stuck.

Methodology:
1. Initialize an EvolvingLLMAgent with a Qwen-based brain.
2. Simulate a "Reasoning Block" by degrading weights to induce repetitive loops.
3. Run generation loop:
   - Agent monitors internal sigma.
   - If sigma < threshold (frozen), trigger mutation.
4. Compare recovery speed and text diversity against a static baseline.
"""

import sys
import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agents.self_evolving_llm.evolving_llm_agent import EvolvingLLMAgent
from agents.thermodynamic.thermo_injector import ThermodynamicInjector
from core.chaos import LorenzGenerator

def simulate_reasoning_block(agent):
    """
    Degrade weights to simulate a model stuck in a repetitive loop.
    We scale down weights to reduce variance (sigma) and induce 'freezing'.
    """
    with torch.no_grad():
        for param in agent.model.parameters():
            param.mul_(0.1) # Scale down to 10%

def run_recovery_trial(seed, use_evolution=True, steps=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize Agent
    agent = EvolvingLLMAgent(vocab_size=1000, hidden_dim=64, num_layers=2)
    
    # Injector for evolution
    # CRITICAL FIX 2: Drastically lower mutation rate for Transformers.
    # 0.05 was still too high. 0.005 is a gentle nudge.
    injector = ThermodynamicInjector(LorenzGenerator(), base_mutation_rate=0.005)
    
    # Simulate Block
    simulate_reasoning_block(agent)
    print(f"   [Seed {seed}] Reasoning Block Simulated (Weights Scaled Down).")
    
    sigma_history = []
    diversity_history = [] # Entropy of output distribution
    
    # Dummy input for generation loop
    input_ids = torch.randint(0, 1000, (1, 10))
    
    for step in range(steps):
        # Forward pass
        logits = agent(input_ids)
        
        # Measure Output Diversity (Entropy)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        diversity_history.append(entropy)
        
        # Measure Internal Work (Sigma)
        current_sigma = agent.current_sigma
        sigma_history.append(current_sigma)
        
        # Self-Diagnosis & Evolution
        if use_evolution:
            status = agent.get_thermodynamic_status()
            if status == 'frozen':
                print(f"   [!!!] THERMODYNAMIC TRIGGER at step {step}: Agent is FROZEN (sigma={current_sigma:.4f}). Injecting chaos.")
                agent = injector.mutate(agent)
                
    return sigma_history, diversity_history

def main():
    print("--- SELF-EVOLVING LLM RECOVERY TEST ---")
    seeds = [42]
    
    evolving_sigmas = []
    evolving_diversity = []
    static_sigmas = []
    static_diversity = []
    
    print("\n>>> Running EVOLVING AGENT (Self-Healing)...")
    for seed in seeds:
        sigmas, diversity = run_recovery_trial(seed, use_evolution=True)
        evolving_sigmas.append(sigmas)
        evolving_diversity.append(diversity)
        
    print("\n>>> Running STATIC AGENT (Baseline)...")
    for seed in seeds:
        sigmas, diversity = run_recovery_trial(seed, use_evolution=False)
        static_sigmas.append(sigmas)
        static_diversity.append(diversity)
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot Internal Sigma (Work)
    ax1.plot(evolving_sigmas[0], color='blue', label='Evolving Agent (Recovers)')
    ax1.plot(static_sigmas[0], color='red', linestyle='--', label='Static Agent (Stuck)')
    ax1.set_ylabel('Internal Sigma (Work)')
    ax1.set_title('Internal Thermodynamics: Recovery from "Brain Freeze"')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Output Diversity (Entropy)
    ax2.plot(evolving_diversity[0], color='blue', label='Evolving Agent')
    ax2.plot(static_diversity[0], color='red', linestyle='--', label='Static Agent')
    ax2.set_ylabel('Output Entropy (Diversity)')
    ax2.set_xlabel('Generation Step')
    ax2.set_title('External Behavior: Restoration of Creativity')
    ax2.grid(True, alpha=0.3)
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/reasoning_recovery_test.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    main()
