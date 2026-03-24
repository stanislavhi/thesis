#!/usr/bin/env python3
"""
Experiment: Autonomous Reasoning Recovery Test (The Homeostasis Test).

Goal: Demonstrate that the EvolvingLLMAgent can autonomously detect and recover from
      "Cognitive Fatigue" (gradual attention collapse) without manual intervention.

Methodology:
1. Initialize an EvolvingLLMAgent.
2. Run a long generation loop (100 steps).
3. Apply "Cognitive Fatigue": At every step, slightly dampen the model's weights (decay by 1%).
   This simulates a natural drift towards a low-energy, "frozen" state (mode collapse).
4. Compare two agents:
   - Static Agent: Should succumb to fatigue, with sigma dropping to zero and output becoming repetitive.
   - Evolving Agent: Should detect the dropping sigma and trigger a chaotic injection to "wake up"
     and restore variance/diversity.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agents.self_evolving_llm.evolving_llm_agent import EvolvingLLMAgent
from agents.thermodynamic.thermo_injector import ThermodynamicInjector
from core.chaos import LorenzGenerator

def apply_cognitive_fatigue(agent, decay_rate=0.95):
    """
    Simulates 'Cognitive Fatigue' or 'Attention Collapse'.
    Gradually scales down weights to simulate loss of signal/variance over time.
    """
    with torch.no_grad():
        for param in agent.model.parameters():
            param.mul_(decay_rate)

def run_homeostasis_trial(seed, use_evolution=True, steps=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize Agent
    agent = EvolvingLLMAgent(vocab_size=1000, hidden_dim=64, num_layers=2)
    
    # Injector for evolution (Gentle nudge)
    injector = ThermodynamicInjector(LorenzGenerator(), base_mutation_rate=0.01)
    
    sigma_history = []
    diversity_history = []
    
    # Dummy input
    input_ids = torch.randint(0, 1000, (1, 10))
    
    print(f"   [Seed {seed}] Starting run (Evolution={'ON' if use_evolution else 'OFF'})...")
    
    for step in range(steps):
        # 1. Forward pass (Reasoning Step)
        logits = agent(input_ids)
        
        # 2. Measure Metrics
        # Output Entropy (Diversity)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        diversity_history.append(entropy)
        
        # Internal Sigma (Work)
        current_sigma = agent.current_sigma
        sigma_history.append(current_sigma)
        
        # 3. Apply Natural Decay (The Challenge)
        # This forces the agent towards a frozen state naturally over time
        apply_cognitive_fatigue(agent, decay_rate=0.90) 
        
        # 4. Self-Diagnosis & Evolution (The Solution)
        if use_evolution:
            status = agent.get_thermodynamic_status()
            
            # We check for 'frozen' status. 
            # Note: The agent's internal history buffer needs to fill up first.
            if status == 'frozen':
                print(f"      [Step {step}] Freezing detected (sigma={current_sigma:.4f}). Injecting Chaos!")
                agent = injector.mutate(agent)
                
    return sigma_history, diversity_history

def main():
    print("--- AUTONOMOUS REASONING RECOVERY TEST ---")
    print("Simulating 'Cognitive Fatigue' and testing Homeostatic Recovery.\n")
    
    seed = 42
    
    # Run Evolving Agent
    evo_sigma, evo_div = run_homeostasis_trial(seed, use_evolution=True)
    
    # Run Static Agent
    static_sigma, static_div = run_homeostasis_trial(seed, use_evolution=False)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Internal Sigma (The "Heartbeat")
    ax1.plot(evo_sigma, color='blue', linewidth=2, label='Evolving Agent (Homeostatic)')
    ax1.plot(static_sigma, color='red', linestyle='--', linewidth=2, label='Static Agent (Decaying)')
    ax1.axhline(y=0.15, color='gray', linestyle=':', label='Freeze Threshold')
    ax1.set_ylabel('Internal Sigma (Activity)')
    ax1.set_title('Internal Thermodynamics: Resisting Entropy Death')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Output Diversity (The "Mind")
    ax2.plot(evo_div, color='blue', linewidth=2, label='Evolving Agent')
    ax2.plot(static_div, color='red', linestyle='--', linewidth=2, label='Static Agent')
    ax2.set_ylabel('Output Entropy (Cognitive Capacity)')
    ax2.set_xlabel('Time Steps')
    ax2.set_title('External Behavior: Sustaining Complexity')
    ax2.grid(True, alpha=0.3)
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/autonomous_reasoning_test.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
    print("\nInterpretation:")
    print("1. Static Agent (Red): Should show a collapse in both Sigma and Entropy as fatigue sets in.")
    print("2. Evolving Agent (Blue): Should show 'heartbeats' - dips followed by spikes as it wakes itself up.")

if __name__ == "__main__":
    main()
