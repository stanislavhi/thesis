#!/usr/bin/env python3
"""
Experiment: Thermodynamic Sampling vs Standard Sampling.

Goal: Compare the trade-off between diversity (entropy) and coherence (perplexity)
      for standard sampling methods vs. thermodynamic sampling.

Metrics:
1. Perplexity (lower is better coherence)
2. Entropy (higher is better diversity)
3. Repetition Rate (lower is better)
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict

# Fix Python path to allow imports from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import our thermodynamic components
from qwen.tests.test_qwen_thermodynamic import QwenThermodynamicModel
from qwen.inference.qwen_thermodynamic_inferencer import (
    QwenThermodynamicInferencer, 
    InferenceConfig,
    ThermodynamicSampler
)

@dataclass
class ExperimentResult:
    method_name: str
    perplexity: float
    entropy: float
    repetition_rate: float
    generation_time: float

def calculate_perplexity(logits, target_ids):
    """Calculate perplexity of a sequence given logits."""
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
    return torch.exp(loss).item()

def calculate_repetition_rate(tokens):
    """Calculate the ratio of repeated n-grams (n=3)."""
    tokens = tokens.tolist()
    if len(tokens) < 4: return 0.0
    
    ngrams = set()
    count = 0
    for i in range(len(tokens) - 3):
        ngram = tuple(tokens[i:i+3])
        if ngram in ngrams:
            count += 1
        ngrams.add(ngram)
    
    return count / (len(tokens) - 3)

def run_comparison():
    print("=" * 60)
    print("Running Sampling Comparison Experiment")
    print("=" * 60)

    # 1. Setup Model
    device = torch.device('cpu') # Use CPU for this demo
    vocab_size = 1000
    hidden_dim = 256
    model = QwenThermodynamicModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=4,
        max_seq_len=128
    ).to(device)
    model.eval()

    # 2. Define Prompts (Dummy for now)
    prompts = [torch.randint(0, vocab_size, (1, 10)).to(device) for _ in range(5)]
    
    results = []

    # 3. Define Methods to Compare
    # IMPORTANT: Set max_length explicitly to avoid generating 2048 tokens!
    common_max_len = 64
    
    methods = [
        ("Greedy", InferenceConfig(temperature=1.0, top_k=1, top_p=1.0, max_length=common_max_len)),
        ("Standard (T=1.0)", InferenceConfig(temperature=1.0, top_k=50, top_p=0.9, max_length=common_max_len)),
        ("Standard (T=0.7)", InferenceConfig(temperature=0.7, top_k=50, top_p=0.9, max_length=common_max_len)),
        ("Thermodynamic (Adaptive)", InferenceConfig(temperature=1.0, top_k=50, top_p=0.9, entropy_threshold=2.5, max_length=common_max_len))
    ]

    # 4. Run Experiment
    for name, config in methods:
        print(f"\nTesting Method: {name}")
        
        perplexities = []
        entropies = []
        repetition_rates = []
        
        # Use appropriate sampler
        if "Thermodynamic" in name:
            sampler = ThermodynamicSampler(model, config)
            # Note: sampler.sample uses config.max_length internally
            generate_fn = lambda p: sampler.sample(p, temperature_schedule=True)
        else:
            inferencer = QwenThermodynamicInferencer(model, config)
            generate_fn = lambda p: inferencer.generate(p, max_length=common_max_len)

        for i, prompt in enumerate(prompts):
            print(f"  Prompt {i+1}/5...", end='\r')
            # Generate
            with torch.no_grad():
                output, diagnostics = generate_fn(prompt)
            
            # Calculate Metrics
            with torch.no_grad():
                logits, _ = model(output)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = output[..., 1:].contiguous()
                ppl = calculate_perplexity(shift_logits, shift_labels)
            
            # Entropy (avg over sequence)
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
            
            # Repetition
            rep_rate = calculate_repetition_rate(output[0])
            
            perplexities.append(ppl)
            entropies.append(entropy)
            repetition_rates.append(rep_rate)
        
        print(f"  Prompt 5/5... Done.")

        # Aggregate
        avg_ppl = np.mean(perplexities)
        avg_ent = np.mean(entropies)
        avg_rep = np.mean(repetition_rates)
        
        print(f"  Perplexity: {avg_ppl:.2f}")
        print(f"  Entropy:    {avg_ent:.2f}")
        print(f"  Repetition: {avg_rep:.2f}")
        
        results.append(ExperimentResult(name, avg_ppl, avg_ent, avg_rep, 0.0))

    # 5. Visualize Results
    plot_results(results)

def plot_results(results: List[ExperimentResult]):
    names = [r.method_name for r in results]
    ppls = [r.perplexity for r in results]
    ents = [r.entropy for r in results]
    reps = [r.repetition_rate for r in results]
    
    x = np.arange(len(names))
    width = 0.25
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot Perplexity (Bar)
    ax1.bar(x - width, ppls, width, label='Perplexity (Lower is Better)', color='skyblue')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Thermodynamic vs Standard Sampling')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    
    # Plot Entropy (Line)
    ax2 = ax1.twinx()
    ax2.plot(x, ents, color='orange', marker='o', linewidth=2, label='Entropy (Higher is Better)')
    ax2.set_ylabel('Entropy')
    
    # Plot Repetition (Line)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(x, reps, color='green', marker='s', linewidth=2, linestyle='--', label='Repetition Rate')
    ax3.set_ylabel('Repetition Rate')

    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sampling_comparison.png'))
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    run_comparison()
