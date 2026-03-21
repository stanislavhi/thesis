import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.base import Mutator, EvolutionaryAgent
from core.chaos import LorenzGenerator

class ThermodynamicInjector(Mutator):
    """
    A specialized mutator that injects chaos based on the agent's internal thermodynamic state.
    
    Logic:
    - If 'frozen' (low sigma): Inject HIGH chaos to melt the weights and restart learning.
    - If 'overheated' (high sigma): Inject LOW chaos (cooling) to stabilize.
    - If 'healthy': Do nothing or apply minimal drift.
    """
    def __init__(self, chaos_generator: LorenzGenerator, base_mutation_rate=0.1):
        self.chaos_gen = chaos_generator
        self.base_rate = base_mutation_rate
        
    def mutate(self, agent: nn.Module, status: str = None) -> nn.Module:
        """
        Mutates the agent.
        Args:
            agent: The neural network to mutate.
            status: Optional explicit status ('frozen', 'healthy'). 
                    If None, calls agent.get_thermodynamic_status().
        """
        # 1. Diagnose the agent if status not provided
        if status is None:
            if hasattr(agent, 'get_thermodynamic_status'):
                status = agent.get_thermodynamic_status()
            else:
                status = 'healthy' # Default if method missing
        
        # 2. Determine Chaos Magnitude based on diagnosis
        if status == 'frozen':
            # High chaos needed to escape local minimum
            magnitude = self.base_rate * 5.0 
            # print(f"   [Injector] Agent is FROZEN. Injecting HIGH chaos ({magnitude:.2f}).")
        elif status == 'overheated':
            # Cooling needed - small negative perturbation or decay
            magnitude = self.base_rate * 0.5
            # print(f"   [Injector] Agent is OVERHEATED. Applying cooling ({magnitude:.2f}).")
        else:
            # Healthy - standard drift
            magnitude = self.base_rate
            
        # 3. Get Chaos Signal from Lorenz Attractor
        chaos_val = self.chaos_gen.get_perturbation()
        
        # 4. Apply Mutation to Weights
        with torch.no_grad():
            for param in agent.parameters():
                # Add noise scaled by chaos value and magnitude
                noise = torch.randn_like(param) * magnitude * abs(chaos_val)
                param.add_(noise)
                
        return agent
