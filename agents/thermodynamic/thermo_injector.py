import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.base import Mutator, EvolutionaryAgent
from core.chaos import LorenzGenerator


# C_V threshold: networks with fewer params than this are "low C_V"
# Calibrated from CartPole (16 neurons, ~114 params) vs LunarLander (64 neurons, ~4416 params)
LOW_CV_THRESHOLD = 500


class ThermodynamicInjector(Mutator):
    """
    Implements the Thermodynamic Operator Selection Rule.

    Selects mutation operator based on the agent's Heat Capacity (C_V),
    estimated from network parameter count:

    - Low C_V  → Additive Noise (global phase transition)
    - High C_V → Targeted Dropout (localized annealing)

    The magnitude is further modulated by the agent's thermodynamic status
    (frozen/healthy/overheated) diagnosed via internal sigma.
    """
    def __init__(self, chaos_generator: LorenzGenerator, base_mutation_rate=0.1):
        self.chaos_gen = chaos_generator
        self.base_rate = base_mutation_rate

    def _estimate_cv(self, agent: nn.Module) -> int:
        """Estimate Heat Capacity from total parameter count."""
        return sum(p.numel() for p in agent.parameters())

    def _get_magnitude(self, status: str, cv: int) -> float:
        """
        Scale chaos magnitude by thermodynamic status AND inversely by C_V.
        Low C_V systems have no thermal inertia — even small heat is catastrophic.
        """
        if status == 'frozen':
            status_scale = 5.0
        elif status == 'overheated':
            status_scale = 0.5
        else:
            status_scale = 1.0

        # Inverse C_V scaling: smaller networks get gentler perturbations
        cv_scale = 1.0 / (1.0 + cv / 100.0)

        return self.base_rate * status_scale * cv_scale

    def _additive_noise(self, agent: nn.Module, magnitude: float, chaos_val: float) -> nn.Module:
        """
        Global entropy injection — melts the entire network.
        Correct operator for low C_V systems (small networks, simple heuristics).
        """
        with torch.no_grad():
            for param in agent.parameters():
                noise = torch.randn_like(param) * magnitude * abs(chaos_val)
                param.add_(noise)
        return agent

    def _targeted_dropout(self, agent: nn.Module, magnitude: float, chaos_val: float) -> nn.Module:
        """
        Localized annealing — zeroes the lowest-variance neurons and
        scales surviving weights to force re-routing.
        Correct operator for high C_V systems (large networks, complex representations).
        """
        with torch.no_grad():
            param_dict = dict(agent.named_parameters())
            for name, param in param_dict.items():
                if 'weight' not in name or param.dim() < 2:
                    continue

                # Compute per-neuron variance (output dimension)
                neuron_var = param.var(dim=1)
                # Drop fraction scaled by chaos magnitude
                drop_frac = min(0.3, magnitude * abs(chaos_val))
                n_drop = max(1, int(drop_frac * param.shape[0]))

                # Zero the lowest-variance neurons (frozen ones)
                _, lowest_idx = torch.topk(neuron_var, n_drop, largest=False)
                param[lowest_idx] = 0.0

                # Also zero corresponding bias if it exists
                bias_name = name.replace('weight', 'bias')
                bias = param_dict.get(bias_name)
                if bias is not None:
                    bias[lowest_idx] = 0.0

        return agent

    def select_operator(self, agent: nn.Module) -> str:
        """Select operator based on C_V estimation."""
        cv = self._estimate_cv(agent)
        op = 'additive_noise' if cv < LOW_CV_THRESHOLD else 'targeted_dropout'
        return op

    def mutate(self, agent: nn.Module, status: str = None,
               operator_override: str = None) -> nn.Module:
        """
        Mutates the agent using the thermodynamically correct operator.

        Args:
            operator_override: Force a specific operator ('additive_noise' or
                'targeted_dropout'). Use this for pure ES contexts where there
                is no gradient recovery to revive zeroed neurons.
        """
        # 1. Diagnose
        if status is None:
            if hasattr(agent, 'get_thermodynamic_status'):
                status = agent.get_thermodynamic_status()
            else:
                status = 'healthy'

        # 2. Select operator
        cv = self._estimate_cv(agent)
        if operator_override:
            operator = operator_override
        else:
            operator = 'additive_noise' if cv < LOW_CV_THRESHOLD else 'targeted_dropout'
        magnitude = self._get_magnitude(status, cv)
        chaos_val = self.chaos_gen.get_perturbation()

        print(f"   [Operator Selection] C_V={cv} → {operator} | status={status} | "
              f"magnitude={magnitude:.3f} | chaos={chaos_val:.2f}", flush=True)

        # 3. Apply the selected operator
        if operator == 'additive_noise':
            return self._additive_noise(agent, magnitude, chaos_val)
        else:
            return self._targeted_dropout(agent, magnitude, chaos_val)
