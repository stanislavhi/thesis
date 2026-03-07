import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from physics.core.dynamics import CoupledDynamics
from physics.core.entropy import calculate_entropy_production
from physics.substrate.double_well import DoubleWellPotential
from physics.substrate.kramers import calculate_kramers_rate

def sweep_barrier():
    """
    Sweeps the barrier height (Delta E) to see how stability changes.
    Higher barrier -> Lower k_escape -> More stable substrate -> Lower Sigma.
    """
    print("--- EXPERIMENT: SWEEP BARRIER HEIGHT (Delta E) ---")
    
    barriers = np.linspace(0.5, 5.0, 20)
    sigmas = []
    
    T = 0.5
    eta = 0.1
    
    t = np.linspace(0, 50, 500)
    dt = t[1] - t[0]
    
    for b in barriers:
        # Construct potential with desired barrier
        # Barrier = b^2 / 4a. Let a=1, so Barrier = b^2/4 => b = sqrt(4*Barrier)
        param_b = np.sqrt(4 * b)
        potential = DoubleWellPotential(a=1.0, b=param_b)
        
        # Calculate k_escape to estimate alpha scaling
        # We assume alpha scales with k_escape for this experiment
        k_escape = calculate_kramers_rate(b, T)
        alpha = k_escape * 10.0 # Heuristic scaling
        
        model = CoupledDynamics(eta, alpha, temperature=T)
        traj = model.simulate(0.2, 0.8, t)
        
        sigma = calculate_entropy_production(traj, dt)
        sigmas.append(sigma)
        
        print(f"Barrier: {b:.2f} | k_escape: {k_escape:.4f} | Sigma: {sigma:.4f}")
        
    plt.figure()
    plt.plot(barriers, sigmas, 'g-o')
    plt.xlabel('Barrier Height (Delta E)')
    plt.ylabel('Entropy Production (Sigma)')
    plt.title('Stability vs Entropy')
    plt.grid(True)
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/experiment_sweep_barrier.png'))
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    sweep_barrier()
