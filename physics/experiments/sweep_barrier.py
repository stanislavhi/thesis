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
    fixed_alpha = 1.0 
    
    # Extended simulation time to 250
    t = np.linspace(0, 250, 2500)
    dt = t[1] - t[0]
    
    for b in barriers:
        # Construct potential with desired barrier
        param_b = np.sqrt(4 * b)
        potential = DoubleWellPotential(a=1.0, b=param_b)
        
        k_escape = calculate_kramers_rate(b, T)
        
        model = CoupledDynamics(eta, fixed_alpha, temperature=T)
        traj = model.simulate(0.2, 0.8, t)
        
        # Take steady-state average over last 50% of trajectory
        half_idx = len(traj) // 2
        active_traj = traj[half_idx:]
        
        sigma = calculate_entropy_production(active_traj, dt, eta, T)
        sigmas.append(sigma)
        
        print(f"Barrier: {b:.2f} | k_escape: {k_escape:.4f} | Sigma: {sigma:.4f}")
        
    plt.figure()
    plt.plot(barriers, sigmas, 'g-o')
    plt.xlabel('Barrier Height (Delta E)')
    plt.ylabel('Entropy Production (Sigma)')
    plt.title(f'Stability vs Entropy (Fixed Alpha={fixed_alpha})')
    plt.grid(True)
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/experiment_sweep_barrier.png'))
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    sweep_barrier()
