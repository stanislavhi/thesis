import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from physics.core.dynamics import CoupledDynamics
from physics.core.entropy import calculate_entropy_production
from physics.core.kl_divergence import calculate_kl_divergence
from physics.substrate.double_well import DoubleWellPotential
from physics.substrate.kramers import calculate_kramers_rate
from physics.substrate.heat_capacity import calculate_schottky_heat_capacity

def run_verification():
    print("--- THERMODYNAMIC BOUND VERIFICATION (PHASE SWEEP) ---")
    
    # 1. System Parameters
    T = 0.5
    eta = 0.5 # Increased learning rate to force dynamics
    k_B = 1.0
    
    # 2. Substrate Physics
    potential = DoubleWellPotential(a=1.0, b=2.0)
    delta_E = potential.barrier_height()
    k_escape = calculate_kramers_rate(delta_E, T)
    C_V = calculate_schottky_heat_capacity(delta_E, T)
    
    print(f"Parameters: T={T}, DeltaE={delta_E}, k_escape={k_escape:.4f}, C_V={C_V:.4f}")
    
    # 3. Sweep Alpha (Coupling Strength)
    alphas = np.linspace(0, 5.0, 20) # Wider range
    results = []
    
    q0, p0 = 0.2, 0.8 
    t = np.linspace(0, 100, 2000) # More steps for resolution
    dt = t[1] - t[0]
    
    print(f"\n{'Alpha':<10} | {'Sigma':<10} | {'Epsilon':<10} | {'LHS':<12} | {'RHS':<12} | {'Result'}")
    print("-" * 75)

    valid_count = 0
    
    for alpha in alphas:
        # Run Dynamics with Thermal Noise
        model = CoupledDynamics(eta, alpha, temperature=T)
        traj = model.simulate(q0, p0, t)
        
        # Calculate Metrics (Steady State)
        active_traj = traj[500:] 
        
        sigma = calculate_entropy_production(active_traj, dt)
        
        q_mean = np.mean(active_traj[:, 0])
        p_mean = np.mean(active_traj[:, 1])
        epsilon = calculate_kl_divergence(q_mean, p_mean)
        
        # The Bound
        lhs = (sigma**2) * epsilon
        
        ln2 = np.log(2)
        rhs = (k_B**2) * (ln2**3) * eta * k_escape * delta_E * abs(1 - 2*p_mean) / C_V
        
        is_valid = lhs >= rhs
        if is_valid: valid_count += 1
        
        res_str = "VALID" if is_valid else "VIOLATED"
        print(f"{alpha:<10.2f} | {sigma:<10.4f} | {epsilon:<10.4f} | {lhs:<12.2e} | {rhs:<12.2e} | {res_str}")
        
        results.append((alpha, lhs, rhs))

    # 4. Plotting
    alphas_plot = [r[0] for r in results]
    lhs_plot = [r[1] for r in results]
    rhs_plot = [r[2] for r in results]

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(alphas_plot, lhs_plot, 'b-o', label='LHS (Sigma^2 * Eps)')
    plt.plot(alphas_plot, rhs_plot, 'r--', label='RHS (Physical Bound)')
    plt.xlabel('Coupling Strength (Alpha)')
    plt.yscale('log')
    plt.title('Thermodynamic Bound Check')
    plt.legend()
    plt.grid(True)
    
    # Plot a sample trajectory for high alpha
    model_high = CoupledDynamics(eta, 5.0, temperature=T)
    traj_high = model_high.simulate(q0, p0, t)
    
    plt.subplot(1, 2, 2)
    plt.plot(t, traj_high[:, 0], label='Model (q)', alpha=0.7)
    plt.plot(t, traj_high[:, 1], label='Physical State (p)', alpha=0.7)
    plt.title('Stochastic Regress (Alpha=5.0)')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/physics_verification.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
    
    if valid_count > 0:
        print(f"\n>>> THEORY PARTIALLY VALIDATED. Bound holds for {valid_count}/{len(alphas)} regimes.")
    else:
        print("\n>>> BOUND VIOLATED EVERYWHERE. Re-check derivation.")

if __name__ == "__main__":
    run_verification()
