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

def run_deterministic_verification():
    print("--- THERMODYNAMIC BOUND VERIFICATION (DETERMINISTIC LIMIT) ---")
    
    # 1. System Parameters (Fixed)
    T = 0.5  # Used only for C_phys calculation, no noise injected
    eta = 0.5
    alpha = 1.0 # Fixed manually, breaking the circularity for this test
    k_B = 1.0
    
    t = np.linspace(0, 100, 5000) 
    dt = t[1] - t[0]
    
    # Substrate Physics (for RHS calculation)
    potential = DoubleWellPotential(a=1.0, b=1.0)
    delta_E = 0.5
    k_escape = calculate_kramers_rate(delta_E, T, attempt_freq=10.0)
    C_V = calculate_schottky_heat_capacity(delta_E, T)
    
    # 2. Sweep Initial Conditions (q0, p0)
    q0_vals = np.linspace(0.1, 0.9, 10)
    p0_vals = np.linspace(0.1, 0.9, 10)
    
    results = []
    valid_count = 0
    total_count = 0
    
    print(f"\n{'q0':<6} | {'p0':<6} | {'Sigma':<10} | {'Epsilon':<10} | {'LHS':<12} | {'RHS':<12} | {'Result'}")
    print("-" * 75)

    for q0 in q0_vals:
        for p0 in p0_vals:
            # Skip symmetric start where nothing happens
            if abs(q0 - p0) < 0.05:
                continue
                
            total_count += 1
            
            # Run Deterministic Dynamics (T=0.0 means no noise in simulate())
            model = CoupledDynamics(eta, alpha, temperature=0.0)
            traj = model.simulate(q0, p0, t)
            
            # Calculate Metrics
            active_traj = traj[500:] 
            
            # Pass effective T for entropy calculation formula scaling
            sigma = calculate_entropy_production(active_traj, dt, eta, T)
            
            q_mean = np.mean(active_traj[:, 0])
            p_mean = np.mean(active_traj[:, 1])
            epsilon = calculate_kl_divergence(q_mean, p_mean)
            
            # The Bound
            lhs = (sigma**2) * epsilon
            
            ln2 = np.log(2)
            rhs = (k_B**2) * (ln2**3) * eta * k_escape * delta_E * abs(1 - 2*p_mean) / C_V
            
            is_valid = lhs >= rhs
            if is_valid: valid_count += 1
            
            # Only print a subset to avoid console spam
            if total_count % 10 == 0:
                res_str = "VALID" if is_valid else "VIOLATED"
                print(f"{q0:<6.2f} | {p0:<6.2f} | {sigma:<10.4f} | {epsilon:<10.4f} | {lhs:<12.2e} | {rhs:<12.2e} | {res_str}")
            
            results.append((q0, p0, lhs, rhs, is_valid))

    print(f"\n>>> DETERMINISTIC CORE TEST: Bound holds for {valid_count}/{total_count} initial conditions.")

    # 3. Plotting
    q0_plot = [r[0] for r in results]
    p0_plot = [r[1] for r in results]
    valid_plot = [r[4] for r in results]

    plt.figure(figsize=(12, 6))
    
    # Phase space plot of valid vs invalid regions
    plt.subplot(1, 2, 1)
    colors = ['green' if v else 'red' for v in valid_plot]
    plt.scatter(q0_plot, p0_plot, c=colors, s=100, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'k--', label='q=p (Singularity)')
    plt.xlabel('Initial Model (q0)')
    plt.ylabel('Initial Physical State (p0)')
    plt.title('Bound Validity in Phase Space\n(Green=Valid, Red=Violated)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Example Trajectory
    plt.subplot(1, 2, 2)
    model_ex = CoupledDynamics(eta, alpha, temperature=0.0)
    traj_ex = model_ex.simulate(0.2, 0.8, t)
    
    slice_idx = 2000
    plt.plot(t[:slice_idx], traj_ex[:slice_idx, 0], label='Model (q)', alpha=0.9, linewidth=1.5)
    plt.plot(t[:slice_idx], traj_ex[:slice_idx, 1], label='Physical State (p)', alpha=0.7, linewidth=1.5)
    plt.title('Deterministic Regress Trajectory\n(q0=0.2, p0=0.8)')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/deterministic_bound_check.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    run_deterministic_verification()
