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

def is_boundary_collapse(trajectory, tolerance=1e-3, consecutive_steps=100):
    """
    Checks if the trajectory has collapsed into a boundary state where both q and p
    are stuck near 0 or near 1.
    """
    q, p = trajectory[:, 0], trajectory[:, 1]
    
    # Check lower boundary (near 0)
    near_zero = (q < tolerance) & (p < tolerance)
    zero_runs = np.convolve(near_zero.astype(int), np.ones(consecutive_steps), mode='valid')
    if np.any(zero_runs == consecutive_steps):
        return True
        
    # Check upper boundary (near 1)
    near_one = ((1.0 - q) < tolerance) & ((1.0 - p) < tolerance)
    one_runs = np.convolve(near_one.astype(int), np.ones(consecutive_steps), mode='valid')
    if np.any(one_runs == consecutive_steps):
        return True
        
    return False

def run_deterministic_verification():
    print("--- THERMODYNAMIC BOUND VERIFICATION (DETERMINISTIC LIMIT) ---")
    
    # 1. System Parameters (Fixed)
    T = 0.5  # Used only for C_phys calculation, no noise injected
    eta = 0.5
    
    # FIX 3: Reduce coupling strength to observe the regress instead of instant collapse
    # alpha=1.0 is too strong, drives everything immediately into a boundary.
    alpha = 0.1 
    k_B = 1.0
    
    # FIX 1: Reduce dt dramatically (50,000 steps) for stability
    # Previous dt=0.02 was too large for high eta=0.5
    t = np.linspace(0, 100, 50000) 
    dt = t[1] - t[0]
    
    # Substrate Physics (for RHS calculation)
    potential = DoubleWellPotential(a=1.0, b=1.0)
    delta_E = 0.5
    k_escape = calculate_kramers_rate(delta_E, T, attempt_freq=10.0)
    C_V = calculate_schottky_heat_capacity(delta_E, T)
    
    # 2. Sweep Initial Conditions (q0, p0)
    # Using slightly higher resolution to see the gradient clearly
    q0_vals = np.linspace(0.1, 0.9, 15)
    p0_vals = np.linspace(0.1, 0.9, 15)
    
    results = []
    valid_count = 0
    total_count = 0
    collapse_count = 0
    
    print(f"\n{'q0':<6} | {'p0':<6} | {'Sigma':<10} | {'Epsilon':<10} | {'LHS':<12} | {'RHS':<12} | {'Result'}")
    print("-" * 75)

    for q0 in q0_vals:
        for p0 in p0_vals:
            if abs(q0 - p0) < 0.05:
                continue
                
            # Run Deterministic Dynamics
            model = CoupledDynamics(eta, alpha, temperature=0.0)
            traj = model.simulate(q0, p0, t)
            
            # FIX 2: Check for boundary collapse
            # We check the later part of the trajectory
            if is_boundary_collapse(traj[-5000:]):
                collapse_count += 1
                # Skip bound check for collapsed trajectories
                results.append((q0, p0, np.nan, np.nan, -1)) # -1 indicates collapse
                continue
                
            total_count += 1
            
            # Calculate Metrics (discard transient initial phase)
            active_traj = traj[-10000:] # Last 20%
            
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
            
            results.append((q0, p0, lhs, rhs, 1 if is_valid else 0))
            
            # Print a subset
            if total_count % 15 == 0:
                res_str = "VALID" if is_valid else "VIOLATED"
                print(f"{q0:<6.2f} | {p0:<6.2f} | {sigma:<10.4f} | {epsilon:<10.4f} | {lhs:<12.2e} | {rhs:<12.2e} | {res_str}")

    print(f"\n>>> DETERMINISTIC CORE TEST: Bound holds for {valid_count}/{total_count} valid trajectories.")
    print(f">>> (Excluded {collapse_count} trajectories due to boundary collapse)")

    # 3. Plotting
    q0_plot = [r[0] for r in results]
    p0_plot = [r[1] for r in results]
    valid_plot = [r[4] for r in results]

    plt.figure(figsize=(12, 6))
    
    # Phase space plot
    plt.subplot(1, 2, 1)
    
    # Color mapping: Green=Valid, Red=Violated, Gray=Collapsed
    colors = []
    for v in valid_plot:
        if v == 1: colors.append('green')
        elif v == 0: colors.append('red')
        else: colors.append('lightgray')
        
    plt.scatter(q0_plot, p0_plot, c=colors, s=100, alpha=0.8, edgecolors='k', linewidths=0.5)
    plt.plot([0, 1], [0, 1], 'k--', label='q=p')
    plt.xlabel('Initial Model (q0)')
    plt.ylabel('Initial Physical State (p0)')
    plt.title('Bound Validity in Phase Space\n(Green=Valid, Red=Violated, Gray=Collapsed)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Example Trajectory (find a valid one that isn't collapsed)
    plt.subplot(1, 2, 2)
    valid_starts = [(r[0], r[1]) for r in results if r[4] == 1]
    
    if valid_starts:
        # Pick one far from diagonal if possible
        q0_ex, p0_ex = max(valid_starts, key=lambda x: abs(x[0]-x[1]))
    else:
        # Fallback if all failed/collapsed
        q0_ex, p0_ex = 0.2, 0.8
        
    model_ex = CoupledDynamics(eta, alpha, temperature=0.0)
    traj_ex = model_ex.simulate(q0_ex, p0_ex, t)
    
    # Subsample for plotting to avoid huge files
    plot_skip = 100
    plt.plot(t[::plot_skip], traj_ex[::plot_skip, 0], label='Model (q)', alpha=0.9, linewidth=2)
    plt.plot(t[::plot_skip], traj_ex[::plot_skip, 1], label='Physical State (p)', alpha=0.7, linewidth=2)
    plt.title(f'Deterministic Regress Trajectory\n(q0={q0_ex:.2f}, p0={p0_ex:.2f}, alpha={alpha})')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/deterministic_bound_check.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    run_deterministic_verification()
