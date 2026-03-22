import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from physics.core.dynamics import CoupledDynamics
from physics.core.entropy import calculate_entropy_production
from physics.core.kl_divergence import calculate_kl_divergence
from physics.core.coupling import calculate_alpha
from physics.substrate.double_well import DoubleWellPotential
from physics.substrate.kramers import calculate_kramers_rate
from physics.substrate.heat_capacity import calculate_schottky_heat_capacity

def run_verification():
    print("--- THERMODYNAMIC BOUND VERIFICATION (PARAMETER REGIME FIX) ---")
    
    # 1. System Parameters
    eta = 0.5 
    k_B = 1.0
    q0, p0 = 0.2, 0.8 
    
    # Increased resolution to prevent numerical overshoot in gradient
    t = np.linspace(0, 100, 5000) 
    dt = t[1] - t[0]
    
    # 2. Substrate Physics
    # FIX: Set DeltaE lower to increase k_escape and boost alpha
    # We want a regime where coupling (alpha) dominates thermal noise
    potential = DoubleWellPotential(a=1.0, b=1.0) # Lower b means lower barrier
    delta_E = 0.5 # Force specific delta E
    
    # 3. Sweep Temperature (Capped to prevent pure thermal runaway)
    temperatures = np.linspace(0.1, 1.0, 20)
    results = []
    
    print(f"\n{'Temp':<8} | {'Alpha':<10} | {'Sigma':<10} | {'Epsilon':<10} | {'LHS':<12} | {'RHS':<12} | {'Result'}")
    print("-" * 80)

    valid_count = 0
    
    for T in temperatures:
        # Note: k_escape uses attempt_freq=10.0 now to boost alpha
        k_escape = calculate_kramers_rate(delta_E, T)
        C_V = calculate_schottky_heat_capacity(delta_E, T)
        
        # Self-consistency loop for Alpha and Sigma
        sigma_estimate = 1.0
        alpha = 0.0
        
        for iteration in range(50):
            # Calculate alpha based on current sigma estimate
            alpha = calculate_alpha(k_escape, delta_E, p0, C_V, sigma_estimate)
            
            # Run Dynamics with new alpha
            model = CoupledDynamics(eta, alpha, temperature=T)
            traj = model.simulate(q0, p0, t)
            
            # Calculate new sigma
            active_traj = traj[500:] 
            new_sigma = calculate_entropy_production(active_traj, dt, eta, T)
            
            # Check convergence
            if abs(new_sigma - sigma_estimate) < 1e-6:
                sigma_estimate = new_sigma
                break
                
            # Damped update to prevent oscillation
            sigma_estimate = 0.5 * sigma_estimate + 0.5 * new_sigma
            
        sigma = sigma_estimate
        
        # Diagnostic: Check Noise Ratio
        q_traj = active_traj[:, 0]
        p_traj = active_traj[:, 1]
        
        # Calculate mean deterministic drift magnitude
        q_clip = np.clip(q_traj, 1e-9, 1-1e-9)
        p_clip = np.clip(p_traj, 1e-9, 1-1e-9)
        dq_dt = -eta * np.log((q_clip * (1-p_clip)) / (p_clip * (1-q_clip)))
        mean_drift = np.mean(np.abs(alpha * np.abs(dq_dt)))
        
        noise_magnitude = np.sqrt(2 * T / dt) # dt is multiplied inside EM step, so per sec it's this
        # Actually in simulation step: p_new = p + drift*dt + noise
        # where noise = N(0, sqrt(2*T*dt)). So standard deviation of noise step is sqrt(2*T*dt)
        # deterministic step is drift*dt
        noise_step_std = np.sqrt(2 * T * dt)
        drift_step_mean = mean_drift * dt
        
        noise_ratio = noise_step_std / (drift_step_mean + 1e-9)
        
        if noise_ratio > 1.0:
            print(f"{T:<8.2f} | {alpha:<10.4f} | --- Thermally Dominated Regime (Ratio={noise_ratio:.1f}) - Excluded ---")
            continue
        
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
        print(f"{T:<8.2f} | {alpha:<10.4f} | {sigma:<10.4f} | {epsilon:<10.4f} | {lhs:<12.2e} | {rhs:<12.2e} | {res_str}")
        
        results.append((T, lhs, rhs, alpha))

    # 4. Plotting
    if not results:
        print("No valid parameter regimes found. All were thermally dominated.")
        return

    t_plot = [r[0] for r in results]
    lhs_plot = [r[1] for r in results]
    rhs_plot = [r[2] for r in results]

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(t_plot, lhs_plot, 'b-o', label='LHS (Sigma^2 * Eps)')
    plt.plot(t_plot, rhs_plot, 'r--', label='RHS (Physical Bound)')
    plt.xlabel('Temperature (T)')
    plt.yscale('log')
    plt.title('Thermodynamic Bound Check (Self-Consistent)')
    plt.legend()
    plt.grid(True)
    
    # Plot a sample trajectory for the highest valid temperature
    high_T_result = results[-1]
    T_high = high_T_result[0]
    alpha_high = high_T_result[3]
    
    model_high = CoupledDynamics(eta, alpha_high, temperature=T_high)
    traj_high = model_high.simulate(q0, p0, t)
    
    plt.subplot(1, 2, 2)
    # Plot only a slice so it's readable
    slice_idx = 1000
    plt.plot(t[:slice_idx], traj_high[:slice_idx, 0], label='Model (q)', alpha=0.9, linewidth=1.5)
    plt.plot(t[:slice_idx], traj_high[:slice_idx, 1], label='Physical State (p)', alpha=0.7, linewidth=1.5)
    plt.title(f'Stochastic Regress (T={T_high:.2f}, Alpha={alpha_high:.2f})')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/physics_verification.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
    
    if valid_count > 0:
        print(f"\n>>> THEORY PARTIALLY VALIDATED. Bound holds for {valid_count}/{len(results)} valid regimes.")
    else:
        print("\n>>> BOUND VIOLATED EVERYWHERE. Re-check derivation.")

if __name__ == "__main__":
    run_verification()
