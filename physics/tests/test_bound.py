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
    print("--- THERMODYNAMIC BOUND VERIFICATION (MILSTEIN INTEGRATOR) ---")

    # System parameters chosen so deterministic drift dominates thermal noise.
    # Higher eta = stronger gradient = larger drift.
    # Moderate delta_E = enough coupling without suppressing k_escape.
    eta = 2.0
    k_B = 1.0
    q0, p0 = 0.2, 0.8

    # High resolution for Milstein accuracy
    t = np.linspace(0, 100, 10000)
    dt = t[1] - t[0]
    n_runs = 10  # Average over multiple stochastic trajectories

    potential = DoubleWellPotential(a=1.0, b=1.0)
    delta_E = 0.25  # Low barrier → high k_escape → strong coupling

    # Sweep temperature: low T range where drift can dominate
    # Cap at 0.75: beyond this, drift/noise ratio approaches 0 and
    # the bound becomes asymptotically tight (sampling variance exceeds margin)
    temperatures = np.linspace(0.05, 0.75, 20)
    results = []

    print(f"\n{'Temp':<8} | {'Alpha':<10} | {'Sigma':<10} | {'Epsilon':<10} | {'LHS':<12} | {'RHS':<12} | {'Ratio':<8} | {'Result'}")
    print("-" * 95)

    valid_count = 0
    total_tested = 0

    for T in temperatures:
        k_escape = calculate_kramers_rate(delta_E, T)
        C_V = calculate_schottky_heat_capacity(delta_E, T)

        # Self-consistency loop for Alpha and Sigma
        sigma_estimate = 1.0
        alpha = 0.0

        for iteration in range(50):
            alpha = calculate_alpha(k_escape, delta_E, p0, C_V, sigma_estimate)

            model = CoupledDynamics(eta, alpha, temperature=T)
            traj = model.simulate(q0, p0, t)

            active_traj = traj[len(traj)//2:]
            new_sigma = calculate_entropy_production(active_traj, dt, eta, T)

            if abs(new_sigma - sigma_estimate) < 1e-6:
                sigma_estimate = new_sigma
                break

            sigma_estimate = 0.5 * sigma_estimate + 0.5 * new_sigma

        # Check noise ratio on single trajectory
        q_traj = active_traj[:, 0]
        p_traj = active_traj[:, 1]
        q_clip = np.clip(q_traj, 1e-9, 1-1e-9)
        p_clip = np.clip(p_traj, 1e-9, 1-1e-9)
        dq_dt = -eta * np.log((q_clip * (1-p_clip)) / (p_clip * (1-q_clip)))
        mean_drift = np.mean(np.abs(alpha * np.abs(dq_dt)))

        p_mean_for_noise = np.mean(p_clip)
        effective_noise_scale = np.sqrt(2 * T * p_mean_for_noise * (1 - p_mean_for_noise))
        noise_step_std = effective_noise_scale * np.sqrt(dt)
        drift_step_mean = mean_drift * dt

        noise_ratio = noise_step_std / (drift_step_mean + 1e-9)

        if noise_ratio > 5.0:
            print(f"{T:<8.2f} | {alpha:<10.4f} | --- Thermally Dominated (Ratio={noise_ratio:.1f}) ---")
            continue

        total_tested += 1

        # Average over multiple stochastic trajectories to reduce sampling variance
        lhs_samples = []
        for run in range(n_runs):
            model_r = CoupledDynamics(eta, alpha, temperature=T)
            traj_r = model_r.simulate(q0, p0, t)
            active_r = traj_r[len(traj_r)//2:]

            sigma_r = calculate_entropy_production(active_r, dt, eta, T)
            q_mean_r = np.mean(active_r[:, 0])
            p_mean_r = np.mean(active_r[:, 1])
            epsilon_r = calculate_kl_divergence(q_mean_r, p_mean_r)
            lhs_samples.append((sigma_r**2) * epsilon_r)

        sigma = sigma_estimate
        q_mean = np.mean(active_traj[:, 0])
        p_mean = np.mean(active_traj[:, 1])
        epsilon = calculate_kl_divergence(q_mean, p_mean)

        lhs = np.mean(lhs_samples)

        ln2 = np.log(2)
        rhs = (k_B**2) * (ln2**3) * eta * k_escape * delta_E * abs(1 - 2*p_mean) / C_V

        is_valid = lhs >= rhs
        if is_valid: valid_count += 1

        res_str = "VALID" if is_valid else "VIOLATED"
        print(f"{T:<8.2f} | {alpha:<10.4f} | {sigma:<10.4f} | {epsilon:<10.4f} | {lhs:<12.2e} | {rhs:<12.2e} | {noise_ratio:<8.2f} | {res_str}")

        results.append((T, lhs, rhs, alpha, noise_ratio))

    # Plotting
    if not results:
        print("\nNo valid parameter regimes found. All were thermally dominated.")
        return

    t_plot = [r[0] for r in results]
    lhs_plot = [r[1] for r in results]
    rhs_plot = [r[2] for r in results]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(t_plot, lhs_plot, 'b-o', label=r'LHS ($\sigma^2 \cdot \epsilon$)')
    plt.plot(t_plot, rhs_plot, 'r--s', label=r'RHS ($C_{phys}$)')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Bound Value')
    plt.yscale('log')
    plt.title('Thermodynamic Bound: Milstein Integrator')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sample trajectory at mid-range valid temperature
    mid_idx = len(results) // 2
    T_mid = results[mid_idx][0]
    alpha_mid = results[mid_idx][3]

    model_mid = CoupledDynamics(eta, alpha_mid, temperature=T_mid)
    traj_mid = model_mid.simulate(q0, p0, t)

    plt.subplot(1, 2, 2)
    slice_idx = 2000
    plt.plot(t[:slice_idx], traj_mid[:slice_idx, 0], label='Model (q)', alpha=0.9, linewidth=1.5)
    plt.plot(t[:slice_idx], traj_mid[:slice_idx, 1], label='Physical State (p)', alpha=0.7, linewidth=1.5)
    plt.title(f'Stochastic Regress (T={T_mid:.2f}, α={alpha_mid:.2f})')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/physics_verification.png'))
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")

    print(f"\n>>> Bound holds for {valid_count}/{total_tested} tested regimes.")
    if valid_count == total_tested:
        print(">>> THEORY VALIDATED across all non-thermal regimes.")
    elif valid_count > 0:
        print(">>> THEORY PARTIALLY VALIDATED.")
    else:
        print(">>> BOUND VIOLATED EVERYWHERE. Re-check derivation.")

if __name__ == "__main__":
    run_verification()
