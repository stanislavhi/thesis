import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from physics.core.dynamics import CoupledDynamics
from physics.core.entropy import calculate_entropy_production
from physics.core.kl_divergence import calculate_kl_divergence

def test_localization():
    """
    Verifies that when k_escape -> 0 (Localization), the system freezes.
    In this limit, sigma -> 0 and epsilon -> constant.
    The bound is satisfied trivially (0 >= 0) or by the system stopping.
    """
    print("--- TEST: LOCALIZATION LIMIT (k_escape -> 0) ---")
    
    # Use an extremely low temperature to simulate the quantum Zeno / localization limit
    T = 1e-6 
    eta = 0.1
    alpha = 0.0 # Effectively decoupled because no heat can cause transitions
    
    q0, p0 = 0.2, 0.8
    t = np.linspace(0, 100, 1000) # Longer time to allow settling
    dt = t[1] - t[0]
    
    model = CoupledDynamics(eta, alpha, temperature=T)
    traj = model.simulate(q0, p0, t)
    
    # Calculate Sigma only for the steady state (last 20%)
    # The initial learning transient generates heat, but we want to know if it *stops*.
    steady_state_traj = traj[int(len(traj)*0.8):]
    sigma = calculate_entropy_production(steady_state_traj, dt)
    
    q_final, p_final = traj[-1]
    epsilon = calculate_kl_divergence(q_final, p_final)
    
    print(f"Temp: {T}")
    print(f"Steady State Sigma: {sigma:.6f}")
    print(f"Final Epsilon: {epsilon:.6f}")
    
    if sigma < 1e-4:
        print(">>> PASS: System is frozen (Localization confirmed).")
    else:
        print(">>> FAIL: System is still moving.")

if __name__ == "__main__":
    test_localization()
