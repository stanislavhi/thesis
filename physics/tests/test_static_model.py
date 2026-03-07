import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from physics.core.dynamics import CoupledDynamics
from physics.core.kl_divergence import calculate_kl_divergence

def test_static_model():
    """
    Verifies that when alpha = 0 (Static Target), the model converges perfectly.
    This confirms that the 'regress' is caused by the coupling, not the model itself.
    """
    print("--- TEST: STATIC MODEL (Alpha = 0) ---")
    
    T = 0.0 # No noise
    eta = 0.1
    alpha = 0.0 # No physical drift from update
    
    q0, p0 = 0.2, 0.8
    t = np.linspace(0, 50, 500)
    
    model = CoupledDynamics(eta, alpha, temperature=T)
    traj = model.simulate(q0, p0, t)
    
    q_final, p_final = traj[-1]
    epsilon = calculate_kl_divergence(q_final, p_final)
    
    print(f"Final q: {q_final:.4f}")
    print(f"Final p: {p_final:.4f}")
    print(f"Epsilon: {epsilon:.6e}")
    
    if epsilon < 1e-4:
        print(">>> PASS: Model converged to static target.")
    else:
        print(">>> FAIL: Model failed to converge.")

if __name__ == "__main__":
    test_static_model()
