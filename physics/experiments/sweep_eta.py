import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from physics.core.dynamics import CoupledDynamics
from physics.core.entropy import calculate_entropy_production
from physics.core.kl_divergence import calculate_kl_divergence

def sweep_eta():
    """
    Sweeps the learning rate (eta) to see how it affects entropy production.
    Prediction: Higher eta -> Faster updates -> Higher Sigma.
    """
    print("--- EXPERIMENT: SWEEP LEARNING RATE (Eta) ---")
    
    etas = np.linspace(0.01, 1.0, 20)
    sigmas = []
    epsilons = []
    
    alpha = 1.0
    T = 0.1
    t = np.linspace(0, 50, 500)
    dt = t[1] - t[0]
    
    for eta in etas:
        model = CoupledDynamics(eta, alpha, temperature=T)
        traj = model.simulate(0.2, 0.8, t)
        
        sigma = calculate_entropy_production(traj, dt)
        q, p = traj[-1]
        eps = calculate_kl_divergence(q, p)
        
        sigmas.append(sigma)
        epsilons.append(eps)
        print(f"Eta: {eta:.2f} | Sigma: {sigma:.4f} | Epsilon: {eps:.4f}")
        
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(etas, sigmas, 'r-o')
    plt.xlabel('Learning Rate (eta)')
    plt.ylabel('Entropy Production (Sigma)')
    plt.title('Cost of Speed')
    
    plt.subplot(1, 2, 2)
    plt.plot(etas, epsilons, 'b-o')
    plt.xlabel('Learning Rate (eta)')
    plt.ylabel('Error (Epsilon)')
    plt.title('Accuracy vs Speed')
    
    plt.tight_layout()
    plt.savefig('experiment_sweep_eta.png')
    print("Plot saved to experiment_sweep_eta.png")

if __name__ == "__main__":
    sweep_eta()
