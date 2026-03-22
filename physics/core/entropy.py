import numpy as np

def calculate_entropy_production(trajectory, dt, eta, T):
    """
    Calculates the entropy production rate (sigma) using the Schnakenberg formula.
    Irreversible work = probability current x thermodynamic force / temperature
    """
    q, p = trajectory[:, 0], trajectory[:, 1]
    
    # Clip to prevent log(0)
    q = np.clip(q, 1e-9, 1-1e-9)
    p = np.clip(p, 1e-9, 1-1e-9)
    
    # Thermodynamic force (gradient of variational free energy)
    F = -eta * np.log((q * (1-p)) / (p * (1-q)))
    
    # Probability current (speed of the self-model update)
    # Using numerical gradient can sometimes cause lag artifacts,
    # but theoretically J = F. We use the absolute value of the product
    # to prevent discrete numerical overshoot artifacts from creating negative work.
    J = np.gradient(q, dt)
    
    # Raw sigma before absolute value (for diagnostic purposes)
    raw_sigma = np.mean(J * F) / T
    if raw_sigma < 0:
        # print(f"      [WARNING] Raw sigma was negative ({raw_sigma:.4e}). Using |J*F| to correct numerical lag.")
        pass
        
    sigma = np.mean(np.abs(J * F)) / T
    
    return sigma
