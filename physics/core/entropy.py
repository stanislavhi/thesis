import numpy as np

def calculate_entropy_production(trajectory, dt):
    """
    Calculates the entropy production rate (sigma) from the trajectory.
    sigma ~ sum( (dq/dt)^2 ) * dt
    
    In this simplified model, entropy production is proportional to the 
    speed of the model update (heat dissipation).
    """
    q = trajectory[:, 0]
    
    # Calculate dq/dt numerically
    dq_dt = np.gradient(q, dt)
    
    # Sigma is the integral of the squared rate (Joule heating analogy)
    # averaged over time
    sigma = np.mean(dq_dt**2)
    
    return sigma
