import numpy as np

class CoupledDynamics:
    """
    Implements the two coupled ODEs governing the self-modeling system.
    
    dq/dt = -eta * log( q(1-p) / p(1-q) )     (Gradient Descent on Free Energy)
    dp/dt = alpha * |dq/dt| + noise           (Physical Drift + Thermal Fluctuations)
    """
    def __init__(self, eta, alpha, temperature=0.0):
        self.eta = eta      # Learning rate
        self.alpha = alpha  # Coupling constant
        self.T = temperature

    def simulate(self, q0, p0, t_span):
        """
        Runs the simulation over the given time span using Euler-Maruyama.
        """
        dt = t_span[1] - t_span[0]
        n_steps = len(t_span)
        
        trajectory = np.zeros((n_steps, 2))
        trajectory[0] = [q0, p0]
        
        current_state = np.array([q0, p0])
        
        for i in range(1, n_steps):
            q, p = current_state
            epsilon = 1e-9
            q = np.clip(q, epsilon, 1 - epsilon)
            p = np.clip(p, epsilon, 1 - epsilon)
            
            # Deterministic drift
            dq_dt = -self.eta * np.log( (q * (1 - p)) / (p * (1 - q)) )
            
            # FIX 2: Adaptive clipping to prevent numerical blowup at high eta
            # If the gradient is massive, the discrete time step will overshoot [0,1] bounds violently
            dq_dt = np.clip(dq_dt, -10.0, 10.0)
            
            dp_dt_det = self.alpha * np.abs(dq_dt)
            
            # Update q
            q_new = q + dq_dt * dt
            
            # Update p with thermal noise
            noise = 0.0
            if self.T > 0:
                # FIX: Ensure noise scale is physical relative to dt
                noise = np.random.normal(0, np.sqrt(2 * self.T * dt)) 
            
            p_new = p + dp_dt_det * dt + noise
            
            # Clamp to probability space
            q_new = np.clip(q_new, epsilon, 1 - epsilon)
            p_new = np.clip(p_new, epsilon, 1 - epsilon)
            
            current_state = np.array([q_new, p_new])
            trajectory[i] = current_state

        return trajectory
