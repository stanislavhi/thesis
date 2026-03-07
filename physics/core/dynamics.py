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

    def system(self, state, t):
        q, p = state
        
        # Safety clamps
        epsilon = 1e-9
        q = np.clip(q, epsilon, 1 - epsilon)
        p = np.clip(p, epsilon, 1 - epsilon)
        
        # 1. Model Update (dq/dt)
        dq_dt = -self.eta * np.log( (q * (1 - p)) / (p * (1 - q)) )
        
        # 2. Physical Drift (dp/dt)
        # Deterministic part
        drift = self.alpha * np.abs(dq_dt)
        
        # Stochastic part (Langevin noise)
        # We approximate noise here for the derivative. 
        # In a rigorous SDE solver, noise is added to the state update, not the derivative.
        # But for this RK4 implementation, we'll add a fluctuating term to the drift.
        # Magnitude ~ sqrt(2 * T)
        noise = 0.0
        if self.T > 0:
            noise = np.random.normal(0, np.sqrt(2 * self.T * 0.01)) # Scaled for dt approx
            
        dp_dt = drift + noise
        
        return np.array([dq_dt, dp_dt])

    def simulate(self, q0, p0, t_span):
        """
        Runs the simulation over the given time span using RK4.
        """
        dt = t_span[1] - t_span[0]
        n_steps = len(t_span)
        
        trajectory = np.zeros((n_steps, 2))
        trajectory[0] = [q0, p0]
        
        current_state = np.array([q0, p0])
        
        for i in range(1, n_steps):
            t = t_span[i-1]
            
            # RK4 Integration (Deterministic part)
            # Note: Adding noise inside RK4 steps is mathematically dubious for SDEs (Ito vs Stratonovich).
            # For a verification prototype, we will use Euler-Maruyama for the stochastic part
            # and RK4 for the deterministic part, or just switch to Euler for simplicity/correctness with noise.
            
            # Let's switch to Euler-Maruyama for correct noise handling
            # dq = f(q,p)*dt
            # dp = g(q,p)*dt + sigma*dW
            
            q, p = current_state
            epsilon = 1e-9
            q = np.clip(q, epsilon, 1 - epsilon)
            p = np.clip(p, epsilon, 1 - epsilon)
            
            dq_dt = -self.eta * np.log( (q * (1 - p)) / (p * (1 - q)) )
            dp_dt_det = self.alpha * np.abs(dq_dt)
            
            # Update
            q_new = q + dq_dt * dt
            
            noise = 0.0
            if self.T > 0:
                noise = np.random.normal(0, np.sqrt(2 * self.T * dt)) # Correct SDE scaling
            
            p_new = p + dp_dt_det * dt + noise
            
            # Clamp
            q_new = np.clip(q_new, epsilon, 1 - epsilon)
            p_new = np.clip(p_new, epsilon, 1 - epsilon)
            
            current_state = np.array([q_new, p_new])
            trajectory[i] = current_state

        return trajectory
