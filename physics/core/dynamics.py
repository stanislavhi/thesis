import numpy as np

class CoupledDynamics:
    """
    Implements the two coupled ODEs governing the self-modeling system.

    dq/dt = -eta * log( q(1-p) / p(1-q) )     (Gradient Descent on Free Energy)
    dp/dt = alpha * |dq/dt| + noise           (Physical Drift + Thermal Fluctuations)

    Uses the Milstein method for the stochastic integration of p,
    which adds a correction term over Euler-Maruyama to reduce
    discretization error for multiplicative/additive noise.
    """
    def __init__(self, eta, alpha, temperature=0.0):
        self.eta = eta      # Learning rate
        self.alpha = alpha  # Coupling constant
        self.T = temperature

    def simulate(self, q0, p0, t_span):
        """
        Runs the simulation over the given time span using the Milstein method.

        For additive noise (diffusion coefficient independent of state),
        the Milstein correction is zero and it reduces to Euler-Maruyama.
        However, we apply it to the reflected (clamped) process where the
        effective diffusion depends on proximity to boundaries, making the
        correction non-trivial.
        """
        dt = t_span[1] - t_span[0]
        n_steps = len(t_span)
        sqrt_dt = np.sqrt(dt)

        trajectory = np.zeros((n_steps, 2))
        trajectory[0] = [q0, p0]

        current_state = np.array([q0, p0])
        epsilon = 1e-9

        for i in range(1, n_steps):
            q, p = current_state
            q = np.clip(q, epsilon, 1 - epsilon)
            p = np.clip(p, epsilon, 1 - epsilon)

            # Deterministic drift for q
            log_arg = (q * (1 - p)) / (p * (1 - q))
            dq_dt = -self.eta * np.log(log_arg)
            dq_dt = np.clip(dq_dt, -10.0, 10.0)

            # Deterministic drift for p
            dp_dt_det = self.alpha * np.abs(dq_dt)

            # Update q (deterministic — no stochastic term)
            q_new = q + dq_dt * dt

            # Update p with Milstein method
            if self.T > 0:
                # Diffusion coefficient: sigma(p) = sqrt(2T) * sqrt(p(1-p))
                # This ensures noise vanishes at boundaries [0,1]
                # and is maximal at p=0.5
                diff_p = np.sqrt(2 * self.T) * np.sqrt(p * (1 - p))

                # Milstein correction: d(sigma)/dp = sqrt(2T) * (1-2p) / (2*sqrt(p(1-p)))
                dsigma_dp = np.sqrt(2 * self.T) * (1 - 2*p) / (2 * np.sqrt(p * (1 - p)) + epsilon)

                # Wiener increment
                dW = np.random.normal(0, sqrt_dt)

                # Milstein step: drift*dt + sigma*dW + 0.5*sigma*sigma'*(dW^2 - dt)
                p_new = (p
                         + dp_dt_det * dt
                         + diff_p * dW
                         + 0.5 * diff_p * dsigma_dp * (dW**2 - dt))
            else:
                p_new = p + dp_dt_det * dt

            # Clamp to probability space
            q_new = np.clip(q_new, epsilon, 1 - epsilon)
            p_new = np.clip(p_new, epsilon, 1 - epsilon)

            current_state = np.array([q_new, p_new])
            trajectory[i] = current_state

        return trajectory
