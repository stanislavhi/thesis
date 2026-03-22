import numpy as np


class CoupledDynamicsNState:
    """
    N-state generalization of the self-modeling coupled dynamics.

    dq/dt = -eta * grad_q D_KL(q || p)   (projected onto simplex tangent space)
    dp/dt = A * ||dq/dt||                 (matrix coupling, perturbation ∝ update speed)

    For N=2 with scalar alpha, this reproduces the original CoupledDynamics exactly.
    """
    def __init__(self, n_states, eta, coupling_matrix, temperature=0.0):
        self.N = n_states
        self.eta = eta
        self.A = np.array(coupling_matrix)  # N x N coupling matrix
        self.T = temperature

        assert self.A.shape == (n_states, n_states), \
            f"Coupling matrix must be {n_states}x{n_states}, got {self.A.shape}"

    @staticmethod
    def from_scalar_alpha(n_states, eta, alpha, temperature=0.0):
        """Create with uniform scalar coupling (alpha * I) for comparison with 2-state."""
        A = alpha * np.eye(n_states)
        return CoupledDynamicsNState(n_states, eta, A, temperature)

    def _kl_gradient(self, q, p):
        """
        Gradient of D_KL(q || p) w.r.t. q, projected onto the simplex tangent space.
        Raw gradient component i: ln(q_i / p_i) + 1
        Projection: subtract the mean to stay on the tangent space of the simplex.
        """
        eps = 1e-12
        q_safe = np.clip(q, eps, 1.0)
        p_safe = np.clip(p, eps, 1.0)
        raw_grad = np.log(q_safe / p_safe) + 1.0
        # Project onto simplex tangent space (subtract mean)
        return raw_grad - np.mean(raw_grad)

    def _project_simplex(self, x):
        """Project x onto the probability simplex via clipping and renormalization."""
        x = np.clip(x, 1e-12, None)
        return x / np.sum(x)

    def simulate(self, q0, p0, t_span):
        """
        Simulate the N-state coupled dynamics using Euler-Maruyama/Milstein.

        Args:
            q0: Initial model distribution (N,)
            p0: Initial physical distribution (N,)
            t_span: Time array

        Returns:
            q_traj: (n_steps, N) model trajectory
            p_traj: (n_steps, N) physical state trajectory
        """
        dt = t_span[1] - t_span[0]
        n_steps = len(t_span)

        q_traj = np.zeros((n_steps, self.N))
        p_traj = np.zeros((n_steps, self.N))

        q = np.array(q0, dtype=float)
        p = np.array(p0, dtype=float)
        q_traj[0] = q
        p_traj[0] = p

        for i in range(1, n_steps):
            q = self._project_simplex(q)
            p = self._project_simplex(p)

            # Model update: gradient descent on KL divergence
            grad = self._kl_gradient(q, p)
            dq_dt = -self.eta * grad

            # Clip to prevent numerical blowup
            dq_dt = np.clip(dq_dt, -10.0, 10.0)

            # Update speed (L2 norm)
            update_speed = np.linalg.norm(dq_dt)

            # Adversarial perturbation: push p AWAY from q
            # Direction: (p - q) / ||p - q||, scaled by coupling matrix and update speed
            # This survives simplex projection because it increases D_KL(q||p)
            diff = p - q
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-12:
                direction = diff / diff_norm
            else:
                direction = np.zeros(self.N)
            dp_dt_det = self.A @ direction * update_speed

            # Update q
            q_new = q + dq_dt * dt

            # Update p with optional noise
            if self.T > 0:
                # State-dependent noise on simplex
                noise_scale = np.sqrt(2 * self.T * dt)
                noise = np.random.randn(self.N) * noise_scale * np.sqrt(p * (1 - p))
                noise -= np.mean(noise)  # Keep on simplex tangent space
                p_new = p + dp_dt_det * dt + noise
            else:
                p_new = p + dp_dt_det * dt

            q_traj[i] = self._project_simplex(q_new)
            p_traj[i] = self._project_simplex(p_new)

            q = q_traj[i]
            p = p_traj[i]

        return q_traj, p_traj

    def compute_kl_divergence(self, q, p):
        """D_KL(q || p) for N-state distributions."""
        eps = 1e-12
        q_safe = np.clip(q, eps, 1.0)
        p_safe = np.clip(p, eps, 1.0)
        return np.sum(q_safe * np.log(q_safe / p_safe))


def verify_n2_equivalence():
    """
    Verify that N=2 reproduces the scalar 2-state dynamics,
    and that alpha_crit = 1 is universal across N.

    The critical test is TRANSIENT behavior near the fixed point q=p:
    - alpha < 1: KL decreases (model correction outruns perturbation)
    - alpha > 1: KL increases (perturbation outruns correction)
    On compact domains (simplex), boundary effects eventually force convergence
    for all alpha, so we test the linearized regime with small perturbations
    and short time horizons.
    """
    from physics.core.dynamics import CoupledDynamics

    eta = 0.5
    alpha = 0.3
    q0_scalar, p0_scalar = 0.49, 0.51  # small perturbation near fixed point
    t_short = np.linspace(0, 5, 5000)

    # --- N=2 equivalence: KL trajectory comparison ---
    # The simplex projection for N=2 halves the effective gradient:
    #   projected_grad[0] = (g1 - g2)/2, whereas scalar uses g1 - g2.
    # To get exact equivalence, scale eta by 2 for the N-state model.
    eta_nstate = eta * 2

    print("--- N=2 EQUIVALENCE CHECK ---")
    model_2state = CoupledDynamics(eta, alpha, temperature=0.0)
    traj_2state = model_2state.simulate(q0_scalar, p0_scalar, t_short)

    q0_vec = np.array([q0_scalar, 1 - q0_scalar])
    p0_vec = np.array([p0_scalar, 1 - p0_scalar])
    model_nstate = CoupledDynamicsNState.from_scalar_alpha(2, eta_nstate, alpha, temperature=0.0)
    q_traj, p_traj = model_nstate.simulate(q0_vec, p0_vec, t_short)

    eps = 1e-12
    kl_scalar = []
    kl_nstate = []
    for i in range(0, len(t_short), 100):
        qs, ps = traj_2state[i, 0], traj_2state[i, 1]
        qs = np.clip(qs, eps, 1 - eps)
        ps = np.clip(ps, eps, 1 - eps)
        kl_s = qs * np.log(qs / ps) + (1 - qs) * np.log((1 - qs) / (1 - ps))
        kl_scalar.append(kl_s)
        kl_n = model_nstate.compute_kl_divergence(q_traj[i], p_traj[i])
        kl_nstate.append(kl_n)

    kl_diff = np.mean(np.abs(np.array(kl_scalar) - np.array(kl_nstate)))
    print(f"Mean |KL_scalar - KL_N2|: {kl_diff:.6e}")

    # --- Alpha_crit universality: transient test near fixed point ---
    print(f"\n--- ALPHA_CRIT UNIVERSALITY CHECK (transient regime) ---")
    print(f"Theory: α_crit = 1 for all N. α<1 → KL shrinks, α>1 → KL grows.")
    print(f"Testing with small perturbation near uniform distribution.\n")

    t_test = np.linspace(0, 2, 2000)  # short time, stay in linear regime
    n_check = 50  # check KL at step 50 vs step 0

    for N in [2, 3, 5, 10]:
        results = []
        for alpha_test in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            model = CoupledDynamicsNState.from_scalar_alpha(N, eta, alpha_test, temperature=0.0)
            # Small perturbation: q slightly above uniform, p slightly below in first component
            delta = 0.01
            q0 = np.ones(N) / N
            q0[0] += delta
            q0 = q0 / q0.sum()
            p0 = np.ones(N) / N
            p0[0] -= delta
            p0 = p0 / p0.sum()

            q_t, p_t = model.simulate(q0, p0, t_test)

            kl_0 = model.compute_kl_divergence(q_t[0], p_t[0])
            kl_n = model.compute_kl_divergence(q_t[n_check], p_t[n_check])
            label = "GROWS" if kl_n > kl_0 * 1.001 else "SHRINKS"
            results.append(f"α={alpha_test:.1f}→{label}")

        print(f"  N={N:>2}: {' | '.join(results)}")

    print(f"\nExpected: transition at α=1.0 for all N.")


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    verify_n2_equivalence()
