import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def render():
    st.header("🔬 Physics Sandbox")
    st.caption("Explore the thermodynamic bound **σ² · ε ≥ C_phys** interactively.")

    col_params, col_results = st.columns([1, 2])

    with col_params:
        st.subheader("Parameters")

        temperature = st.slider("🌡️ Temperature (T)", 0.01, 2.0, 0.5, 0.01,
                                help="Heat bath temperature")
        eta = st.slider("⚡ Learning Rate (η)", 0.01, 1.0, 0.1, 0.01,
                        help="Model update speed")
        barrier = st.slider("🏔️ Barrier Height (ΔE)", 0.5, 5.0, 1.0, 0.1,
                            help="Energy barrier between states")
        alpha = st.slider("🔗 Coupling (α)", 0.0, 5.0, 1.0, 0.1,
                          help="Strength of self-reference coupling")

        st.divider()
        sim_steps = st.slider("📏 Simulation Steps", 100, 2000, 500, 100)
        dt = 0.01

    # Run simulation
    k_B = 1.0
    D = k_B * temperature
    ln2 = np.log(2)

    # Kramers escape rate
    k_escape = np.exp(-barrier / (k_B * temperature)) if temperature > 1e-9 else 0.0

    # Heat capacity (Schottky)
    if temperature > 1e-9:
        x_cap = barrier / (k_B * temperature)
        C_V = k_B * (x_cap ** 2) * np.exp(x_cap) / (1 + np.exp(x_cap)) ** 2
    else:
        C_V = 1e-9

    # Euler-Maruyama simulation
    q = 0.5  # model state
    p = np.random.uniform(0.3, 0.7)  # physical state

    q_hist, p_hist, sigma_hist, epsilon_hist = [], [], [], []
    lhs_hist, rhs_hist, valid_hist = [], [], []

    for step in range(sim_steps):
        eps = 1e-9
        q = np.clip(q, eps, 1 - eps)
        p = np.clip(p, eps, 1 - eps)

        # dq/dt — model update
        dq = -eta * np.log((q * (1 - p)) / (p * (1 - q))) * dt

        # dp/dt — physical drift + noise
        dp_det = alpha * abs(dq / dt)
        noise = np.random.normal(0, np.sqrt(2 * D * dt)) if D > 0 else 0.0
        dp = dp_det * dt + noise

        q += dq
        p += dp

        q = np.clip(q, eps, 1 - eps)
        p = np.clip(p, eps, 1 - eps)

        # Entropy production (speed of model update)
        sigma = abs(dq / dt)
        # KL divergence
        epsilon = q * np.log(q / (p + eps) + eps) + (1 - q) * np.log((1 - q) / (1 - p + eps) + eps)
        epsilon = max(0, epsilon)

        # The bound
        lhs = sigma ** 2 * epsilon
        rhs = k_B**2 * ln2**3 * eta * k_escape * barrier * abs(1 - 2 * np.mean([q, p])) / C_V

        q_hist.append(q)
        p_hist.append(p)
        sigma_hist.append(sigma)
        epsilon_hist.append(epsilon)
        lhs_hist.append(lhs)
        rhs_hist.append(rhs)
        valid_hist.append(lhs >= rhs)

    # Convert
    t = np.arange(sim_steps) * dt
    sigma_arr = np.array(sigma_hist)
    epsilon_arr = np.array(epsilon_hist)
    lhs_arr = np.array(lhs_hist)
    rhs_arr = np.array(rhs_hist)
    valid_arr = np.array(valid_hist)

    with col_results:
        # Bound validation summary
        valid_pct = np.mean(valid_arr) * 100
        steady_sigma = np.mean(sigma_arr[-50:])
        steady_epsilon = np.mean(epsilon_arr[-50:])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("✅ Bound Valid", f"{valid_pct:.0f}%")
        m2.metric("σ (steady)", f"{steady_sigma:.4f}")
        m3.metric("ε (steady)", f"{steady_epsilon:.4f}")
        m4.metric("k_escape", f"{k_escape:.4f}")

        # Main bound chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.45, 0.3, 0.25],
            vertical_spacing=0.06,
            subplot_titles=("σ²·ε vs C_phys (Bound Verification)", "States q(t), p(t)", "Entropy σ(t)")
        )

        # LHS vs RHS
        fig.add_trace(go.Scatter(
            x=t, y=lhs_arr, mode="lines", name="LHS: σ²·ε",
            line=dict(color="#818cf8", width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=rhs_arr, mode="lines", name="RHS: C_phys",
            line=dict(color="#ef4444", width=2, dash="dash"),
        ), row=1, col=1)

        # States
        fig.add_trace(go.Scatter(
            x=t, y=q_hist, mode="lines", name="q (model)",
            line=dict(color="#34d399", width=2),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=p_hist, mode="lines", name="p (physical)",
            line=dict(color="#f97316", width=2),
        ), row=2, col=1)

        # Entropy production
        fig.add_trace(go.Scatter(
            x=t, y=sigma_arr, mode="lines", name="σ",
            line=dict(color="#a78bfa", width=1.5),
            fill="tozeroy", fillcolor="rgba(167, 139, 250, 0.1)",
        ), row=3, col=1)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=700,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
            margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Derived parameters
    with st.expander("📐 Derived Parameters"):
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | k_B | {k_B} |
        | D (diffusion) | {D:.4f} |
        | k_escape (Kramers) | {k_escape:.6f} |
        | C_V (Schottky) | {C_V:.6f} |
        | ln(2)³ | {ln2**3:.6f} |
        | Bound valid % | {valid_pct:.1f}% |
        """)


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(layout="wide")
    render()
