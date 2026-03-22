import streamlit as st
import numpy as np
import plotly.graph_objects as go


def render():
    st.header("🌀 Lorenz Explorer")
    st.caption("Explore the chaotic attractor that drives mutations in the THESIS architecture.")

    col_params, col_viz = st.columns([1, 3])

    with col_params:
        st.subheader("Parameters")
        sigma = st.slider("σ (Prandtl)", 1.0, 30.0, 10.0, 0.5,
                          help="Rate of convective overturning")
        rho = st.slider("ρ (Rayleigh)", 1.0, 50.0, 28.0, 0.5,
                        help="Temperature difference driving. ρ>24.74 → chaos")
        beta = st.slider("β (Aspect ratio)", 0.5, 10.0, 8/3, 0.1,
                         help="Physical proportions of the layer")

        st.divider()
        n_steps = st.slider("Trail length", 1000, 20000, 10000, 1000)
        dt = st.slider("dt", 0.001, 0.02, 0.005, 0.001)

        st.divider()
        color_by = st.radio("Color by", ["Z value (mutation strength)", "Speed", "Time"])

    # Integrate Lorenz system
    x, y, z = 1.0, 1.0, 1.0
    xs, ys, zs = [x], [y], [z]
    speeds = [0.0]

    for _ in range(n_steps):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x += dx
        y += dy
        z += dz
        xs.append(x)
        ys.append(y)
        zs.append(z)
        speeds.append(np.sqrt(dx**2 + dy**2 + dz**2))

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    speeds = np.array(speeds)

    # Color mapping
    if color_by == "Z value (mutation strength)":
        colors = zs
        colorbar_title = "Z (mutation magnitude)"
        colorscale = "Plasma"
    elif color_by == "Speed":
        colors = speeds
        colorbar_title = "Speed"
        colorscale = "Inferno"
    else:
        colors = np.arange(len(xs))
        colorbar_title = "Time step"
        colorscale = "Viridis"

    with col_viz:
        fig = go.Figure(data=[go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(
                color=colors,
                colorscale=colorscale,
                width=2,
                colorbar=dict(
                    title=colorbar_title,
                    thickness=15,
                    len=0.6,
                ),
            ),
            hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>",
        )])

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=650,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z (mutation magnitude)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Mutation distribution
    st.subheader("📊 Mutation Magnitude Distribution")
    col_hist, col_stats = st.columns([2, 1])

    with col_hist:
        # Z-value histogram (this is what get_perturbation returns)
        z_norm = zs / max(abs(zs.max()), abs(zs.min()))
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=z_norm,
            nbinsx=80,
            marker=dict(
                color="rgba(99, 102, 241, 0.7)",
                line=dict(color="rgba(129, 140, 248, 1)", width=1),
            ),
            name="Z / max(|Z|)",
        ))
        fig_hist.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            xaxis_title="Normalized Z (mutation perturbation)",
            yaxis_title="Count",
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_stats:
        st.metric("Mean Z", f"{zs.mean():.2f}")
        st.metric("Std Z", f"{zs.std():.2f}")
        st.metric("Min Z", f"{zs.min():.2f}")
        st.metric("Max Z", f"{zs.max():.2f}")

        # Chaos indicator
        if rho > 24.74:
            st.success("✅ **Chaotic regime**")
        else:
            st.warning("⚠️ **Non-chaotic** (ρ < 24.74)")

    with st.expander("💡 How this drives mutations"):
        st.markdown("""
        The Lorenz attractor's **Z coordinate** is used as the mutation signal:

        - **High |Z|** → Large topology changes (grow/shrink hidden layer by many neurons)
        - **|Z| > 1.5** → Activation function swap (ReLU ↔ Tanh ↔ GELU)
        - **Low |Z|** → Small adjustments (±1 neuron)

        The chaotic nature ensures mutations are **deterministic but unpredictable**,
        providing structured exploration of the architecture space without random noise.
        """)


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(layout="wide")
    render()
