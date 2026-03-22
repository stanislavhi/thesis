import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import time


def render():
    st.header("📊 Experiment Replayer")
    st.caption("Load a training log and replay the experiment with animated charts.")

    # Find all CSV logs
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs'))
    csv_files = sorted(glob.glob(os.path.join(logs_dir, '*.csv')))

    if not csv_files:
        st.warning("No CSV logs found in `logs/` directory. Run an experiment first!")
        return

    # File selector
    file_names = [os.path.basename(f) for f in csv_files]
    selected = st.selectbox("Select log file", file_names, index=0)
    selected_path = csv_files[file_names.index(selected)]

    try:
        df = pd.read_csv(selected_path)
    except Exception as e:
        st.error(f"Failed to load: {e}")
        return

    # Detect log type and show appropriate view
    columns = set(df.columns)

    if "score" in columns and "avg_score" in columns:
        _render_rl_log(df, selected)
    elif "loss" in columns:
        _render_swarm_log(df, selected)
    else:
        st.info(f"Columns: {list(df.columns)}")
        st.dataframe(df)


def _render_rl_log(df, filename):
    """Render RL / Holographic Swarm training logs."""
    col1, col2, col3, col4 = st.columns(4)

    final_avg = df["avg_score"].iloc[-1] if len(df) > 0 else 0
    max_score = df["score"].max() if "score" in df.columns else 0
    total_eps = len(df)

    col1.metric("📈 Final Avg Score", f"{final_avg:.1f}")
    col2.metric("🏆 Peak Score", f"{max_score:.0f}")
    col3.metric("🔄 Episodes", total_eps)

    if "hidden_size" in df.columns:
        final_hidden = df["hidden_size"].iloc[-1]
        col4.metric("🧠 Final Brain Size", int(final_hidden))
    elif "agent_hidden_sizes" in df.columns:
        col4.metric("👥 Agents", df["agent_hidden_sizes"].iloc[-1])

    # Animated playback
    st.divider()
    animate = st.toggle("🎬 Animate Playback", value=False)

    if animate:
        speed = st.slider("Playback Speed", 1, 50, 10, help="Episodes per frame")
        chart_placeholder = st.empty()
        progress_bar = st.progress(0)

        for i in range(speed, len(df) + 1, speed):
            subset = df.iloc[:i]
            fig = _build_rl_chart(subset, filename)
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"anim_{i}")
            progress_bar.progress(min(i / len(df), 1.0))
            time.sleep(0.05)

        # Final frame
        fig = _build_rl_chart(df, filename)
        chart_placeholder.plotly_chart(fig, use_container_width=True, key="anim_final")
        progress_bar.progress(1.0)
    else:
        fig = _build_rl_chart(df, filename)
        st.plotly_chart(fig, use_container_width=True)

    # Entropy production (if available)
    if "entropy_production" in df.columns:
        st.subheader("🌡️ Entropy Production (Gradient Norm)")
        fig_ent = go.Figure()
        fig_ent.add_trace(go.Scatter(
            x=df["episode"], y=df["entropy_production"],
            mode="lines", name="σ (grad norm)",
            line=dict(color="#f97316", width=1),
            fill="tozeroy", fillcolor="rgba(249, 115, 22, 0.1)",
        ))
        fig_ent.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            xaxis_title="Episode",
            yaxis_title="Entropy Production",
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_ent, use_container_width=True)

    # Data table
    with st.expander("📋 Raw Data"):
        st.dataframe(df, use_container_width=True)


def _build_rl_chart(df, title):
    """Build the main score chart for RL logs."""
    has_hidden = "hidden_size" in df.columns

    fig = make_subplots(
        rows=2 if has_hidden else 1, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3] if has_hidden else [1.0],
        vertical_spacing=0.08,
    )

    # Score trace
    fig.add_trace(go.Scatter(
        x=df["episode"], y=df["score"],
        mode="lines", name="Score",
        line=dict(color="rgba(99, 102, 241, 0.3)", width=1),
    ), row=1, col=1)

    # Avg score trace
    fig.add_trace(go.Scatter(
        x=df["episode"], y=df["avg_score"],
        mode="lines", name="Avg Score",
        line=dict(color="#818cf8", width=3),
    ), row=1, col=1)

    # Hidden size timeline
    if has_hidden:
        fig.add_trace(go.Scatter(
            x=df["episode"], y=df["hidden_size"],
            mode="lines", name="Hidden Size",
            line=dict(color="#34d399", width=2),
            fill="tozeroy", fillcolor="rgba(52, 211, 153, 0.15)",
        ), row=2, col=1)
        fig.update_yaxes(title_text="Neurons", row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=500 if has_hidden else 350,
        title=dict(text=title.replace(".csv", "").replace("_", " ").title(), font=dict(size=16)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_yaxes(title_text="Score", row=1, col=1)

    return fig


def _render_swarm_log(df, filename):
    """Render swarm training logs (epoch, loss)."""
    col1, col2 = st.columns(2)
    col1.metric("📉 Final Loss", f"{df['loss'].iloc[-1]:.6f}")
    col2.metric("🔄 Epochs", len(df))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["epoch"], y=df["loss"],
        mode="lines", name="Loss",
        line=dict(color="#818cf8", width=2),
        fill="tozeroy", fillcolor="rgba(99, 102, 241, 0.1)",
    ))

    # Detect mutation points (sudden loss jumps)
    if len(df) > 1:
        diffs = df["loss"].diff()
        mutations = df[diffs > diffs.std() * 2]
        if len(mutations) > 0:
            fig.add_trace(go.Scatter(
                x=mutations["epoch"], y=mutations["loss"],
                mode="markers", name="Mutation",
                marker=dict(color="#ef4444", size=10, symbol="triangle-up"),
            ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        title=dict(text=filename.replace(".csv", "").replace("_", " ").title(), font=dict(size=16)),
        xaxis_title="Epoch",
        yaxis_title="Loss",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Raw Data"):
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(layout="wide")
    render()
