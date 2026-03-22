import streamlit as st
import numpy as np
import uuid
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from arc.data import (download_sample_tasks, list_local_tasks, download_task,
                       load_task, task_summary, grid_to_string, DATA_DIR)
from arc.solver import solve_task
from arc.hybrid_solver import solve_task_hybrid, GridAnalyzer
from arc.swarm_solver import SwarmSolver

ARC_COLORS = {
    0: "#1a1a2e", 1: "#3498db", 2: "#e74c3c", 3: "#2ecc71", 4: "#f1c40f",
    5: "#bdc3c7", 6: "#9b59b6", 7: "#e67e22", 8: "#1abc9c", 9: "#8b4513",
}


def render():
    st.header("🧩 ARC-AGI Solver")
    st.caption("Thermodynamic program synthesis for abstract reasoning puzzles.")

    col_config, col_viz = st.columns([1, 2])

    with col_config:
        st.subheader("Configuration")

        local_tasks = list_local_tasks("training")

        if not local_tasks:
            if st.button("📥 Download Sample Tasks", type="primary"):
                with st.spinner("Downloading from GitHub..."):
                    downloaded = download_sample_tasks(20)
                    st.success(f"Downloaded {len(downloaded)} tasks!")
                    st.rerun()
            st.info("No tasks downloaded yet. Click above to get started.")
            return

        selected_task = st.selectbox("Select Task", local_tasks)

        # Solver mode
        solver_mode = st.radio("Solver", [
            "🎯 Standard",
            "🧠 Hybrid (Guided)",
            "🐝 Swarm (3 Specialists)",
        ])

        generations = st.slider("Generations", 50, 500, 150, 50)
        population = st.slider("Population Size", 20, 200, 60, 20)

        solve_button = st.button("🧬 Evolve Solution", type="primary", use_container_width=True)

    with col_viz:
        if selected_task:
            task_path = os.path.join(DATA_DIR, "training", f"{selected_task}.json")
            task = load_task(task_path)

            # Task analysis
            analyzer = GridAnalyzer()
            op_weights = analyzer.analyze(task["train"])
            top_ops = sorted(op_weights.items(), key=lambda x: x[1], reverse=True)[:6]

            st.subheader("Task Analysis")
            m1, m2, m3 = st.columns(3)
            summary = task_summary(task)
            m1.metric("Train Examples", summary["n_train"])
            m2.metric("Test Examples", summary["n_test"])
            m3.metric("Top Predicted Op", top_ops[0][0])

            # Op weight bar chart
            with st.expander("🔍 Predicted Op Weights"):
                fig_ops = go.Figure(data=[go.Bar(
                    x=[n for n, _ in top_ops],
                    y=[w for _, w in top_ops],
                    marker_color=["#818cf8", "#a78bfa", "#c4b5fd", "#ddd6fe", "#ede9fe", "#f5f3ff"],
                )])
                fig_ops.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=200,
                    margin=dict(l=20, r=20, t=10, b=40),
                    yaxis_title="Weight",
                )
                st.plotly_chart(fig_ops, use_container_width=True)

            # Training examples
            st.subheader("Training Examples")
            for i, ex in enumerate(task["train"]):
                col_in, col_arrow, col_out = st.columns([3, 1, 3])
                with col_in:
                    st.caption(f"Input {i+1}")
                    _render_grid(np.array(ex["input"]))
                with col_arrow:
                    st.markdown("<br><br><h2 style='text-align:center'>→</h2>",
                                unsafe_allow_html=True)
                with col_out:
                    st.caption(f"Output {i+1}")
                    _render_grid(np.array(ex["output"]))

            st.subheader("Test Input")
            for i, ex in enumerate(task["test"]):
                _render_grid(np.array(ex["input"]))

    # Solve
    if solve_button and selected_task:
        st.divider()
        mode_name = solver_mode.split(" ", 1)[1] if " " in solver_mode else solver_mode
        st.subheader(f"🧬 Evolving Solution ({mode_name})...")

        task_path = os.path.join(DATA_DIR, "training", f"{selected_task}.json")
        task = load_task(task_path)

        progress = st.progress(0)
        t0 = time.time()

        # Run selected solver
        if "Standard" in solver_mode:
            result = solve_task(task, generations=generations,
                                population_size=population, verbose=False)
        elif "Hybrid" in solver_mode:
            result = solve_task_hybrid(task, generations=generations,
                                       population_size=population, verbose=False)
        else:  # Swarm
            swarm = SwarmSolver(population_per_specialist=max(20, population // 3),
                                share_interval=15)
            result = swarm.solve(task, generations=generations, verbose=False)

        elapsed = time.time() - t0
        progress.progress(1.0)

        # Results
        col_prog, col_fit, col_time = st.columns(3)
        col_prog.metric("🧬 Best Program", str(result["best_program"]))
        col_fit.metric("🎯 Train Fitness", f"{result['train_fitness']:.4f}")
        col_time.metric("⏱️ Time", f"{elapsed:.1f}s")

        if result["train_fitness"] >= 1.0 - 1e-6:
            st.success("🏆 PERFECT SOLUTION FOUND!")

        # Fitness history
        if result.get("fitness_history"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=result["fitness_history"],
                mode="lines",
                line=dict(color="#818cf8", width=2),
                fill="tozeroy", fillcolor="rgba(99, 102, 241, 0.1)",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250,
                title="Fitness Over Generations",
                xaxis_title="Generation", yaxis_title="Best Fitness",
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Predictions
        st.subheader("Predictions")
        for i, pred in enumerate(result["predictions"]):
            cols = st.columns(3)
            with cols[0]:
                st.caption("Test Input")
                _render_grid(np.array(task["test"][i]["input"]))
            with cols[1]:
                st.caption("Prediction")
                _render_grid(pred)
            with cols[2]:
                if "output" in task["test"][i]:
                    st.caption("Expected")
                    expected = np.array(task["test"][i]["output"])
                    _render_grid(expected)
                    if pred.shape == expected.shape and np.array_equal(pred, expected):
                        st.success("✅ CORRECT!")
                    else:
                        match = np.sum(pred == expected) / expected.size * 100 if pred.shape == expected.shape else 0
                        st.error(f"❌ {match:.0f}% match")


def _render_grid(grid: np.ndarray):
    """Render an ARC grid using Plotly heatmap."""
    h, w = grid.shape
    colorscale = [[i/9, ARC_COLORS[i]] for i in range(10)]

    fig = go.Figure(data=go.Heatmap(
        z=grid[::-1],
        colorscale=colorscale,
        zmin=0, zmax=9,
        showscale=False,
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        height=max(100, h * 25),
        width=max(100, w * 25),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render()
