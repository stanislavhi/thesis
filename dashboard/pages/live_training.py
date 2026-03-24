import streamlit as st
import subprocess
import sys
import os
import time
import pandas as pd
import plotly.graph_objects as go


def render():
    st.header("🚀 Live Training")
    st.caption("Run experiments directly from the dashboard and watch results stream in.")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    venv_python = sys.executable
    logs_dir = os.path.join(project_root, 'logs')

    # Experiment selector
    col_config, col_run = st.columns([1, 2])

    with col_config:
        st.subheader("Configuration")

        experiment = st.selectbox("Experiment", [
            "RL Agent",
            "Blind Swarm",
        ])

        env = st.selectbox("Environment", ["CartPole-v1"])
        episodes = st.number_input("Episodes", min_value=10, max_value=5000, value=200, step=50)

        if experiment == "RL Agent":
            script = os.path.join(project_root, 'experiments', 'run_rl.py')
            cmd = [venv_python, script, '--env', env, '--episodes', str(episodes)]
            env_tag = env.replace("-", "_").lower()
            log_file = os.path.join(logs_dir, f'rl_{env_tag}_log.csv')
        else:
            script = os.path.join(project_root, 'experiments', 'grand_challenge', 'run_holographic_swarm.py')
            cmd = [venv_python, script, '--env', env, '--episodes', str(episodes)]
            env_tag = env.replace("-", "_").lower()
            log_file = os.path.join(logs_dir, f'holographic_swarm_{env_tag}_log.csv')

        run_button = st.button("▶️ Start Training", type="primary", use_container_width=True)

    with col_run:
        if run_button:
            _run_experiment(cmd, log_file, experiment, project_root)
        else:
            # Show most recent results if available
            _show_latest_results(logs_dir)


def _run_experiment(cmd, log_file, experiment_name, project_root):
    """Run the experiment as a subprocess and stream results."""
    st.subheader(f"Training: {experiment_name}")

    # Terminal output
    terminal = st.empty()
    chart_placeholder = st.empty()
    status = st.status(f"Running {experiment_name}...", expanded=True)

    output_lines = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=project_root,
    )

    # Stream output
    try:
        for line in iter(process.stdout.readline, ''):
            output_lines.append(line.rstrip())
            # Keep last 30 lines
            visible = output_lines[-30:]
            terminal.code("\n".join(visible), language="text")

            # Periodically update chart from log file
            if len(output_lines) % 5 == 0 and os.path.exists(log_file):
                try:
                    df = pd.read_csv(log_file)
                    if len(df) > 1 and "avg_score" in df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df["episode"], y=df["score"],
                            mode="lines", name="Score",
                            line=dict(color="rgba(99, 102, 241, 0.3)", width=1),
                        ))
                        fig.add_trace(go.Scatter(
                            x=df["episode"], y=df["avg_score"],
                            mode="lines", name="Avg Score",
                            line=dict(color="#818cf8", width=3),
                        ))
                        fig.update_layout(
                            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            height=350,
                            xaxis_title="Episode",
                            yaxis_title="Score",
                            margin=dict(l=40, r=20, t=20, b=40),
                            showlegend=True,
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

        process.wait()
    except Exception as e:
        st.error(f"Error: {e}")
        return

    # Final update
    if process.returncode == 0:
        status.update(label="✅ Training Complete!", state="complete")
    else:
        status.update(label="❌ Training Failed", state="error")

    # Final chart
    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file)
            if len(df) > 1 and "avg_score" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["episode"], y=df["score"],
                    mode="lines", name="Score",
                    line=dict(color="rgba(99, 102, 241, 0.3)", width=1),
                ))
                fig.add_trace(go.Scatter(
                    x=df["episode"], y=df["avg_score"],
                    mode="lines", name="Avg Score",
                    line=dict(color="#818cf8", width=3),
                ))
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=400,
                    title="Final Results",
                    xaxis_title="Episode",
                    yaxis_title="Score",
                    margin=dict(l=40, r=20, t=60, b=40),
                    showlegend=True,
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


def _show_latest_results(logs_dir):
    """Show the most recent log file."""
    st.subheader("Latest Results")

    csv_files = []
    for f in os.listdir(logs_dir):
        if f.endswith('.csv'):
            path = os.path.join(logs_dir, f)
            csv_files.append((os.path.getmtime(path), path, f))

    if not csv_files:
        st.info("No results yet. Click **Start Training** to run an experiment!")
        return

    csv_files.sort(reverse=True)
    latest_path = csv_files[0][1]
    latest_name = csv_files[0][2]

    try:
        df = pd.read_csv(latest_path)
        st.caption(f"Showing: **{latest_name}**")

        if "avg_score" in df.columns and len(df) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["episode"], y=df["score"],
                mode="lines", name="Score",
                line=dict(color="rgba(99, 102, 241, 0.3)", width=1),
            ))
            fig.add_trace(go.Scatter(
                x=df["episode"], y=df["avg_score"],
                mode="lines", name="Avg Score",
                line=dict(color="#818cf8", width=3),
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=400,
                title=latest_name.replace(".csv", "").replace("_", " ").title(),
                xaxis_title="Episode",
                yaxis_title="Score",
                margin=dict(l=40, r=20, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Couldn't load latest log: {e}")
