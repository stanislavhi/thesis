import streamlit as st

st.set_page_config(
    page_title="THESIS Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark sidebar accent */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1d29 100%);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e2130 0%, #262940 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px;
    }

    div[data-testid="stMetric"] label {
        color: #a5b4fc !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e0e7ff !important;
        font-weight: 700 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* Plotly chart containers */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar Navigation ----------
st.sidebar.markdown("# 🔥 THESIS")
st.sidebar.caption("Thermodynamic Heat via Structural\nInstability of Self-modeling Systems")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["📊 Experiment Replayer", "🔬 Physics Sandbox", "🌀 Lorenz Explorer", "🚀 Live Training", "🧩 ARC-AGI Solver"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    "**σ² · ε ≥ C_phys**\n\n"
    "*Complete self-modeling is\nthermodynamically forbidden.*"
)

# ---------- Page Routing ----------
if page == "📊 Experiment Replayer":
    from pages.experiment_replayer import render
    render()
elif page == "🔬 Physics Sandbox":
    from pages.physics_sandbox import render
    render()
elif page == "🌀 Lorenz Explorer":
    from pages.lorenz_explorer import render
    render()
elif page == "🚀 Live Training":
    from pages.live_training import render
    render()
elif page == "🧩 ARC-AGI Solver":
    from pages.arc_solver import render
    render()
