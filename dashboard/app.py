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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Hide Streamlit top header line */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Metric cards: Glassmorphism and glow on hover */
    div[data-testid="stMetric"] {
        background: rgba(30, 33, 48, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 240, 255, 0.15);
        border-radius: 16px;
        padding: 20px;
        transition: transform 0.2s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.2s ease-in-out;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 240, 255, 0.1), 0 4px 6px -2px rgba(0, 240, 255, 0.05);
        border-color: rgba(0, 240, 255, 0.4);
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        background: -webkit-linear-gradient(45deg, #00f0ff, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Custom expanders */
    [data-testid="stExpander"] {
        background: rgba(18, 20, 29, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        overflow: hidden;
    }
    
    [data-testid="stExpander"] details summary {
        color: #00f0ff;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        font-weight: 600;
        color: #94a3b8;
        background: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        color: #00f0ff !important;
        border-bottom-color: #00f0ff !important;
        background: rgba(0, 240, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0, 240, 255, 0.1), rgba(59, 130, 246, 0.1));
        border: 1px solid rgba(0, 240, 255, 0.3) !important;
        color: #00f0ff !important;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #00f0ff, #3b82f6) !important;
        color: #090a0f !important;
        border-color: transparent !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 240, 255, 0.3);
    }

    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Elegant custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }

    /* Headings and markdown */
    h1, h2, h3 {
        color: #f8fafc;
        letter-spacing: -0.5px;
    }

    /* Plotly chart transparent containers */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background: rgba(18, 20, 29, 0.6) !important;
        backdrop-filter: blur(8px);
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
