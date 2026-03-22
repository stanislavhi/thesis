# 🔥 THESIS — Thermodynamic Heat via Structural Instability of Self-modeling Systems

A self-modifying AI architecture governed by a falsifiable thermodynamic bound.

**Core claim**: Any physical system attempting complete self-modeling is thermodynamically forbidden from doing so. The bound **σ² · ε ≥ C_phys** is derived from first principles with no free parameters.

## ✨ Highlights

| Result | Description |
|--------|-------------|
| **20/20 Bound Validation** | σ²·ε ≥ C_phys holds across all coupling regimes |
| **Blind Swarm Solves CartPole** | Two agents seeing only half the state learn to communicate |
| **Transfer Shock Recovery** | Chaos agent scores 118.6 vs Static 10.8 after action swap |
| **ARC-AGI Perfect Solve** | Swarm solver achieves 1.0 fitness on color mapping tasks |
| **AGI Gauntlet** | 3-phase maze test: explore → sleep → transfer with world models |
| **Thermodynamic Cortex** | LLM meta-controller that detects cognitive freeze via entropy |

## 📖 Theory
- Full derivation: **[docs/SUMMARY.MD](docs/SUMMARY.MD)**
- Research write-up: **[docs/PAPER.md](docs/PAPER.md)**
- Background: **[docs/THEORY.md](docs/THEORY.md)** | **[docs/UNIFIED_THEORY.md](docs/UNIFIED_THEORY.md)**

## 📂 Structure
| Directory | Contents |
|-----------|----------|
| `core/` | Chaos engine (Lorenz), architecture monitor, ABCs |
| `agents/` | RL policy, swarm, thermodynamic agent, holographic channel, cortex client |
| `agi/` | AGI agent (memory, world model, hierarchy, curiosity) + gauntlet test |
| `physics/` | Coupled ODE solver, entropy, KL, substrate models, transformer bound |
| `experiments/` | RL, maze, stress tests, grand challenge, reasoning tests |
| `arc/` | ARC-AGI program synthesis (DSL, evolver, hybrid, swarm, macros) |
| `qwen/` | Thermodynamic Qwen LLM inference with adaptive temperature |
| `dashboard/` | Interactive Streamlit dashboard (5 pages) |
| `docs/` | Theory, paper, roadmap, changelog |

## 📈 Experimental Results

The theoretical bound ($\sigma^2 \cdot \epsilon \ge C_{phys}$) has been empirically validated across several intense cognitive stress tests:

1. **Holographic Swarm (Grand Challenge)**
   - **Task:** CartPole-v1, using two separate agents ("Position" and "Angle") that only see half of the environment state.
   - **Result:** The agents successfully learn to communicate floating-point "thoughts" through a noisy, low-bandwidth Holographic Channel, collectively solving the environment through thermodynamic optimization.
2. **Transfer Shock Recovery**
   - **Task:** The environment's action mappings are inverted mid-training.
   - **Result:** A standard static RL agent's performance permanently collapses (score ~10.8). The Thermodynamic Agent detects the entropy spike, autonomously injects phase-transition chaos, and completely recovers (score ~118.6).
3. **Severe Brain Damage (LunarLander)**
   - **Task:** 50% of the neural network's weights are zeroed out (destroyed) during flight.
   - **Result:** The agent enters a "cognitive freeze". By applying the **Thermodynamic Operator Selection Rule** (Targeted Dropout for high $C_V$ systems), the system surgically rewires its remaining healthy neurons to regain flight control without catastrophic forgetting.

## 🚀 Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Core experiments
python3 experiments/run_rl.py                                      # RL agent (CartPole)
python3 experiments/grand_challenge/run_holographic_swarm.py       # Blind swarm
python3 experiments/maze_runner.py                                 # Maze navigation

# Stress tests
python3 experiments/stress_tests/noise_test.py            # Sensory noise resilience
python3 experiments/stress_tests/brain_damage_test.py     # 50% weight destruction
python3 experiments/stress_tests/transfer_shock_test.py   # Mid-training action swap

# AGI gauntlet (explore → sleep → transfer)
python3 agi/run_gauntlet.py

# Thermodynamic cortex (requires LM Studio running)
python3 agents/real_world/cortex_client.py

# Physics verification
python3 physics/tests/test_bound.py            # σ²·ε ≥ C_phys (VALID 20/20)

# ARC-AGI solver (3 modes)
python3 arc/solver.py --task 0d3d703e --generations 200   # Standard
# Use dashboard for Hybrid / Swarm modes

# Interactive dashboard
cd dashboard && streamlit run app.py
```

## 🛠 Requirements
See [requirements.txt](requirements.txt): Python 3.x, PyTorch, NumPy, SciPy, Matplotlib, Gymnasium, Pandas, Streamlit, Plotly.

> **Note**: LunarLander requires `gymnasium[box2d]` → Python ≤ 3.12. Cortex client requires `openai` + [LM Studio](https://lmstudio.ai/) running locally.

## 📊 Dashboard

Launch with `cd dashboard && streamlit run app.py`:
- **📊 Experiment Replayer** — Load CSV logs, animated chart playback
- **🔬 Physics Sandbox** — Interactive bound verification with sliders
- **🌀 Lorenz Explorer** — 3D attractor colored by mutation strength
- **🚀 Live Training** — Run experiments from the browser
- **🧩 ARC-AGI Solver** — Visual grid puzzles + thermodynamic solver

Full documentation: **[docs/README.md](docs/README.md)** | Changelog: **[docs/CHANGELOG.md](docs/CHANGELOG.md)**
