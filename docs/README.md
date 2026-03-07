# Thermodynamic AI: The Unified Structural Feature

This project implements a "Thermodynamic AI" system based on a unified theory of self-reference, thermodynamics, and quantum gravity.

## 📖 The Theory
Read the full white paper: **[UNIFIED_THEORY.md](UNIFIED_THEORY.md)**

The core premise is that **complete self-modeling is physically impossible**. We derived a thermodynamic inequality that proves this and implemented an AI architecture that embraces this constraint to evolve.

### The Falsifiable Bound
**σ² · ε ≥ k_B² · (ln2)³ · η · k_escape · ΔE · (1−2p) / C_V**

Every parameter is independently measurable. If any physical system violates this with nonzero k_escape and sustained dynamics, the theory is wrong.

## 🚀 The Code

### 1. The AI Simulation (`experiments/run_swarm.py`)
A multi-agent "Swarm" that evolves via chaotic mutations.
*   **Features**: Bekenstein-limited communication, Hawking noise, Topological mutation.
*   **Run**: `python3 experiments/run_swarm.py`
*   **Visualize**: `python3 visualization/plot_swarm.py`

### 2. The RL Agent (`experiments/run_rl.py`)
An evolving agent that learns to solve tasks (CartPole) by growing its brain.
*   **Features**: Starts with a tiny brain, detects stagnation, and mutates topology to solve the task.
*   **Run**: `python3 experiments/run_rl.py`
*   **Visualize**: `python3 visualization/plot_rl.py`

### 3. The Physics Engine (`physics/`)
A rigorous solver for the coupled thermodynamic equations.
*   **Goal**: Verify the inequality σ² · ε ≥ C_phys.
*   **Method**: Stochastic Euler-Maruyama integration of coupled SDEs with thermal noise.
*   **Run Verification**: `python3 physics/tests/test_bound.py`
*   **Run Experiments**:
    *   `python3 physics/experiments/sweep_eta.py` (Cost of Speed)
    *   `python3 physics/experiments/sweep_barrier.py` (Stability vs Entropy)

### 4. Robustness Testing (`experiments/test_robustness.py`)
A suite to verify the stability and necessity of the chaotic architecture.
*   **Goal**: Compare "Chaotic Evolution" vs. "Static Architecture" across multiple seeds.
*   **Run**: `python3 experiments/test_robustness.py`

### 5. Stress Tests (`experiments/stress_tests/`)
*   **Noise Resilience**: Tests agent performance under extreme sensory noise.
*   **Run**: `python3 experiments/stress_tests/noise_test.py`

### 6. The Grand Challenge (`experiments/grand_challenge/`)
*   **Holographic Swarm**: Two blind agents (one sees position, one sees angle) solve CartPole by communicating through a noisy channel.
*   **Run**: `python3 experiments/grand_challenge/run_holographic_swarm.py`

## 📂 Project Structure
```
├── core/          Core logic (ABCs, Chaos engine, Monitor, Scaler)
├── agents/        Agent definitions (Evolving RL Policy, Swarm, Channel)
├── physics/       Physics engine (Dynamics, Entropy, Substrate models)
├── experiments/   Runnable scripts (Swarm, RL, Robustness, Grand Challenge)
├── visualization/ Plotting scripts
├── docs/          Documentation (Theory, Roadmap, Changelog)
└── logs/          Training logs and generated plots
```

## 🛠 Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dependencies
See [requirements.txt](../requirements.txt): Python 3.x, PyTorch, NumPy, SciPy, Matplotlib, Gymnasium, Pandas.
