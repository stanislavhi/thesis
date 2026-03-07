# Thermodynamic AI: The Unified Structural Feature

This project implements a "Thermodynamic AI" system based on a unified theory of self-reference, thermodynamics, and quantum gravity.

## 📖 The Theory
Read the full white paper: **[UNIFIED_THEORY.md](UNIFIED_THEORY.md)**

The core premise is that **complete self-modeling is physically impossible**. We derived a thermodynamic inequality that proves this and implemented an AI architecture that embraces this constraint to evolve.

## 🚀 The Code

### 1. The AI Simulation (`experiments/run_swarm.py`)
A multi-agent "Swarm" that evolves via chaotic mutations.
*   **Features**: Bekenstein-limited communication, Hawking noise, Topological mutation.
*   **Run**: `python experiments/run_swarm.py`
*   **Visualize**: `python visualization/plot_swarm.py`

### 2. The RL Agent (`experiments/run_rl.py`)
An evolving agent that learns to solve tasks (CartPole) by growing its brain.
*   **Features**: Starts with a tiny brain, detects stagnation, and mutates topology to solve the task.
*   **Run**: `python experiments/run_rl.py`
*   **Visualize**: `python visualization/plot_rl.py`

### 3. The Physics Engine (`physics/`)
A rigorous solver for the coupled thermodynamic equations.
*   **Goal**: Verify the inequality $\sigma^2 \cdot \epsilon \ge C_{phys}$.
*   **Method**: Stochastic Euler-Maruyama integration of coupled ODEs with thermal noise.
*   **Run Verification**: `python physics/tests/test_bound.py`
*   **Run Experiments**:
    *   `python physics/experiments/sweep_eta.py` (Cost of Speed)
    *   `python physics/experiments/sweep_barrier.py` (Stability vs Entropy)

### 4. Robustness Testing (`experiments/test_robustness.py`)
A suite to verify the stability and necessity of the chaotic architecture.
*   **Goal**: Compare "Chaotic Evolution" vs. "Static Architecture" across multiple seeds.
*   **Run**: `python experiments/test_robustness.py`

### 5. The Grand Challenge (`experiments/grand_challenge/`)
Advanced experiments for Multi-Agent AGI.

## 📂 Project Structure
*   `docs/`: Documentation (Theory, Roadmap, Changelog).
*   `core/`: Core logic (Chaos, Monitor, Scaler).
*   `agents/`: Agent definitions (Swarm, RL Policy).
*   `experiments/`: Runnable scripts.
*   `physics/`: The Physics Engine.
*   `visualization/`: Plotting scripts.
*   `logs/`: Training logs and plots.

## 🛠 Requirements
*   Python 3.x
*   PyTorch
*   NumPy
*   Matplotlib
*   Gymnasium (`pip install gymnasium`)
