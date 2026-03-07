# Thermodynamic AI

A falsifiable theory of self-reference, implemented as an AI architecture that evolves under thermodynamic constraints.

**Core claim**: Any physical system attempting complete self-modeling is thermodynamically forbidden from doing so. The bound **σ² · ε ≥ C_phys** is derived from first principles with no free parameters.

## 📖 Theory
Read the full derivation: **[docs/THEORY.md](docs/THEORY.md)** | **[docs/UNIFIED_THEORY.md](docs/UNIFIED_THEORY.md)**

Complete summary: **[docs/SUMMARY.MD](docs/SUMMARY.MD)**

## 📂 Structure
| Directory | Contents |
|-----------|----------|
| `core/` | Chaos engine, architecture monitor, cosmological scaler, ABCs |
| `agents/` | Evolving RL policy, swarm agents, holographic channel |
| `physics/` | Coupled ODE solver, entropy, KL divergence, substrate models |
| `experiments/` | Swarm, RL, robustness, grand challenge, stress tests |
| `visualization/` | Training plotters for swarm and RL |
| `docs/` | Theory, roadmap, changelog, agent descriptions |

## 🚀 Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run experiments
python3 experiments/run_rl.py              # RL agent evolving on CartPole
python3 experiments/run_swarm.py           # Multi-agent swarm simulation

# Verify the thermodynamic bound
python3 physics/tests/test_bound.py        # σ²·ε ≥ C_phys (should be VALID 20/20)
python3 physics/tests/test_static_model.py # Static target convergence test
python3 physics/tests/test_localization.py  # Localization limit test

# Physics experiments
python3 physics/experiments/sweep_eta.py      # Cost of speed
python3 physics/experiments/sweep_barrier.py   # Stability vs entropy

# Robustness & stress tests
python3 experiments/test_robustness.py                # Chaos vs Static ablation
python3 experiments/stress_tests/noise_test.py        # Sensory noise resilience
python3 experiments/grand_challenge/run_holographic_swarm.py  # Blind agents

# Visualize results
python3 visualization/plot_rl.py
python3 visualization/plot_swarm.py
```

## 🛠 Requirements
See [requirements.txt](requirements.txt): Python 3.x, PyTorch, NumPy, SciPy, Matplotlib, Gymnasium, Pandas.

Full documentation: **[docs/README.md](docs/README.md)**
