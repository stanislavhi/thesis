# Thermodynamic Barriers to Self-Modeling: A Physical Theory of Cognitive Limits

A self-modifying AI architecture proving a falsifiable thermodynamic bound: complete self-modeling is thermodynamically forbidden.

**Core result**: When a system updates its self-model, the heat dissipated perturbs the substrate being modeled, creating an infinite regress. We derive the bound **σ² · ε ≥ C_phys** from first principles — as self-model error ε → 0, entropy production σ → ∞.

## Key Results

| Result | Detail |
|--------|--------|
| **Thermodynamic Bound** | σ²·ε ≥ C_phys validated 10/10 regimes (Milstein integrator, 10-run averaging) |
| **α_crit = 1 Universal** | Critical coupling threshold independent of N, verified for N = 2, 3, 5, 10 |
| **N=2 Equivalence** | N-state dynamics match scalar to machine epsilon (KL error 8.5e-17) |
| **Operator Selection Rule** | Low C_V → additive noise, High C_V → targeted dropout |
| **Acrobot Prospective Test** | Additive noise recovers to -370 post-shift; static and dropout flatline at -500 |
| **Brain Damage Recovery** | LunarLander (high C_V): chaos 12.6 vs static 5.3 after 50% weight destruction |
| **Transfer Shock** | Chaos agent 118.6 vs static 10.8 after action inversion |
| **Holographic Swarm** | Two half-blind agents solve CartPole via Bekenstein-limited communication |

## Paper

LaTeX paper in `paper/main.tex` — 12 pages, compiles clean.
- Full derivation of σ²·ε ≥ C_phys from Landauer + Kramers
- N-state generalization on probability simplices (Section 3.6)
- Thermodynamic Operator Selection Rule
- Four-Framework Conjecture (Landauer, Gödel, Bekenstein, Hard Problem)

## Architecture

```
core/           Chaos engine (Lorenz), architecture monitor, config manager
agents/         RL policy, swarm, thermodynamic agent + injector, LLM cortex
agi/            Hippocampus, world model, hierarchical controller, gauntlet
physics/        Milstein ODE solver, entropy, KL, Kramers substrate, N-state dynamics
experiments/    RL training, stress tests (brain damage, transfer shock, noise), prospective test
arc/            ARC-AGI DSL, genetic evolver, grid analyzer, self-inventing macros
qwen/           Thermodynamic Qwen LLM inference (separate sub-project)
dashboard/      5-page Streamlit UI
paper/          LaTeX paper, references, figures
```

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "gymnasium[box2d]"  # for LunarLander

# Core experiments
python3 experiments/run_rl.py                                    # RL (CartPole/LunarLander)
python3 experiments/grand_challenge/run_holographic_swarm.py     # Blind swarm
python3 agi/run_gauntlet.py                                      # AGI maze gauntlet

# Stress tests
python3 experiments/stress_tests/brain_damage_test.py            # 50% weight destruction
python3 experiments/stress_tests/transfer_shock_test.py          # Action swap recovery
python3 experiments/stress_tests/noise_test.py                   # 10x sensory noise
python3 experiments/prospective_operator_test.py                 # Acrobot operator selection

# Physics verification
python3 physics/tests/test_bound.py                              # σ²·ε ≥ C_phys (10/10 VALID)
python3 physics/core/dynamics_n_state.py                         # N-state α_crit universality

# ARC-AGI solver
python3 arc/solver.py --task 0d3d703e --generations 200

# Dashboard
cd dashboard && streamlit run app.py
```

## Requirements

Python 3.12 · PyTorch · NumPy · SciPy · Matplotlib · Gymnasium · Pandas · Streamlit

See [requirements.txt](requirements.txt). LunarLander requires `gymnasium[box2d]`. Cortex client requires `openai` + [LM Studio](https://lmstudio.ai/).
