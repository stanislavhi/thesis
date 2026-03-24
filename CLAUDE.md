# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Self-modifying AI architecture proving a falsifiable thermodynamic bound: **σ² · ε ≥ C_phys** — complete self-modeling is thermodynamically forbidden. Agents detect stagnation via entropy and inject Lorenz-attractor chaos to escape local optima.

Python 3.12 · PyTorch · No build system — just `pip install -r requirements.txt`.

## Commands

```bash
# Experiments
python3 experiments/run_rl.py                                    # RL (CartPole/LunarLander)
python3 experiments/grand_challenge/run_holographic_swarm.py     # Blind swarm
python3 agi/run_gauntlet.py                                      # AGI maze gauntlet
python3 arc/solver.py --task 0d3d703e --generations 200          # ARC-AGI solver

# Tests (standalone scripts, no pytest)
python3 physics/tests/test_bound.py                              # Core bound (10/10 regimes, Milstein)
python3 physics/core/dynamics_n_state.py                         # N-state α_crit universality
python3 physics/tests/test_bound_deterministic.py
python3 experiments/test_robustness.py                           # Chaotic vs static
python3 agi/test_gauntlet.py                                     # AGI module smoke tests (6 tests)

# Stress tests
python3 experiments/stress_tests/brain_damage_test.py            # 50% weight destruction (CartPole + LunarLander)
python3 experiments/stress_tests/transfer_shock_test.py          # Action swap recovery
python3 experiments/stress_tests/noise_test.py                   # 10x sensory noise
python3 experiments/prospective_operator_test.py                 # Acrobot operator selection

# Dashboard
cd dashboard && streamlit run app.py

# Qwen (separate sub-project with its own requirements.txt)
cd qwen && pip install -r requirements.txt
python3 qwen/main.py
```

## Architecture

**Training loop**: config (`core/config.json`) → REINFORCE episodes → `ArchitectureMonitor` detects stagnation → `LorenzGenerator` chaos drives topology mutation (grow/shrink hidden layers) → weight transfer preserves learning → log to `logs/` CSV.

**Modules**:
- **core/** — ABCs (`EvolutionaryAgent`, `Mutator`), chaos engine, stagnation monitor, config manager, holographic scaler
- **agents/** — RL policy (REINFORCE + mutation), swarm (holographic channel), thermodynamic (σ-based health), LLM cortex (LM Studio)
- **agi/** — Hippocampus (memory), WorldModel (prediction + curiosity), HierarchicalController (manager→worker), GauntletMaze (15x15, 8D obs)
- **arc/** — DSL grid operations, genetic ProgramEvolver, heuristic GridAnalyzer, self-inventing macros
- **physics/** — Milstein ODE solver, Schnakenberg entropy, double-well/Kramers substrates, N-state simplex dynamics
- **qwen/** — Thermodynamic-aware Qwen LLM inference with chaos-driven sampling (separate `setup.py`)
- **experiments/** — Training scripts, stress tests, ablations, prospective operator test. Shared utilities in `experiments/utils.py`
- **dashboard/** — 5-page Streamlit UI (replayer, physics sandbox, Lorenz explorer, live training, ARC solver)

**Key patterns**:
- Lorenz chaos (deterministic, not random) for escaping optima
- Holographic channel: Bekenstein-limited communication + Hawking noise ∝ current loss
- Thermodynamic self-diagnosis: σ from hidden-layer variance → healthy/frozen/overheated
- Weight transfer across topology changes prevents catastrophic forgetting
- Operator selection rule: additive noise for low C_V (small nets), targeted dropout for high C_V (large nets). In pure ES contexts (no gradient recovery), always use additive_noise.
- Stagnation detection: CV of recent sigma < 5% = frozen, 5x baseline divergence = overheated (architecture-independent, replaces absolute thresholds)

**Config**: `core/config.json` keyed by environment (`CartPole-v1`, `LunarLander-v3`, `agi_gauntlet`). Access via `ConfigManager.load_config()`.

## Conventions

- **Shared experiment utilities**: New experiment code should import from `experiments/utils.py` (REINFORCE step, brain damage, smoothing, env wrappers, ablation injector) instead of duplicating logic.
- **Virtualenv**: Use `.venv3/bin/python3`, not system python.
- **Checkpoint**: `CHECKPOINT.md` in project root — records session state for context continuity.
- **Pure ES vs gradient**: The thesis uses evolutionary/thermodynamic self-modification. Do not introduce gradient descent (REINFORCE, backprop) into the core evolutionary loop. REINFORCE exists only in the RL experiment scripts as a baseline comparison mechanism.
- **Stochastic physics tests**: Use multi-trajectory averaging (n_runs=10) to reduce sampling variance. Single-trajectory results are unreliable at moderate T.
- **N-state dynamics**: Perturbation must use adversarial direction `(p-q)/||p-q||`. Never use `A @ |dq/dt|` — it gets absorbed by simplex renormalization. N=2 simplex gradient is half the scalar gradient; scale eta by 2 for equivalence.
- **α_crit verification**: Test transient behavior near fixed point (small perturbation, short time). On compact domains, boundary effects force global convergence for all α — asymptotic tests are wrong.
- **Stress tests**: Parametrize `env_name` and `hidden_size` — don't hardcode a single environment.
- **LaTeX compilation**: `eval "$(/usr/libexec/path_helper)"` required for basictex PATH, then `bibtex main && pdflatex main && pdflatex main` (two passes for cross-refs and citations).
- **Logs**: Plots go to `logs/`. Directory is git-ignored; use `git add -f logs/*.png` when committing plots.

## Git
- Do not add co-authored-by lines to commits
