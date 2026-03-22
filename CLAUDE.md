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
python3 physics/tests/test_bound.py                              # Core bound (20 regimes)
python3 physics/tests/test_bound_deterministic.py
python3 experiments/test_robustness.py                           # Chaotic vs static

# Stress tests
python3 experiments/stress_tests/brain_damage_test.py            # 50% weight destruction
python3 experiments/stress_tests/transfer_shock_test.py          # Action swap recovery
python3 experiments/stress_tests/noise_test.py                   # 10x sensory noise

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
- **agi/** — Hippocampus (memory), WorldModel (prediction + curiosity), HierarchicalController (manager→worker)
- **arc/** — DSL grid operations, genetic ProgramEvolver, heuristic GridAnalyzer, self-inventing macros
- **physics/** — Euler-Maruyama ODE solver, Schnakenberg entropy, double-well/Kramers substrates
- **qwen/** — Thermodynamic-aware Qwen LLM inference with chaos-driven sampling (separate `setup.py`)
- **experiments/** — Training scripts, stress tests, ablations
- **dashboard/** — 5-page Streamlit UI (replayer, physics sandbox, Lorenz explorer, live training, ARC solver)

**Key patterns**:
- Lorenz chaos (deterministic, not random) for escaping optima
- Holographic channel: Bekenstein-limited communication + Hawking noise ∝ current loss
- Thermodynamic self-diagnosis: σ from hidden-layer variance → healthy/frozen/overheated
- Weight transfer across topology changes prevents catastrophic forgetting

**Config**: `core/config.json` keyed by environment (`CartPole-v1`, `LunarLander-v3`). Access via `ConfigManager.load_config()`.

## Git
- Do not add co-authored-by lines to commits
