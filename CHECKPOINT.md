# Session Checkpoint — 2026-03-23

## Branch: `feature/agi-review`

## Completed This Session

### 1. Code Quality Cleanup (merged to main via PR #3)
- Fixed `NameError` crash in `run_holographic_swarm.py` (referenced deleted `ENV_CONFIGS`)
- Fixed O(P²) `dict(agent.named_parameters())` rebuild per iteration in `thermo_injector.py`
- Removed redundant `_estimate_cv` call in `ThermodynamicInjector.mutate()`
- Precomputed loop-invariant `sqrt(2T)` in `dynamics.py` and `dynamics_n_state.py`
- Removed dead commented-out print in `prospective_operator_test.py`
- Added `.gitignore` entries for LaTeX build artifacts and `.DS_Store`

### 2. AGI Module Refactor (on `feature/agi-review`, 3 commits)

**Bug Fixes:**
- Sigma now measured from worker hidden-layer activations (was goal vector variance)
- Thermodynamic status uses stagnation detection (CV < 5% = frozen, 5x baseline divergence = overheated) instead of architecture-dependent absolute thresholds
- Divergence check runs before flatness check so stably high sigma = overheated, not frozen
- Mutation always fires — injector handles magnitude scaling per status
- Salience threshold raised to 2.0 (configurable), filters low-information transitions
- AGI agent forces `additive_noise` operator (targeted_dropout permanently kills neurons in pure ES without gradient recovery)

**Refactor:**
- Created `agi/__init__.py` with module exports
- Extracted `GauntletMaze` to `agi/maze.py`
- Upgraded maze: 15x15 grid, multi-room layouts, 8D observation (position, goal direction, wall proximity)
- `WorldModel` uses `nn.Embedding` for discrete actions (was scalar float)
- `HierarchicalController` exposes worker hidden activations
- All hyperparameters centralized in `core/config.json` under `agi_gauntlet`
- Richer `get_topology_info()` (sigma, status, brain params, WM loss, memory size)
- Added `operator_override` param to `ThermodynamicInjector.mutate()` for context-dependent operator selection

**Expansion:**
- Tournament selection (k=3) replaces clone-from-best-only
- Phase 0 baseline with random mutation (no thermodynamic operator selection)
- Phase 3 world model frozen — curiosity-only exploration priors
- CSV logging to `logs/agi_gauntlet_log.csv`
- 3-curve plot: baseline vs fresh WM vs pre-trained WM transfer
- Smoke test: `agi/test_gauntlet.py` (6 tests, all passing)

**Results:**
- Thermodynamic ES outperforms random baseline (785 vs 680 fitness)
- Transfer provides faster convergence (80% solved by gen 20 vs gen 40)
- Adaptive sigma detection correctly transitions between healthy/frozen/overheated

## Current State

| Item | Status |
|------|--------|
| `feature/agi-review` branch | 3 commits ahead of main, ready to merge |
| All AGI smoke tests | Passing |
| `logs/agi_gauntlet.png` | Generated but gitignored (logs/ in .gitignore) |
| `logs/agi_gauntlet_log.csv` | Generated but gitignored |

## Open Issues

- None blocking. All tests pass, experiment produces valid results.
- `experiments/maze_runner.py` exists as a separate older experiment with seaborn heatmap — not yet updated to use new `GauntletMaze` or `AGIAgent`. Low priority.

## Next Steps

1. **Merge `feature/agi-review` to main** (or create PR)
2. **Review remaining modules** — `arc/`, `dashboard/`, `qwen/` have not been reviewed this session
3. **Update `CLAUDE.md`** commands section to include `agi/test_gauntlet.py` and updated `agi/run_gauntlet.py` usage
4. **Consider updating paper** (`paper/main.tex`) with new AGI gauntlet results if they strengthen the thesis narrative
