# Session Checkpoint — 2026-03-24

## Branch: `feature/experiments-review`

## Completed This Session

### 1. Experiments Review — Bug Fixes (14 files)
- `brain_damage_test.py` — Removed over-restrictive `not is_recovering` + `std < 10` guards from stagnation condition; frozen + underperforming now triggers mutation
- `ablation_lunar_lander.py` — `k = max(1, int(...))` prevents rounding to 0 on tiny layers
- `run_holographic_swarm.py` — CSV field quoted to prevent comma-delimited `agent_sizes` from breaking parsing
- `swarm_resilience_test.py` — Rebuild `all_params` list after mutation (old list held stale refs to pre-mutation parameters)
- `prospective_operator_test.py` — Removed dead `-500` padding loop (truncation already handles unequal lengths)
- `noise_test.py` — Replaced last-value padding with truncation to shortest trial (prevents inflated final scores)
- `lunar_lander_gradual_degradation.py` — Fixed smooth() x-axis alignment (smoothing shortens array by box_pts-1)
- `lunar_lander_targeted_damage.py` — Same smooth() x-axis alignment fix
- `lunar_lander_brain_damage.py` — Removed debug print
- `autonomous_reasoning_test.py` — Removed emoji from print output
- `ablation_robustness_test.py` — Added `hasattr` guard on `sigma_history` reset
- `maze_runner.py` — Children now cycle across all elites (was clone-from-best-only, wasting second elite)

### 2. Experiments Refactor — Shared Utilities
- Created `experiments/utils.py` with shared code:
  - `reinforce_update()` — REINFORCE policy gradient step with baseline subtraction, optional gradient clipping, returns grad_norm
  - `inflict_brain_damage()` — weight zeroing with damage ratio
  - `smooth()` / `smooth_with_x()` — moving average with proper x-axis alignment
  - `InvertibleEnv` — action-swapping wrapper, auto-detects 2/3/N-action spaces
  - `EnvironmentShockWrapper` — multi-mode shock (swap_actions, invert_rewards, noisy_obs)
  - `AblationInjector` — additive noise vs targeted dropout mutation operator
- Updated 14 experiment files to use shared utilities
- Eliminated ~350 lines of duplicated REINFORCE logic, damage functions, env wrappers, smoothing code
- All files syntax-checked and import-tested

## Prior Session (2026-03-23, merged to main)
- AGI module full refactor (bug fixes, 15x15 maze, stagnation detection, tournament ES, world model transfer)
- Code quality cleanup across project (thermo_injector, dynamics, holographic swarm)
- See git history on main for details

## Current State

| Item | Status |
|------|--------|
| `feature/experiments-review` branch | Ready to commit, not yet merged |
| All experiment imports | Verified passing |
| `experiments/utils.py` | New shared utility module |
| Bug fixes | 12 fixes across 14 files |
| REINFORCE dedup | 10 files migrated to shared `reinforce_update()` |

## Open Issues

- None blocking. All files parse and import successfully.
- `experiments/maze_runner.py` still uses old `SimpleMazeEnv` (not the new `GauntletMaze`). Low priority — it's a separate visualization experiment.
- `sys.path.append` hacks remain in every file. Would require package restructuring to eliminate — deferred.

## Next Steps

1. **Merge `feature/experiments-review` to main** (or create PR)
2. **Review `arc/` module** — ARC-AGI solver, most thesis-relevant unreviewed module
3. **Review `dashboard/`** — Streamlit UI, presentation layer
4. **Review `qwen/`** — Separate sub-project, thermodynamic-aware LLM inference
5. **Update `CLAUDE.md`** commands section with new test paths
6. **Add `run_all_experiments.py`** for reproducibility (single script to generate all figures)
7. **Consider updating paper** (`paper/main.tex`) with new results
