# Changelog

All notable changes to the "Thermodynamic AI" project will be documented in this file.

## [Unreleased] - Current State
### Added
- **Robustness Testing**: Created `experiments/test_robustness.py` to perform ablation studies (Chaos vs. Static) and stability checks across multiple seeds.
- **Project Reorganization**: Refactored the codebase into a modular structure (`core/`, `agents/`, `experiments/`, `physics/`, `visualization/`, `docs/`).
- **Reinforcement Learning Adapter**: Created `experiments/run_rl.py` to apply the Thermodynamic AI architecture to OpenAI Gymnasium environments (CartPole).
    - **Evolving Policy**: A policy network that dynamically resizes its hidden layer based on reward stagnation.
    - **RL Visualization**: Added `visualization/plot_rl.py` to plot the agent's score and brain size over time.
- **Stochastic Physics Engine**: Upgraded the physics core to use Euler-Maruyama integration for coupled SDEs.
    - **Thermal Noise**: Added Langevin dynamics ($\sqrt{2D}\xi(t)$) to the physical substrate.
    - **Verification Suite**: `physics/tests/test_bound.py` now performs a phase sweep over coupling strength $\alpha$ with thermal noise.
- **Holographic Channel**: A new module (`HolographicChannel`) that simulates physical communication constraints (Bekenstein Bandwidth, Hawking Noise).
- **Cosmological Scaler**: A physics engine (`CosmologicalScaler`) that calculates theoretical limits for physical systems.
- **Thermodynamic Swarm Architecture**: Replaced the single-agent model with a multi-agent swarm (`SwarmAgent` A, B, C).

### Changed
- **Documentation**: Moved all markdown files to `docs/` and updated them to reflect the new structure and RL capabilities.
- **Theory Integration**: Updated `docs/UNIFIED_THEORY.md` to include the stochastic mechanism.
- **Training Loop**: Refactored `train_chaotic_agent` to `train_swarm` in `experiments/run_swarm.py`.

## [0.3.0] - Chemical Phase Shifts
### Added
- **Activation Function Mutation**: The `ChaosInjector` can now swap activation functions (ReLU <-> Tanh <-> GELU).
- **Safety Clamps**: Added overflow protection to the `LorenzGenerator`.

## [0.2.0] - The Thermodynamic Engine
### Added
- **Mackey-Glass Generator**: Replaced random noise with a chaotic time series dataset.
- **Topological Mutation**: Implemented dynamic resizing of the hidden layer.
- **Memory Preservation**: Added `_transfer_weights` to copy learned weights.
- **Architecture Monitor**: A "Thermostat" that detects loss plateaus.
- **Visualization**: Added `visualize.py`.
- **Logging**: Added CSV logging.

## [0.1.0] - Initial Prototype
### Added
- Basic `LorenzGenerator` for deterministic chaos.
- Simple MLP structure.
- Basic training loop.
