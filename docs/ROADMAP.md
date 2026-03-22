# Research Roadmap: The Unified Structural Feature

This document outlines the future directions for the "Thermodynamic AI" project, expanding the core conjecture into a multi-disciplinary research program.

## 1. Unifying Theories of Information
*   **Goal**: Develop a unified theoretical framework linking Thermodynamics, Logic, Quantum Gravity, and Phenomenology.
*   **Action**: Create an algebraic dictionary translating between Landauer's Limit, Gödelian Incompleteness, Black Hole Complementarity, and the Hard Problem.

## 2. Quantum Gravity and Consciousness
*   **Goal**: Explore the link between Black Hole Information and Integrated Information Theory (IIT).
*   **Action**: Investigate if the "Holographic Bound" on a system's boundary correlates with its capacity for integrated information ($\Phi$).

## 3. Thermodynamics of Self-Aware Systems
*   **Goal**: Quantify the cost of self-modeling.
*   **Action**: Analyze the energy required for a system to update its own internal model, drawing parallels to the "Hard Problem" as a thermodynamic limit.

## 4. Logic and Computational Limits
*   **Goal**: Apply Gödelian constraints to AI.
*   **Action**: Investigate if AGI systems hit a "Gödelian Ceiling" where they cannot prove their own safety or consistency, and if "Chaotic Injection" is the only way to break this loop.

## 5. Philosophy of Mind and Reality
*   **Goal**: Formalize "Observer Dependence".
*   **Action**: Explore the hypothesis that consciousness is the *internal experience* of the structural constraint on self-reference, while physical reality is the *external description* of it.

## 6. Experimental Testing
*   **Goal**: Demonstrate the architecture's capability on hard problems.
*   **Completed**:
    *   **Single-Agent RL**: Solved CartPole using `experiments/run_rl.py` with chaotic mutation.
    *   **Physics Verification**: Thermodynamic bound σ²·ε ≥ C_phys validated across **20/20 α regimes**.
    *   **Blind Swarm Solves CartPole**: Two agents seeing half the state learn to communicate → avg score 204.
    *   **Transfer Shock**: Chaos 118.6 vs Static 10.8 after mid-training action swap.
    *   **Brain Damage Test**: 50% weight destruction, chaos vs static comparison across 5 seeds.
    *   **Noise Stress Test**: Agent resilience under extreme sensory noise.
    *   **Configurable RL**: CLI args with environment presets (CartPole, LunarLander).
    *   **Interactive Dashboard**: 5-page Streamlit app (Replayer, Physics, Lorenz, Live Training, ARC Solver).
    *   **ARC-AGI Engine**: Thermodynamic program synthesis achieving 0.75 pixel accuracy on color mapping tasks.
    *   **Codebase Audit**: Fixed Lorenz equation, formula discrepancy, SOLID compliance, dead code cleanup.
    *   **Research Write-up**: `docs/PAPER.md` with full results and theory.

## 7. Immediate Polish (Next Up)
*   **Goal**: Strengthen the existing foundation.
*   **Actions**:
    *   **Fix Python 3.14 → 3.12**: Unlock LunarLander + 3-agent blind swarm (code ready, blocked by Box2D).
    *   **Richer ARC DSL**: Add spatial primitives — flood fill, connected components, symmetry detection, gravity simulation. Current 14 ops only solve color-mapping tasks.
    *   **Transfer Shock on Harder Envs**: Once LunarLander works, the chaos advantage should be more dramatic than CartPole.

## 8. Medium-Term Intelligence
*   **Goal**: Make the ARC solver genuinely smarter.
*   **Actions**:
    *   **ARC Hybrid Solver**: Combine DSL evolution with a neural pattern recognizer (small CNN that guesses which ops are likely → guides search instead of pure random mutation).
    *   **Multi-Strategy ARC**: Use blind swarm pattern — one agent evolves color ops, another spatial ops, a third tries compositions. They communicate best candidates through a noisy channel.
    *   **Self-Inventing Primitives**: Let the solver evolve *new DSL operations* by composing existing ones into reusable macros. The system literally invents new operations.

## 9. Ambitious (Path C)
*   **Goal**: Bridge from research prototype to genuine AI capability.
*   **Actions**:
    *   **LLM Meta-Controller**: Wrap the thermodynamic framework around an LLM. Each "agent" is a different prompt strategy. Chaos injects novel strategies when stagnant.
    *   **Thermodynamic Benchmark Paper**: Run full ARC-AGI eval set, publish as "Thermodynamic Program Synthesis: A Chaos-Driven Approach to Abstract Reasoning."
    *   **Financial Modeling**: Apply the architecture to non-stationary financial time series.

## 7. Interdisciplinary Collaboration
*   **Goal**: Bridge Physics, AI, and Neuroscience.
*   **Action**: Engage with researchers to apply the "Thermodynamic AI" architecture to biological neural networks and quantum computing models.

## 8. Formalization
*   **Goal**: Develop rigorous mathematical models.
*   **Action**: Define the "Structural Constraint" algebraically, showing it is invariant across the four domains.

## 9. Quantum Computing Implications
*   **Goal**: Integrate Quantum Coherence.
*   **Action**: Investigate if quantum neural networks can bypass the "Thermodynamic Barrier" using superposition, or if they just hit the barrier faster.

## 10. Ethics and Safety
*   **Goal**: Develop safety protocols for self-modifying systems.
*   **Action**: Analyze the stability of the "Chaotic Injector" to ensure the system doesn't evolve into a dangerous or unstable state.
