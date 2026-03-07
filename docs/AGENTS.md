# The Agents of Chaos

This system is designed as a set of interacting "agents" or microservices, each with a specific role in the thermodynamic evolution of the AI.

## 1. The Swarm (The Local Observers)
*   **Role**: A collection of small neural networks (Agents A, B, C).
*   **Function**: Each agent observes only a partial slice of the input data (simulating a local reference frame). They compress their observations into "Thought Vectors".
*   **Constraint**: They cannot see the global state directly; they must infer it through collaboration.

## 2. The Evolving Policy (The RL Agent)
*   **Role**: A specialized agent for Reinforcement Learning tasks.
*   **Function**: Maps game states to action probabilities.
*   **Evolution**: Starts with a minimal topology (e.g., 4 neurons) and grows/shrinks based on reward stagnation, driven by the Chaos Injector.

## 3. The Holographic Channel (The Medium)
*   **Role**: The communication link between the Swarm and the Aggregator.
*   **Function**:
    *   **Bekenstein Limit**: Constrains the bandwidth (size of thought vectors) based on the physical capacity of the system (calculated by the `CosmologicalScaler`).
    *   **Hawking Noise**: Injects Gaussian noise into the transmission. The noise level is proportional to the system's "temperature" (current loss), making communication unreliable when the system is struggling.

## 4. The Aggregator (The Global Mind)
*   **Role**: The central processing unit.
*   **Function**: Receives the noisy, compressed thought vectors from the Swarm and synthesizes a final prediction.

## 5. The Monitor (The Thermostat)
*   **Role**: Detects "Thermodynamic Barriers" (local minima).
*   **Function**: Tracks the variance of the loss (or reward). If the system stagnates, it signals for a "Chaotic Push".

## 6. The Chaos Injector (The Mutator)
*   **Role**: Applies the "Discontinuous Leap".
*   **Function**:
    *   **Topological Mutation**: Resizes individual agents and the aggregator based on the Lorenz Attractor.
    *   **Chemical Phase Shift**: Swaps activation functions (ReLU <-> Tanh <-> GELU) during high-chaos states.
    *   **Memory Preservation**: Transfers weights to the new architecture to simulate neuroplasticity.

## 7. The Chaos Engine (Lorenz Generator)
*   **Role**: Provides the deterministic chaos signal.
*   **Function**: Simulates the Lorenz Attractor. The `z` component drives the magnitude and direction of mutations.

## 8. The Cosmological Scaler (The Physics Engine)
*   **Role**: Grounds the simulation in physical reality.
*   **Function**: Calculates the theoretical limits (Bit Capacity, Energy Cost, Ink Color) for a given physical system (Universe, Black Hole, Brain) to set the constraints for the Holographic Channel.
