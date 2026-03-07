# The Agents of Chaos

This system is designed as a set of interacting "agents" or microservices, each with a specific role in the thermodynamic evolution of the AI.

## 1. The Substrate (The Worker)
*   **Role**: The neural network itself.
*   **Function**: Trains on the data (Mackey-Glass time series) using standard gradient descent.
*   **State**: Currently a simple MLP (`Linear -> Activation -> Linear`), but capable of dynamic resizing.

## 2. The Monitor (The Thermostat)
*   **Role**: Detects "Thermodynamic Barriers" (local minima).
*   **Function**: Tracks the variance of the loss over a sliding window. If the variance drops below a threshold (`is_plateaued`), it signals that the system is stuck and needs a "Chaotic Push".
*   **Analogy**: Like a thermostat that turns on the heat when the room gets too cold (stagnant).

## 3. The Chaos Injector (The Mutator)
*   **Role**: Applies the "Discontinuous Leap".
*   **Function**:
    *   **Topological Mutation**: Resizes the hidden layer based on the current state of the Lorenz Attractor.
    *   **Chemical Phase Shift**: Swaps the activation function (ReLU <-> Tanh <-> GELU) if the chaos level is high.
    *   **Memory Preservation**: Transfers weights from the old architecture to the new one to simulate neuroplasticity.
*   **Analogy**: A biological mutation engine or a cosmic ray hitting DNA.

## 4. The Chaos Engine (Lorenz Generator)
*   **Role**: Provides the deterministic chaos signal.
*   **Function**: Simulates the Lorenz Attractor (`dx/dt`, `dy/dt`, `dz/dt`). The `z` component is used to determine the magnitude and direction of the mutation.
*   **Why Chaos?**: Unlike random noise, chaos is deterministic but unpredictable, allowing the system to explore the search space in a structured yet non-repetitive way.
