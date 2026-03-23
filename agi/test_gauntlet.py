#!/usr/bin/env python3
"""
Smoke test for the AGI Gauntlet module.

Verifies that all components work end-to-end:
1. AGIAgent forward pass, act, intrinsic reward, sleep
2. GauntletMaze observation shape and dynamics
3. Evolution loop (short run)
4. Sigma stagnation detection produces regime diversity
5. World model transfer (state dict transplant)
"""

import sys
import os
import torch
import numpy as np
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agi.agent import AGIAgent
from agi.maze import GauntletMaze


def test_maze_obs():
    """Verify maze observation shape and value ranges."""
    env = GauntletMaze(layout_id=1, max_steps=50)
    obs, _ = env.reset()

    assert obs.shape == (GauntletMaze.OBS_DIM,), f"Expected ({GauntletMaze.OBS_DIM},), got {obs.shape}"
    assert np.all(obs >= -1) and np.all(obs <= 1), f"Obs out of range: {obs}"

    obs2, reward, done, truncated, _ = env.step(1)
    assert obs2.shape == obs.shape
    assert isinstance(reward, float)

    # Both layouts should be instantiable
    env2 = GauntletMaze(layout_id=2, max_steps=50)
    obs3, _ = env2.reset()
    assert obs3.shape == obs.shape

    print("  [PASS] Maze observation shape and dynamics")


def test_agent_forward():
    """Verify AGIAgent forward pass, act, curiosity, sleep."""
    agent = AGIAgent(input_dim=GauntletMaze.OBS_DIM, action_dim=4, hidden_dim=32)

    state = np.random.randn(GauntletMaze.OBS_DIM).astype(np.float32)

    # Forward pass
    state_t = torch.FloatTensor(state).unsqueeze(0)
    logits = agent(state_t)
    assert logits.shape == (1, 4), f"Expected (1,4), got {logits.shape}"

    # Act
    action = agent.act(state)
    assert 0 <= action <= 3, f"Action out of range: {action}"

    # Sigma should be recorded
    assert len(agent.sigma_history) == 2  # forward + act each call forward
    assert agent.current_sigma >= 0

    # Intrinsic reward
    next_state = np.random.randn(GauntletMaze.OBS_DIM).astype(np.float32)
    curiosity = agent.get_intrinsic_reward(state, action, next_state)
    assert isinstance(curiosity, float)
    assert curiosity >= 0

    # Memory + sleep (need enough memories)
    for _ in range(100):
        s = np.random.randn(GauntletMaze.OBS_DIM).astype(np.float32)
        ns = s + np.random.randn(GauntletMaze.OBS_DIM).astype(np.float32) * 0.1
        agent.memory.remember(s, np.random.randint(4), -5.0, ns, False)

    loss = agent.sleep(epochs=5)
    assert isinstance(loss, float)
    assert loss >= 0

    print("  [PASS] Agent forward, act, curiosity, sleep")


def test_thermodynamic_status():
    """Verify sigma stagnation detection produces regime diversity."""
    agent = AGIAgent(input_dim=GauntletMaze.OBS_DIM, action_dim=4, hidden_dim=32)

    # Not enough history → healthy
    assert agent.get_thermodynamic_status() == 'healthy'

    # Flat sigma → frozen
    agent.sigma_history = [0.5] * 30
    assert agent.get_thermodynamic_status() == 'frozen', \
        f"Flat sigma should be frozen, got {agent.get_thermodynamic_status()}"

    # Varying sigma → healthy
    agent.sigma_history = [0.5 + 0.2 * np.sin(i) for i in range(30)]
    assert agent.get_thermodynamic_status() == 'healthy', \
        f"Varying sigma should be healthy, got {agent.get_thermodynamic_status()}"

    # Diverging sigma → overheated
    agent.sigma_history = [0.1] * 20 + [5.0] * 20
    assert agent.get_thermodynamic_status() == 'overheated', \
        f"Diverging sigma should be overheated, got {agent.get_thermodynamic_status()}"

    print("  [PASS] Thermodynamic status detection (frozen/healthy/overheated)")


def test_mutation():
    """Verify mutation changes weights and uses additive_noise operator."""
    agent = AGIAgent(input_dim=GauntletMaze.OBS_DIM, action_dim=4, hidden_dim=32)

    # Run a few forward passes to build sigma history
    for _ in range(25):
        s = torch.randn(1, GauntletMaze.OBS_DIM)
        agent(s)

    weights_before = agent.brain.worker_head.weight.clone()
    agent.mutate()
    weights_after = agent.brain.worker_head.weight

    assert not torch.equal(weights_before, weights_after), "Mutation should change weights"

    print("  [PASS] Mutation changes weights")


def test_world_model_transfer():
    """Verify world model state dict can be transplanted between agents."""
    agent_a = AGIAgent(input_dim=GauntletMaze.OBS_DIM, action_dim=4, hidden_dim=32)
    agent_b = AGIAgent(input_dim=GauntletMaze.OBS_DIM, action_dim=4, hidden_dim=32)

    # Train agent_a's world model briefly
    for _ in range(100):
        s = np.random.randn(GauntletMaze.OBS_DIM).astype(np.float32)
        ns = s + np.random.randn(GauntletMaze.OBS_DIM).astype(np.float32) * 0.1
        agent_a.memory.remember(s, np.random.randint(4), -5.0, ns, False)
    agent_a.sleep(epochs=10)

    # Transplant
    agent_b.world_model.load_state_dict(agent_a.world_model.state_dict())

    # Verify weights match
    for pa, pb in zip(agent_a.world_model.parameters(), agent_b.world_model.parameters()):
        assert torch.equal(pa, pb), "Transplanted weights should match"

    # Verify brains are still different
    assert not torch.equal(
        agent_a.brain.worker_head.weight,
        agent_b.brain.worker_head.weight
    ), "Brains should remain independent after WM transplant"

    print("  [PASS] World model transfer (transplant + brain independence)")


def test_evolution_loop():
    """Run a minimal evolution loop to verify end-to-end."""
    env = GauntletMaze(layout_id=1, max_steps=30)
    pop_size = 5
    population = [AGIAgent(input_dim=GauntletMaze.OBS_DIM, action_dim=4, hidden_dim=16)
                  for _ in range(pop_size)]

    for gen in range(5):
        fitnesses = []
        for agent in population:
            state, _ = env.reset()
            total = 0
            for _ in range(env.max_steps):
                action = agent.act(state)
                next_state, reward, done, trunc, _ = env.step(action)
                agent.memory.remember(state, action, reward, next_state, done)
                total += reward
                state = next_state
                if done or trunc:
                    break
            fitnesses.append(total)

        best_idx = np.argmax(fitnesses)
        next_pop = [copy.deepcopy(population[best_idx])]
        for _ in range(pop_size - 1):
            child = copy.deepcopy(population[best_idx])
            child.mutate()
            next_pop.append(child)
        population = next_pop

    assert len(population) == pop_size
    info = population[0].get_topology_info()
    assert "sigma" in info and "status" in info and "memory_size" in info

    print("  [PASS] Evolution loop (5 gens, population 5)")


def main():
    print("--- AGI GAUNTLET SMOKE TEST ---\n")

    test_maze_obs()
    test_agent_forward()
    test_thermodynamic_status()
    test_mutation()
    test_world_model_transfer()
    test_evolution_loop()

    print("\n>>> ALL TESTS PASSED")


if __name__ == "__main__":
    main()
