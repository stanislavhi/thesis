#!/usr/bin/env python3
"""
AGI Gauntlet: Evolutionary test of the Thermodynamic AGI architecture.

Phases:
  0. Baseline:      Fresh agents + random mutation on Maze B (control).
  1. Exploration:    Evolve population on Maze A with thermodynamic mutation + curiosity.
  2. Consolidation:  Elite agents sleep — train world model on salient memories.
  3. Transfer:       Fresh brains + frozen pre-trained world model on Maze B.
                     Tests if learned physics provides useful curiosity priors.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agi.agent import AGIAgent
from agi.maze import GauntletMaze
from core.config_manager import ConfigManager


def load_config():
    full = ConfigManager.load_config()
    return full.get("agi_gauntlet", {})


def make_agent(cfg):
    """Create an AGIAgent from config."""
    return AGIAgent(
        input_dim=GauntletMaze.OBS_DIM,
        action_dim=4,
        hidden_dim=cfg["hidden_dim"],
        brain_lr=cfg["brain_lr"],
        wm_lr=cfg["wm_lr"],
        base_mutation_rate=cfg["base_mutation_rate"],
        memory_capacity=cfg["memory_capacity"],
        salience_threshold=cfg["salience_threshold"],
    )


def evaluate_agent(agent, env, curiosity_weight=0.1):
    """Run one episode. Returns (fitness, solved)."""
    state, _ = env.reset()
    total_fitness = 0

    for _ in range(env.max_steps):
        action = agent.act(state)
        next_state, fitness, done, truncated, _ = env.step(action)

        curiosity = agent.get_intrinsic_reward(state, action, next_state)
        total_fitness += fitness + (curiosity * curiosity_weight)

        agent.memory.remember(state, action, fitness, next_state, done)
        state = next_state

        if done or truncated:
            break

    return total_fitness, done


def tournament_select(population, fitnesses, k=3):
    """Select one individual via tournament selection."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


def run_evolution_phase(population, env, generations, phase_name, cfg,
                        use_thermodynamic=True, log_rows=None):
    """
    Evolve a population on an environment.

    Args:
        use_thermodynamic: If False, apply uniform random noise instead of
                          thermodynamic operator selection (baseline mode).
        log_rows: If provided, append CSV rows to this list.
    """
    print(f"\n>>> PHASE: {phase_name} ({generations} generations)")

    pop_size = len(population)
    num_elites = cfg["num_elites"]
    tournament_k = cfg["tournament_k"]
    curiosity_weight = cfg["curiosity_weight"]
    best_fitness_history = []

    for gen in range(generations):
        results = [evaluate_agent(a, env, curiosity_weight) for a in population]
        fitnesses = [r[0] for r in results]
        successes = [r[1] for r in results]

        sorted_idx = np.argsort(fitnesses)[::-1]
        elites = [population[i] for i in sorted_idx[:num_elites]]

        best_fitness = fitnesses[sorted_idx[0]]
        mean_fitness = np.mean(fitnesses)
        success_rate = np.mean(successes)
        best_fitness_history.append(best_fitness)

        # Next generation: elites + tournament-selected offspring
        next_pop = [copy.deepcopy(e) for e in elites]

        for _ in range(pop_size - num_elites):
            parent = tournament_select(population, fitnesses, k=tournament_k)
            child = copy.deepcopy(parent)
            if use_thermodynamic:
                child.mutate()
            else:
                # Baseline: uniform random noise, no thermodynamic diagnosis
                with torch.no_grad():
                    for p in child.brain.parameters():
                        p.add_(torch.randn_like(p) * 0.02)
            next_pop.append(child)

        population = next_pop

        if log_rows is not None:
            best_info = elites[0].get_topology_info()
            log_rows.append(
                f"{phase_name},{gen},{best_fitness:.2f},{mean_fitness:.2f},"
                f"{success_rate:.3f},{best_info['sigma']:.4f},{best_info['wm_loss']:.4f}"
            )

        if gen % 20 == 0:
            print(f"   Gen {gen:>3d} | Best: {best_fitness:>8.1f} | "
                  f"Mean: {mean_fitness:>8.1f} | Solved: {success_rate:.0%}", flush=True)

    return population, best_fitness_history


def run_gauntlet():
    cfg = load_config()
    log_rows = []

    # --- Phase 1: Exploration on Maze A ---
    env_a = GauntletMaze(layout_id=1, max_steps=cfg["max_steps"])
    pop_a = [make_agent(cfg) for _ in range(cfg["population_size"])]
    trained_pop, history_a = run_evolution_phase(
        pop_a, env_a, cfg["generations_phase1"],
        "1_Exploration", cfg, log_rows=log_rows
    )

    # --- Phase 2: Deep Sleep (Consolidation) ---
    print("\n>>> PHASE: 2. Deep Sleep (Consolidation)")
    results = [evaluate_agent(a, env_a, cfg["curiosity_weight"]) for a in trained_pop]
    sorted_idx = np.argsort([r[0] for r in results])[::-1]
    best_elite = trained_pop[sorted_idx[0]]

    loss = best_elite.sleep(epochs=cfg["sleep_epochs"])
    mem_size = len(best_elite.memory.buffer)
    print(f"   Elite world model trained. Loss: {loss:.4f} | Memories: {mem_size}")

    # Snapshot the trained world model state dict
    trained_wm_state = copy.deepcopy(best_elite.world_model.state_dict())

    # --- Phase 0: Baseline on Maze B (random mutation, no thermodynamic) ---
    env_b = GauntletMaze(layout_id=2, max_steps=cfg["max_steps"])
    baseline_pop = [make_agent(cfg) for _ in range(cfg["population_size"])]
    _, history_baseline = run_evolution_phase(
        baseline_pop, env_b, cfg["generations_phase3"],
        "0_Baseline", cfg, use_thermodynamic=False, log_rows=log_rows
    )

    # --- Phase 3a: Transfer — fresh brain + frozen pre-trained world model ---
    transfer_pop = []
    for _ in range(cfg["population_size"]):
        agent = make_agent(cfg)
        agent.world_model.load_state_dict(trained_wm_state)
        # Freeze world model — use it only for curiosity, don't train it
        for p in agent.world_model.parameters():
            p.requires_grad = False
        transfer_pop.append(agent)

    _, history_transfer = run_evolution_phase(
        transfer_pop, env_b, cfg["generations_phase3"],
        "3a_Transfer", cfg, log_rows=log_rows
    )

    # --- Phase 3b: Control — fresh brain + fresh world model ---
    control_pop = [make_agent(cfg) for _ in range(cfg["population_size"])]
    _, history_control = run_evolution_phase(
        control_pop, env_b, cfg["generations_phase3"],
        "3b_Control", cfg, log_rows=log_rows
    )

    # --- CSV logging ---
    log_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../logs/agi_gauntlet_log.csv'))
    with open(log_path, 'w') as f:
        f.write("phase,generation,best_fitness,mean_fitness,success_rate,best_sigma,wm_loss\n")
        f.write("\n".join(log_rows))
    print(f"\nCSV saved to {log_path}")

    # --- Plotting ---
    def smooth(y, box_pts=10):
        box = np.ones(box_pts) / box_pts
        return np.convolve(y, box, mode='same')

    plt.figure(figsize=(12, 6))
    plt.plot(smooth(history_baseline), color='gray', linestyle='--', linewidth=2,
             label='Baseline (Random Mutation)')
    plt.plot(smooth(history_control), color='red', linestyle='-.', linewidth=2,
             label='Fresh World Model')
    plt.plot(smooth(history_transfer), color='blue', linestyle='-', linewidth=2.5,
             label='Pre-Trained World Model (Transfer)')

    plt.title("AGI Gauntlet: Transfer Learning via Thermodynamic World Models")
    plt.xlabel("Generation (Maze B)")
    plt.ylabel("Best Fitness")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../logs/agi_gauntlet.png'))
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    run_gauntlet()
