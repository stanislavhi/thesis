#!/usr/bin/env python3
"""
Experiment: The Maze Runner (Population-Based Evolution).

Goal: Visually demonstrate the Thermodynamic Agent escaping a local optimum in a 2D maze.

Methodology:
- A population of agents is used.
- In each generation, the top agents ("elites") are preserved.
- The rest of the population are "children" of the elites, mutated with high chaos.
- This allows for both exploitation (keeping the best solution) and exploration (risky mutations).
"""

import sys
import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agents.thermodynamic.thermo_agent import ThermodynamicAgent
from agents.thermodynamic.thermo_injector import ThermodynamicInjector
from core.chaos import LorenzGenerator

class SimpleMazeEnv(gym.Env):
    """
    A simple 10x10 maze with a deceptive trap.
    """
    def __init__(self):
        super().__init__()
        self.grid_size = 10
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4) # Up, Down, Left, Right
        
        self.layout = np.zeros((10, 10))
        self.start_pos = np.array([1, 1])
        self.goal_pos = np.array([8, 8])
        self.agent_pos = self.start_pos.copy()
        
        # The Trap Wall
        self.layout[4:9, 4] = 1 
        self.layout[4, 4:9] = 1
        
        self.max_steps = 50
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos.copy()
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        move = np.array([0, 0])
        if action == 0: move = np.array([-1, 0])
        elif action == 1: move = np.array([1, 0])
        elif action == 2: move = np.array([0, -1])
        elif action == 3: move = np.array([0, 1])
        
        new_pos = self.agent_pos + move
        new_pos = np.clip(new_pos, 0, self.grid_size - 1)
        
        if self.layout[new_pos[0], new_pos[1]] == 1:
            new_pos = self.agent_pos 
            
        self.agent_pos = new_pos
        self.current_step += 1
        
        dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        done = np.array_equal(self.agent_pos, self.goal_pos)
        
        fitness = -dist
        if done: fitness += 100.0
        
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), fitness, done, truncated, {}

    def _get_obs(self):
        return self.agent_pos.astype(np.float32) / self.grid_size

def evaluate(agent, env):
    """Run one episode and return total fitness."""
    state, _ = env.reset()
    total_fitness = 0
    path = [env.agent_pos.copy()]
    
    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = agent(state_tensor)
            action = torch.argmax(probs).item()
        
        next_state, fitness, done, truncated, _ = env.step(action)
        total_fitness += fitness
        state = next_state
        path.append(env.agent_pos.copy())
        
        if done or truncated:
            break
            
    return total_fitness, path, done

def run_maze_experiment():
    print("--- MAZE RUNNER EXPERIMENT (POPULATION-BASED ES) ---")
    
    env = SimpleMazeEnv()
    
    # --- Population Settings ---
    population_size = 20
    num_elites = 2
    generations = 300
    
    # --- Initialization ---
    population = [ThermodynamicAgent(input_dim=2, hidden_dim=32, output_dim=4) for _ in range(population_size)]
    injector = ThermodynamicInjector(LorenzGenerator(), base_mutation_rate=0.02)
    
    visit_counts = np.zeros((10, 10))
    success_history = []
    best_fitness_history = []
    
    print("Evolving Population...")
    
    for gen in range(generations):
        # 1. Evaluate Population
        fitnesses = []
        paths = []
        successes = []
        for agent in population:
            fitness, path, success = evaluate(agent, env)
            fitnesses.append(fitness)
            paths.append(path)
            successes.append(success)
            
        # 2. Selection
        sorted_indices = np.argsort(fitnesses)[::-1] # Descending
        elites = [population[i] for i in sorted_indices[:num_elites]]
        
        best_fitness = fitnesses[sorted_indices[0]]
        best_fitness_history.append(best_fitness)
        
        # 3. Reproduction & Mutation
        # STABILITY FIX: Implement strict elitism and seeding
        next_population = [copy.deepcopy(elite) for elite in elites] # Elites carry over unchanged
        
        # The rest of the population are mutated children of the BEST elite
        best_elite = elites[0]
        
        for i in range(population_size - num_elites):
            child = copy.deepcopy(best_elite)
            
            # Mutate the child
            child = injector.mutate(child)
            next_population.append(child)
            
        population = next_population
        
        # Track stats
        gen_success_rate = np.mean(successes)
        success_history.append(gen_success_rate)
        
        # Log visits of the best agent in this generation
        best_path = paths[sorted_indices[0]]
        for pos in best_path:
            visit_counts[pos[0], pos[1]] += 1
        
        if gen % 20 == 0:
            print(f"Gen {gen} | Best Fitness: {best_fitness:.2f} | Success Rate: {gen_success_rate:.0%}")
            
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(visit_counts, cmap="hot", alpha=0.8, cbar_kws={'label': 'Visit Frequency (Best Agent per Gen)'})
    
    walls = np.argwhere(env.layout == 1)
    if len(walls) > 0:
        plt.scatter(walls[:, 1] + 0.5, walls[:, 0] + 0.5, c='gray', marker='s', s=100, label='Wall')
    
    plt.scatter(env.start_pos[1] + 0.5, env.start_pos[0] + 0.5, c='green', marker='o', s=200, label='Start')
    plt.scatter(env.goal_pos[1] + 0.5, env.goal_pos[0] + 0.5, c='blue', marker='*', s=300, label='Goal')
    
    plt.title(f"Thermodynamic Population Exploration (Final Success Rate: {np.mean(success_history[-10:]):.2%})")
    plt.legend()
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/maze_runner_heatmap.png'))
    plt.savefig(output_file)
    print(f"\nHeatmap saved to {output_file}")

if __name__ == "__main__":
    run_maze_experiment()
