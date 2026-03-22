#!/usr/bin/env python3
"""
Experiment: The AGI Gauntlet (Evolutionary Edition).

Goal: Test the full Thermodynamic AGI architecture using a robust Evolutionary Strategy.

Phases:
1. Exploration: Evolve a population to solve Maze A, rewarding curiosity.
2. Consolidation: The elite agents from Phase 1 "sleep" to train their World Models.
3. Transfer: Seed a new population with FRESH BRAINS but PRE-TRAINED WORLD MODELS.
   This tests if knowing the physics helps learn a new path faster.
"""

import sys
import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import copy
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agi.agent import AGIAgent

class GauntletMaze(gym.Env):
    """A maze that can be inverted for transfer learning tests."""
    def __init__(self, layout_id=1):
        super().__init__()
        self.grid_size = 10
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        
        self.layout = np.zeros((10, 10))
        self.start_pos = np.array([1, 1])
        self.goal_pos = np.array([8, 8])
        
        if layout_id == 1:
            self.layout[4:9, 4] = 1 
            self.layout[4, 4:9] = 1
        else: # Inverted layout
            self.layout[1:6, 6] = 1
            self.layout[6, 1:6] = 1
            
        self.agent_pos = self.start_pos.copy()
        self.max_steps = 200
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
        
        # Fitness is negative distance, with a big bonus for success
        fitness = -dist
        if done: fitness += 1000.0
        
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), fitness, done, truncated, {}

    def _get_obs(self):
        return self.agent_pos.astype(np.float32) / self.grid_size

def evaluate_agent(agent, env):
    """Run one episode, return total fitness and if successful."""
    state, _ = env.reset()
    total_fitness = 0
    
    for step in range(env.max_steps):
        action = agent.act(state)
        next_state, fitness, done, truncated, _ = env.step(action)
        
        # Curiosity as a fitness bonus
        curiosity = agent.get_intrinsic_reward(state, action, next_state)
        total_fitness += fitness + (curiosity * 0.1) # Weight curiosity
        
        agent.memory.remember(state, action, fitness, next_state, done)
        state = next_state
        
        if done or truncated:
            break
            
    return total_fitness, done

def run_evolution_phase(population, env, generations, phase_name):
    print(f"\n>>> PHASE: {phase_name} ({generations} generations)")
    
    population_size = len(population)
    num_elites = 5
    best_fitness_history = []

    for gen in range(generations):
        # Evaluate all agents
        results = [evaluate_agent(agent, env) for agent in population]
        fitnesses = [r[0] for r in results]
        successes = [r[1] for r in results]
        
        sorted_indices = np.argsort(fitnesses)[::-1]
        elites = [population[i] for i in sorted_indices[:num_elites]]
        
        best_fitness = fitnesses[sorted_indices[0]]
        best_fitness_history.append(best_fitness)
        
        # Create next generation
        next_population = [copy.deepcopy(elite) for elite in elites]
        best_elite = elites[0]
        
        for _ in range(population_size - num_elites):
            child = copy.deepcopy(best_elite)
            child.mutate() # Use internal thermodynamic mutation
            next_population.append(child)
            
        population = next_population
        
        if gen % 20 == 0:
            success_rate = np.mean(successes)
            print(f"   Gen {gen} | Best Fitness: {best_fitness:.2f} | Success Rate: {success_rate:.1%}")
            
    return population, best_fitness_history

def run_gauntlet():
    # --- Phase 1: Maze A (Exploration) ---
    env_a = GauntletMaze(layout_id=1)
    initial_population = [AGIAgent(input_dim=2, action_dim=4, hidden_dim=32) for _ in range(20)]
    trained_population, history_a = run_evolution_phase(initial_population, env_a, 100, "1. Exploration (Maze A)")
    
    # --- Phase 2: Deep Sleep (Consolidation) ---
    print("\n>>> PHASE: 2. Deep Sleep (Consolidation)")
    results = [evaluate_agent(p, env_a) for p in trained_population]
    elites_from_a = [trained_population[i] for i in np.argsort([r[0] for r in results])[::-1][:2]]
    
    # Train the World Model of the best elite
    best_elite = elites_from_a[0]
    loss = best_elite.sleep(epochs=100)
    print(f"   Elite trained World Model. Final Loss: {loss:.4f}")
    
    # --- Phase 3: Maze B (Transfer) ---
    env_b = GauntletMaze(layout_id=2)
    
    # Population 1: FRESH Brains + PRE-TRAINED World Model
    # We clone the elite's world model into fresh agents
    pretrained_population = []
    for _ in range(20):
        agent = AGIAgent(input_dim=2, action_dim=4, hidden_dim=32)
        agent.world_model.load_state_dict(best_elite.world_model.state_dict()) # Transplant
        pretrained_population.append(agent)
    
    # Population 2: FRESH Brains + FRESH World Model (Control)
    fresh_population = [AGIAgent(input_dim=2, action_dim=4, hidden_dim=32) for _ in range(20)]

    _, history_b_pretrained = run_evolution_phase(pretrained_population, env_b, 100, "3a. Transfer Test (Pre-Trained WM)")
    _, history_b_fresh = run_evolution_phase(fresh_population, env_b, 100, "3b. Control Test (Fresh WM)")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Smooth curves
    def smooth(y, box_pts=10):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    plt.plot(smooth(history_b_pretrained), label='Pre-Trained World Model', color='blue')
    plt.plot(smooth(history_b_fresh), label='Fresh World Model', color='red', linestyle='--')

    plt.title("AGI Gauntlet: Transfer Learning via World Models")
    plt.xlabel("Generation (in Maze B)")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '../logs/agi_gauntlet_good.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    run_gauntlet()
