#!/usr/bin/env python3
"""
Experiment: The AGI Gauntlet.

Goal: Test the full Thermodynamic AGI architecture (Memory, World Model, Hierarchy, Curiosity).

Phases:
1. Exploration: Solve Maze A using Curiosity.
2. Consolidation: Sleep and train the World Model on memories.
3. Transfer: Solve Maze B (inverted). Does the World Model help?
"""

import sys
import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import copy

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
            # Standard Trap (Wall at 4,4 blocking diagonal)
            self.layout[4:9, 4] = 1 
            self.layout[4, 4:9] = 1
        else:
            # Inverted Trap (Wall at 6,6 blocking from other side)
            self.layout[1:6, 6] = 1
            self.layout[6, 1:6] = 1
            
        self.agent_pos = self.start_pos.copy()
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
        
        done = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = self.current_step >= self.max_steps
        
        # Sparse reward for the environment (Curiosity will be added by agent)
        reward = 100.0 if done else -0.1
        
        return self._get_obs(), reward, done, truncated, {}

    def _get_obs(self):
        return self.agent_pos.astype(np.float32) / self.grid_size

def run_phase(agent, env, episodes, phase_name, train_wm=False):
    print(f"\n>>> PHASE: {phase_name} ({episodes} episodes)")
    success_count = 0
    rewards = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, env_reward, done, truncated, _ = env.step(action)
            
            # 1. Curiosity Reward (Intrinsic)
            # If the world model is surprised, this is a good state to explore
            curiosity = agent.get_intrinsic_reward(state, action, next_state)
            
            # Total Reward = External + Intrinsic
            # We weight curiosity high initially to encourage exploration
            combined_reward = env_reward + (curiosity * 5.0)
            
            # 2. Memory
            agent.memory.remember(state, action, combined_reward, next_state, done)
            
            # 3. Update Brain (REINFORCE-style for simplicity here, or ES)
            # For this gauntlet, we'll use a simple ES mutation if stuck
            if agent.get_thermodynamic_status() == 'frozen':
                agent.mutate()
            
            state = next_state
            total_reward += env_reward # Track external performance
            
            if done or truncated:
                break
        
        if done: success_count += 1
        rewards.append(total_reward)
        
        # 4. Sleep (Consolidation)
        # Train World Model every 10 episodes
        if train_wm and ep % 10 == 0:
            loss = agent.sleep(epochs=5)
            # print(f"   [Sleep] WM Loss: {loss:.4f}")
            
    print(f"   Success Rate: {success_count/episodes:.1%}")
    return rewards

def run_gauntlet():
    # 1. Initialize
    agent = AGIAgent(input_dim=2, action_dim=4, hidden_dim=32)
    
    # 2. Phase 1: Maze A (Exploration & Learning Physics)
    env_a = GauntletMaze(layout_id=1)
    rewards_a = run_phase(agent, env_a, episodes=200, phase_name="1. Exploration (Maze A)", train_wm=True)
    
    # 3. Phase 2: Deep Sleep (Consolidation)
    print("\n>>> PHASE: 2. Deep Sleep (Consolidation)")
    print("   Agent is dreaming and refining its World Model...")
    wm_loss = agent.sleep(epochs=100)
    print(f"   Final World Model Loss: {wm_loss:.4f}")
    
    # 4. Phase 3: Maze B (Transfer)
    # We compare the pre-trained agent vs a fresh agent
    env_b = GauntletMaze(layout_id=2)
    
    # Clone for fair comparison (same architecture, fresh weights)
    fresh_agent = AGIAgent(input_dim=2, action_dim=4, hidden_dim=32)
    
    print("\n>>> PHASE: 3. Transfer Test (Maze B)")
    print("   Comparing Pre-Trained Agent vs Fresh Agent...")
    
    rewards_b_pretrained = run_phase(agent, env_b, episodes=200, phase_name="Pre-Trained Agent", train_wm=True)
    rewards_b_fresh = run_phase(fresh_agent, env_b, episodes=200, phase_name="Fresh Agent", train_wm=True)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Smooth curves
    def smooth(y, box_pts=20):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    plt.plot(smooth(rewards_b_pretrained), label='Pre-Trained (With World Model)', color='blue')
    plt.plot(smooth(rewards_b_fresh), label='Fresh (No World Model)', color='red', linestyle='--')
    
    plt.title("AGI Gauntlet: Transfer Learning via World Models")
    plt.xlabel("Episode (Maze B)")
    plt.ylabel("External Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/agi_gauntlet.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    run_gauntlet()
