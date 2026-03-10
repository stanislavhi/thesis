import sys
import os
import torch
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agents.thermodynamic.thermo_agent import ThermodynamicAgent
from agents.thermodynamic.thermo_injector import ThermodynamicInjector
from core.chaos import LorenzGenerator

def run_thermo_trial(seed, use_thermo_injection=True, max_episodes=300):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    # Initialize Thermodynamic Agent
    agent = ThermodynamicAgent(input_dim, 16, output_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.01)
    
    # Initialize Thermodynamic Injector
    chaos_gen = LorenzGenerator()
    injector = ThermodynamicInjector(chaos_gen, base_mutation_rate=0.1)
    
    scores = deque(maxlen=20)
    history = []
    sigma_history = []
    
    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed + episode)
        log_probs = []
        rewards = []
        
        # --- Episode Loop ---
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = agent(state_tensor) # Forward pass calculates sigma internally
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            next_state, reward, done, truncated, _ = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            state = next_state
            
            if done or truncated:
                break
                
        total_reward = sum(rewards)
        scores.append(total_reward)
        avg_score = np.mean(scores)
        history.append(total_reward)
        sigma_history.append(agent.current_sigma)
        
        # --- REINFORCE Update ---
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        if policy_loss:
            torch.stack(policy_loss).sum().backward()
            optimizer.step()
            
        # --- Thermodynamic Injection ---
        # Only inject if enabled AND agent is struggling (low score) OR frozen (low sigma)
        if use_thermo_injection and episode > 20:
            status = agent.get_thermodynamic_status()
            
            # If agent is frozen (low sigma), inject chaos regardless of score!
            # This is the key difference: standard RL waits for low reward.
            # Thermo RL acts on internal physics.
            if status == 'frozen' or (avg_score < 50 and np.std(scores) < 5.0):
                agent = injector.mutate(agent)
                # Reset optimizer after mutation
                optimizer = optim.Adam(agent.parameters(), lr=0.01)
                scores.clear() # Clear history to give new mutation a chance
                
    env.close()
    return history, sigma_history

def main():
    print("--- THERMODYNAMIC AGENT EXPERIMENT ---")
    print("Comparing Thermo-Driven Mutation vs Standard RL")
    
    seeds = [42, 101, 999]
    thermo_results = []
    standard_results = []
    thermo_sigmas = []
    
    print("\n>>> Running THERMO Trials (Internal Physics Driven)...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        res, sigmas = run_thermo_trial(seed, use_thermo_injection=True)
        thermo_results.append(res)
        thermo_sigmas.append(sigmas)

    print("\n>>> Running STANDARD Trials (Reward Driven Only)...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        # Standard RL: disable the specialized injector logic
        res, _ = run_thermo_trial(seed, use_thermo_injection=False)
        standard_results.append(res)

    # Analysis
    thermo_mean = np.mean(thermo_results, axis=0)
    standard_mean = np.mean(standard_results, axis=0)
    avg_sigma = np.mean(thermo_sigmas, axis=0)
    
    print("\n--- RESULTS ---")
    print(f"Thermo Avg Final Score: {thermo_mean[-1]:.2f}")
    print(f"Standard Avg Final Score: {standard_mean[-1]:.2f}")
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot Scores
    ax1.plot(thermo_mean, color='blue', label='Thermodynamic Agent (Physics-Driven)')
    ax1.plot(standard_mean, color='red', linestyle='--', label='Standard RL (Reward-Driven)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')
    
    # Plot Sigma (Entropy Production) on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(avg_sigma, color='green', alpha=0.3, label='Avg Internal Sigma (Entropy Production)')
    ax2.set_ylabel('Sigma (Internal Work)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    # ax2.legend(loc='upper right')
    
    plt.title('Thermodynamic Agent: Performance vs Internal Entropy Production')
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/thermo_agent_experiment.png'))
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    main()
