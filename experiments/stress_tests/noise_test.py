import sys
import os
import torch
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from agents.rl_policy import EvolvingPolicy, RLChaosInjector
from core.chaos import LorenzGenerator

class NoisyEnvWrapper(gym.Wrapper):
    """
    Injects noise into the observation to simulate sensor degradation.
    """
    def __init__(self, env, noise_level=0.0):
        super().__init__(env)
        self.noise_level = noise_level
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._add_noise(obs), info
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._add_noise(obs), reward, done, truncated, info
        
    def _add_noise(self, obs):
        noise = np.random.normal(0, self.noise_level, size=obs.shape)
        return obs + noise

def run_noise_trial(noise_level, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    base_env = gym.make("CartPole-v1")
    env = NoisyEnvWrapper(base_env, noise_level)
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    agent = EvolvingPolicy(input_dim, 4, output_dim)
    mutator = RLChaosInjector(LorenzGenerator())
    optimizer = optim.Adam(agent.parameters(), lr=0.01)
    
    scores = deque(maxlen=50)
    history = []
    
    for episode in range(300):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs = agent(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            next_state, reward, done, truncated, _ = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            state = next_state
            
            if done or truncated: break
            
        total_reward = sum(rewards)
        scores.append(total_reward)
        avg_score = np.mean(scores)
        history.append(total_reward)
        
        # REINFORCE
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
            
        # Mutate if stagnant
        if episode > 20 and avg_score < 100 and np.std(scores) < 5.0:
            agent = mutator.mutate(agent)
            optimizer = optim.Adam(agent.parameters(), lr=0.01)
            scores.clear()
            
    env.close()
    return history

def main():
    print("--- STRESS TEST: EXTREME NOISE ---")
    noise_levels = [0.0, 0.1, 0.5]
    results = {}
    
    for noise in noise_levels:
        print(f"Running with Noise Level: {noise}...")
        # Run 3 seeds
        trials = []
        for seed in [42, 101, 999]:
            trials.append(run_noise_trial(noise, seed))
        
        # Average
        max_len = max(len(t) for t in trials)
        padded = np.array([t + [t[-1]]*(max_len-len(t)) for t in trials])
        results[noise] = np.mean(padded, axis=0)
        
    # Plot
    plt.figure(figsize=(10, 6))
    for noise, avg_hist in results.items():
        plt.plot(avg_hist, label=f'Noise {noise}')
        
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Thermodynamic Resilience to Sensory Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/stress_test_noise.png'))
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
