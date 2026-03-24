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

from core.chaos import LorenzGenerator
from agents.rl_policy import EvolvingPolicy, RLChaosInjector
from experiments.utils import reinforce_update, EnvironmentShockWrapper


def run_transfer_trial(seed, use_chaos=True, shock_episode=150, max_episodes=400,
                       shock_mode="swap_actions"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    base_env = gym.make("CartPole-v1")
    env = EnvironmentShockWrapper(base_env, shock_mode=shock_mode)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy = EvolvingPolicy(input_dim, 16, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    chaos_gen = LorenzGenerator()
    injector = RLChaosInjector(chaos_gen)

    scores = deque(maxlen=20)
    history = []

    for episode in range(max_episodes):
        # --- SHOCK at the specified episode ---
        if episode == shock_episode and not env.shock_active:
            env.activate_shock()
            print(f"   [Seed {seed}] ENVIRONMENT SHOCK at ep {episode}: {shock_mode}")
            scores.clear()

        state, _ = env.reset()
        log_probs = []
        rewards = []

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
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

        # REINFORCE
        reinforce_update(log_probs, rewards, optimizer)

        # Chaos Injection — only if enabled
        if use_chaos and episode > 20 and avg_score < 50:
            if np.std(scores) < 10.0:
                policy = injector.mutate(policy)
                optimizer = optim.Adam(policy.parameters(), lr=0.01)
                scores.clear()

    env.close()
    return history


def main():
    shock_mode = "swap_actions"

    print("--- STRESS TEST: TRANSFER LEARNING SHOCK ---")
    print(f"Training 150 eps -> Environment Shock ({shock_mode}) -> Continue 250 eps")
    print("Comparing CHAOS (adaptive topology) vs STATIC (fixed architecture)\n")

    seeds = [42, 101, 999, 123, 777]
    chaos_results = []
    static_results = []

    print(">>> Running CHAOS trials...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        res = run_transfer_trial(seed, use_chaos=True, shock_mode=shock_mode)
        chaos_results.append(res)

    print("\n>>> Running STATIC trials...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        res = run_transfer_trial(seed, use_chaos=False, shock_mode=shock_mode)
        static_results.append(res)

    # Analysis
    chaos_matrix = np.array(chaos_results)
    static_matrix = np.array(static_results)

    chaos_mean = np.mean(chaos_matrix, axis=0)
    chaos_std = np.std(chaos_matrix, axis=0)
    static_mean = np.mean(static_matrix, axis=0)
    static_std = np.std(static_matrix, axis=0)

    # Recovery metrics (last 50 episodes)
    chaos_final = np.mean(chaos_matrix[:, -50:])
    static_final = np.mean(static_matrix[:, -50:])

    print("\n--- RESULTS ---")
    print(f"Final 50-ep Avg Score:   Chaos={chaos_final:.1f}  |  Static={static_final:.1f}")

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(chaos_mean))

    plt.plot(x, chaos_mean, color='blue', linewidth=2, label='Chaos (Adaptive)')
    plt.fill_between(x, chaos_mean - chaos_std, chaos_mean + chaos_std, color='blue', alpha=0.15)

    plt.plot(x, static_mean, color='red', linewidth=2, label='Static (Fixed)')
    plt.fill_between(x, static_mean - static_std, static_mean + static_std, color='red', alpha=0.15)

    plt.axvline(x=150, color='black', linestyle='--', linewidth=2,
                label=f'Environment Shock ({shock_mode})')

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Transfer Shock: Can Chaos Recover? ({shock_mode}, n={len(seeds)} seeds)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.annotate(f'Chaos: {chaos_final:.0f}', xy=(370, chaos_final),
                 fontsize=10, color='blue', fontweight='bold')
    plt.annotate(f'Static: {static_final:.0f}', xy=(370, static_final),
                 fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '../../logs/stress_test_transfer_shock.png'))
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")


if __name__ == "__main__":
    main()
