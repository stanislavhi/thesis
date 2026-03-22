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
from agents.thermodynamic.thermo_agent import ThermodynamicAgent
from agents.thermodynamic.thermo_injector import ThermodynamicInjector


def inflict_brain_damage(policy, damage_ratio=0.5):
    """
    Randomly zeroes out a fraction of all weights in the policy network.
    This simulates catastrophic neural damage.
    Returns the number of parameters zeroed.
    """
    total_zeroed = 0
    with torch.no_grad():
        for param in policy.parameters():
            mask = torch.rand_like(param) > damage_ratio
            param.mul_(mask.float())
            total_zeroed += (~mask).sum().item()
    return int(total_zeroed)


def run_brain_damage_trial(seed, use_chaos=True, damage_episode=150, max_episodes=400):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1")

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy = ThermodynamicAgent(input_dim, 16, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    chaos_gen = LorenzGenerator()
    injector = ThermodynamicInjector(chaos_gen)

    scores = deque(maxlen=20)
    history = []
    last_mutation_ep = -100
    mutation_cooldown = 30

    for episode in range(max_episodes):
        # --- BRAIN DAMAGE at the specified episode ---
        if episode == damage_episode:
            n_zeroed = inflict_brain_damage(policy, damage_ratio=0.5)
            total_params = sum(p.numel() for p in policy.parameters())
            print(f"   [Seed {seed}] BRAIN DAMAGE at ep {episode}: "
                  f"zeroed {n_zeroed}/{total_params} params ({100*n_zeroed/total_params:.0f}%)")
            optimizer = optim.Adam(policy.parameters(), lr=0.01)
            scores.clear()
            last_mutation_ep = episode

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

        # Thermodynamic chaos injection — uses Operator Selection Rule
        if use_chaos and episode > 20 and len(scores) == scores.maxlen:
            eps_since_mutation = episode - last_mutation_ep
            if eps_since_mutation < mutation_cooldown:
                continue

            # Let the agent self-diagnose
            status = policy.get_thermodynamic_status()

            # Check if scores are trending upward (recovering, not stagnated)
            scores_list = list(scores)
            first_half = np.mean(scores_list[:len(scores_list)//2])
            second_half = np.mean(scores_list[len(scores_list)//2:])
            is_recovering = second_half > first_half + 5.0

            # Only inject if truly frozen AND not recovering
            if status == 'frozen' and avg_score < 100 and np.std(scores) < 10.0 and not is_recovering:
                policy = injector.mutate(policy, status=status)
                optimizer = optim.Adam(policy.parameters(), lr=0.01)
                scores.clear()
                last_mutation_ep = episode

    env.close()
    return history


def main():
    print("--- STRESS TEST: BRAIN DAMAGE RESILIENCE ---")
    print("Training 150 eps -> Zero 50% of weights -> Continue 250 eps")
    print("Comparing CHAOS (Operator Selection Rule) vs STATIC (fixed architecture)\n")

    seeds = [42, 101, 999, 123, 777]
    chaos_results = []
    static_results = []

    print(">>> Running CHAOS trials (ThermodynamicAgent + Operator Selection)...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        res = run_brain_damage_trial(seed, use_chaos=True)
        chaos_results.append(res)

    print("\n>>> Running STATIC trials...")
    for seed in seeds:
        print(f"   Seed {seed}...", flush=True)
        res = run_brain_damage_trial(seed, use_chaos=False)
        static_results.append(res)

    # Statistical Analysis
    chaos_matrix = np.array(chaos_results)
    static_matrix = np.array(static_results)

    chaos_mean = np.mean(chaos_matrix, axis=0)
    chaos_std = np.std(chaos_matrix, axis=0)
    static_mean = np.mean(static_matrix, axis=0)
    static_std = np.std(static_matrix, axis=0)

    # Recovery metrics (post-damage)
    post_damage_chaos = chaos_matrix[:, 150:]
    post_damage_static = static_matrix[:, 150:]

    chaos_recovery_avg = np.mean(post_damage_chaos)
    static_recovery_avg = np.mean(post_damage_static)

    # Best post-damage average (per trial, take last 50 episodes)
    chaos_final = np.mean(chaos_matrix[:, -50:])
    static_final = np.mean(static_matrix[:, -50:])

    print("\n--- RESULTS ---")
    print(f"Post-Damage Avg Score:   Chaos={chaos_recovery_avg:.1f}  |  Static={static_recovery_avg:.1f}")
    print(f"Final 50-ep Avg Score:   Chaos={chaos_final:.1f}  |  Static={static_final:.1f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    x = np.arange(len(chaos_mean))

    plt.plot(x, chaos_mean, color='blue', linewidth=2, label='Chaos (Operator Selection)')
    plt.fill_between(x, chaos_mean - chaos_std, chaos_mean + chaos_std, color='blue', alpha=0.15)

    plt.plot(x, static_mean, color='red', linewidth=2, label='Static (Fixed)')
    plt.fill_between(x, static_mean - static_std, static_mean + static_std, color='red', alpha=0.15)

    plt.axvline(x=150, color='black', linestyle='--', linewidth=2, label='Brain Damage (50% zeroed)')

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Neuroplasticity: Recovery from Brain Damage (n={len(seeds)} seeds)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.annotate(f'Chaos recovery: {chaos_final:.0f}', xy=(370, chaos_final),
                 fontsize=10, color='blue', fontweight='bold')
    plt.annotate(f'Static recovery: {static_final:.0f}', xy=(370, static_final),
                 fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/stress_test_brain_damage.png'))
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")


if __name__ == "__main__":
    main()
