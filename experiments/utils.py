"""Shared utilities for experiments — eliminates duplication across test scripts."""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym


# ---------------------------------------------------------------------------
# REINFORCE update step (used by ~10 experiment files)
# ---------------------------------------------------------------------------

def reinforce_update(log_probs, rewards, optimizer, gamma=0.99, clip_grad=None):
    """
    Standard REINFORCE policy gradient step with baseline subtraction.

    Returns grad_norm (float) for logging. Returns 0.0 if no update was made.
    """
    if not log_probs:
        return 0.0

    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards)
    if len(discounted_rewards) > 1:
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    policy_loss = []
    for log_prob, R in zip(log_probs, discounted_rewards):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    loss_tensor = torch.stack(policy_loss).sum()

    if torch.isnan(loss_tensor):
        return 0.0

    loss_tensor.backward()

    if clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group['params']],
            clip_grad
        )

    # Compute grad norm before step
    grad_norm = 0.0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    optimizer.step()
    return grad_norm


# ---------------------------------------------------------------------------
# Brain damage / weight destruction
# ---------------------------------------------------------------------------

def inflict_brain_damage(policy, damage_ratio=0.5):
    """
    Randomly zeroes out a fraction of all weights in the policy network.
    Returns the number of parameters zeroed.
    """
    total_zeroed = 0
    with torch.no_grad():
        for param in policy.parameters():
            mask = torch.rand_like(param) > damage_ratio
            param.mul_(mask.float())
            total_zeroed += (~mask).sum().item()
    return int(total_zeroed)


# ---------------------------------------------------------------------------
# Smoothing for plots
# ---------------------------------------------------------------------------

def smooth(y, box_pts=20):
    """Moving average smoothing. Returns array shortened by (box_pts - 1)."""
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='valid')


def smooth_with_x(y, box_pts=20):
    """Returns (x_aligned, y_smoothed) so x-axis matches original episode numbers."""
    y_smooth = smooth(y, box_pts)
    x = np.arange(box_pts - 1, box_pts - 1 + len(y_smooth))
    return x, y_smooth


# ---------------------------------------------------------------------------
# Environment wrappers
# ---------------------------------------------------------------------------

class InvertibleEnv(gym.Wrapper):
    """Inverts action mapping. For 2-action envs (CartPole): swaps 0<->1.
    For 3-action envs (Acrobot): swaps 0<->2, keeps 1."""

    def __init__(self, env):
        super().__init__(env)
        self.inverted = False
        n = env.action_space.n
        if n == 2:
            self._map = {0: 1, 1: 0}
        elif n == 3:
            self._map = {0: 2, 1: 1, 2: 0}
        else:
            self._map = {i: (n - 1 - i) for i in range(n)}

    def step(self, action):
        if self.inverted:
            action = self._map.get(action, action)
        return self.env.step(action)

    def invert(self):
        self.inverted = not self.inverted


class EnvironmentShockWrapper(gym.Wrapper):
    """Multi-mode environment shock: swap_actions, invert_rewards, noisy_obs."""

    def __init__(self, env, shock_mode="swap_actions"):
        super().__init__(env)
        self.shock_active = False
        self.shock_mode = shock_mode
        n = env.action_space.n
        self._action_map = {i: (n - 1 - i) for i in range(n)}

    def activate_shock(self):
        self.shock_active = True

    def step(self, action):
        if self.shock_active:
            if self.shock_mode == "swap_actions":
                action = self._action_map.get(action, action)
            elif self.shock_mode == "invert_rewards":
                obs, reward, done, truncated, info = self.env.step(action)
                return obs, -reward, done, truncated, info
            elif self.shock_mode == "noisy_obs":
                obs, reward, done, truncated, info = self.env.step(action)
                noise = np.random.normal(0, 0.5, size=obs.shape)
                return obs + noise, reward, done, truncated, info

        return self.env.step(action)


# ---------------------------------------------------------------------------
# Ablation injector (shared by ablation_robustness_test, ablation_lunar_lander)
# ---------------------------------------------------------------------------

class AblationInjector:
    """Mutation operator for ablation studies: additive noise or targeted dropout."""

    def __init__(self, strategy="additive", base_rate=0.05):
        self.strategy = strategy
        self.base_rate = base_rate

    def mutate(self, agent):
        status = getattr(agent, 'get_thermodynamic_status', lambda: 'frozen')()

        if status == 'frozen':
            if self.strategy == "additive":
                magnitude = self.base_rate * 5.0
                with torch.no_grad():
                    for param in agent.parameters():
                        noise = torch.randn_like(param) * magnitude
                        param.add_(noise)

            elif self.strategy == "dropout":
                with torch.no_grad():
                    weights = agent.layer1.weight
                    variances = torch.var(weights, dim=1)
                    k = max(1, int(weights.shape[0] * 0.2))
                    _, indices = torch.topk(variances, k, largest=False)
                    weights[indices] = 0.0
                    if agent.layer1.bias is not None:
                        agent.layer1.bias[indices] = 0.0

        return agent
