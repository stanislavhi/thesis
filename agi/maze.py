import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GauntletMaze(gym.Env):
    """
    A 10x10 grid maze with swappable wall layouts for transfer learning tests.

    Layout 1: L-shaped wall in bottom-right quadrant.
    Layout 2: L-shaped wall in top-right / bottom-left (inverted).
    """
    def __init__(self, layout_id=1, max_steps=200):
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
        else:
            self.layout[1:6, 6] = 1
            self.layout[6, 1:6] = 1

        self.agent_pos = self.start_pos.copy()
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos.copy()
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        move = np.array(moves.get(action, [0, 0]))

        new_pos = np.clip(self.agent_pos + move, 0, self.grid_size - 1)

        if self.layout[new_pos[0], new_pos[1]] == 1:
            new_pos = self.agent_pos

        self.agent_pos = new_pos
        self.current_step += 1

        dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        done = np.array_equal(self.agent_pos, self.goal_pos)

        fitness = -dist
        if done:
            fitness += 1000.0

        truncated = self.current_step >= self.max_steps

        return self._get_obs(), fitness, done, truncated, {}

    def _get_obs(self):
        return self.agent_pos.astype(np.float32) / self.grid_size
