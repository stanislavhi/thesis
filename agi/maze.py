import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GauntletMaze(gym.Env):
    """
    A 15x15 grid maze with multiple rooms and swappable wall layouts.

    Observation (8D):
      [0-1] Normalized agent position (row, col)
      [2-3] Direction to goal (unit vector)
      [4-7] Wall proximity in 4 directions (0=wall adjacent, 1=clear path)

    The richer observation space gives the world model non-trivial dynamics
    to learn — wall proximity changes discontinuously at walls, making
    prediction error a meaningful curiosity signal for novel layouts.

    Layout 1: Three-room layout (horizontal dividers with gaps).
    Layout 2: Three-room layout (vertical dividers with gaps).
    Same number of rooms, different topology — tests structural transfer.
    """
    OBS_DIM = 8

    def __init__(self, layout_id=1, max_steps=300):
        super().__init__()
        self.grid_size = 15
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.OBS_DIM,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.layout = np.zeros((self.grid_size, self.grid_size))
        self.start_pos = np.array([1, 1])
        self.goal_pos = np.array([13, 13])

        if layout_id == 1:
            # Horizontal walls creating 3 rooms with narrow passages
            self.layout[5, :] = 1
            self.layout[5, 3] = 0   # gap left
            self.layout[5, 11] = 0  # gap right
            self.layout[10, :] = 1
            self.layout[10, 7] = 0  # gap center
            self.layout[10, 13] = 0 # gap right
            # Vertical obstacle in room 2
            self.layout[6:10, 6] = 1
            self.layout[8, 6] = 0   # gap
        else:
            # Vertical walls creating 3 rooms with narrow passages
            self.layout[:, 5] = 1
            self.layout[3, 5] = 0   # gap top
            self.layout[11, 5] = 0  # gap bottom
            self.layout[:, 10] = 1
            self.layout[7, 10] = 0  # gap center
            self.layout[13, 10] = 0 # gap bottom
            # Horizontal obstacle in room 2
            self.layout[7, 6:10] = 1
            self.layout[7, 8] = 0   # gap

        # Ensure start and goal are clear
        self.layout[self.start_pos[0], self.start_pos[1]] = 0
        self.layout[self.goal_pos[0], self.goal_pos[1]] = 0

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

        # Shaped reward: step penalty + distance + large bonus for goal
        fitness = -dist - 0.5  # step penalty discourages stalling
        if done:
            fitness += 1000.0

        truncated = self.current_step >= self.max_steps

        return self._get_obs(), fitness, done, truncated, {}

    def _wall_proximity(self, pos, direction, max_dist=5):
        """Distance to nearest wall or boundary in a given direction, normalized to [0,1]."""
        r, c = pos
        dr, dc = direction
        for d in range(1, max_dist + 1):
            nr, nc = r + dr * d, c + dc * d
            if nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size:
                return (d - 1) / max_dist
            if self.layout[nr, nc] == 1:
                return (d - 1) / max_dist
        return 1.0

    def _get_obs(self):
        pos_norm = self.agent_pos.astype(np.float32) / self.grid_size

        # Direction to goal (unit vector)
        diff = self.goal_pos - self.agent_pos
        dist = np.linalg.norm(diff)
        goal_dir = (diff / dist).astype(np.float32) if dist > 0 else np.zeros(2, dtype=np.float32)

        # Wall proximity in 4 cardinal directions
        prox = np.array([
            self._wall_proximity(self.agent_pos, [-1, 0]),  # up
            self._wall_proximity(self.agent_pos, [1, 0]),   # down
            self._wall_proximity(self.agent_pos, [0, -1]),  # left
            self._wall_proximity(self.agent_pos, [0, 1]),   # right
        ], dtype=np.float32)

        return np.concatenate([pos_norm, goal_dir, prox])
