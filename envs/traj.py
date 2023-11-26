import numpy as np

import gym
from gym import spaces
from typing import Optional


class Traj(gym.Env):
    """The 2D-Trajectory environment.
    An agent can move freely within a plane.

    reward_eval is the mode specific reward only for evaluation.
    reward_train is the reward shared by all modes.
    """
    def __init__(self, num_modes=4, scale=10.):
        self.observation_space = spaces.Box(
            low=-20/scale, high=20/scale, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float32
        )
        self.scale = scale
        self.step_size = 1.0 / scale
        self.state = np.array([0., 0., 0.])
        self.theta = 0.0
        self.theta_state = 0.0
        self.action_scale = np.pi / 4
        self._max_episode_steps = 60
        self.center_set = np.array([[0, 5], [0, 10], [0, -5], [0, -10]]) / scale
        self.radius_set = np.array([5, 10, 5, 10]) / scale
        self.num_modes = num_modes
        self.mode_idx = None
        self.t = 0

    def step(self, action: np.ndarray):
        self.t += 1
        terminated, truncated = False, False
        action = action[0] * self.action_scale
        self.theta = self.theta + action
        n_round = np.abs(self.theta) // (np.pi * 2)

        if self.theta < 0:
            self.theta_state = self.theta + n_round * np.pi * 2 + np.pi * 2
        else:
            self.theta_state = self.theta - n_round * np.pi * 2

        delta_x = self.step_size * np.cos(self.theta)
        delta_y = self.step_size * np.sin(self.theta)
        self.state[0] = self.state[0] + delta_x
        self.state[1] = self.state[1] + delta_y
        self.state[2] = self.theta_state

        center_dis = self.distance(self.state[:2], self.center_set[self.mode_idx])
        origin_dis = self.distance(self.state[:2], [0, 0])

        obs_l = np.min(self.observation_space.low)
        obs_h = np.max(self.observation_space.high)
        margin = 1.5

        reward_eval = -((center_dis - self.radius_set[self.mode_idx])**2)**0.5
        reward_train = 0.0

        if self.state[0] < obs_l * margin or self.state[0] > obs_h * margin or \
            self.state[1] < obs_l * margin or self.state[1] > obs_h * margin:
            reward_train = -origin_dis

        if self.t == self._max_episode_steps:
            truncated = True

        reward = reward_train

        return self.state, reward, terminated, truncated, dict(reward_eval=reward_eval, reward_train=reward_train)

    def reset(self,
              mode_idx=0,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None, ):
        self.t = 0
        self.state = np.array([0., 0., 0.])
        self.theta = 0.0
        self.theta_state = 0.0
        self.mode_idx = mode_idx

        return self.state

    @staticmethod
    def distance(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx**2 + dy**2)**0.5
