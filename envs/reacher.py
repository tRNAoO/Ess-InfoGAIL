import copy
import gym
import numpy as np
from gym import spaces
from gym.envs.mujoco.reacher_v4 import ReacherEnv


class MultimodalReacher(ReacherEnv):
    """A multi-modal variant of the Reacher environment.

    https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher_v4.py

    The goal information in _get_obs() are removed. The first dof is initialized between [-pi, pi] and the
    second dof is initialized between [-pi/2, pi/2]. The goal position is generated according to mode_idx.

    reward_eval is the mode specific reward only for evaluation.
    reward_train is the reward shared by all modes.
    """

    def __init__(self, num_modes=6, goal_radius=0.15, train_expert=False):
        super().__init__()
        self.num_modes = num_modes
        self.goal_radius = goal_radius
        self.train_expert = train_expert
        self._max_episode_steps = 50
        self.t = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)

    def reset(
        self,
        *,
        seed=None,
        return_info=False,
        options=None,
        mode_idx=0
    ):
        super().reset(seed=seed)

        self._reset_simulation()
        self.t = 0
        ob = self.reset_model(mode_idx)
        # self.renderer.reset()
        # self.renderer.render_step()
        if not return_info:
            return ob
        else:
            return ob, {}

    def reset_model(self, mode_idx=0):
        qpos = (
                self.np_random.uniform(low=-np.pi / 2, high=np.pi / 2, size=self.model.nq)
                + self.init_qpos
        )
        qpos[0] = self.np_random.uniform(low=-np.pi, high=np.pi, size=1) + self.init_qpos[0]
        qpos[1] = self.np_random.uniform(low=-np.pi / 2, high=np.pi / 2, size=1) + self.init_qpos[0]
        while True:
            theta = mode_idx * 2 * np.pi / self.num_modes
            coord_x = np.cos(theta) * self.goal_radius
            coord_y = np.sin(theta) * self.goal_radius
            self.goal = np.array([coord_x, coord_y])

            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, a):
        self.t += 1
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        if self.train_expert:
            reward = reward_dist + reward_ctrl
        else:
            reward = reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        # self.renderer.render_step()
        truncated = False
        if self.t == self._max_episode_steps:
            truncated = True
        return (
            ob,
            reward,
            False,
            truncated,
            dict(reward_eval=reward_dist, reward_train=reward_ctrl),
        )

    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.data.qvel.flat[:2],
            ]
        )
