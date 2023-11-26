import copy
import gym
import numpy as np
from gym import spaces
from gym.envs.mujoco.walker2d_v4 import Walker2dEnv


class MultimodalWalker(Walker2dEnv):
    """A multi-modal variant of the Walker-2D environment.

    https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d_v4.py

    reward_eval is the mode specific reward only for evaluation.
    reward_train is the reward shared by all modes.
    """

    def __init__(self, num_modes=3, train_expert=False):
        super().__init__()
        self.num_modes = num_modes
        self.train_expert = train_expert
        self._max_episode_steps = 1000
        self.t = 0
        self.mode_idx = None
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)

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
        self.mode_idx = mode_idx
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def step(self, action):
        self.t += 1
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)
        healthy_reward = self.healthy_reward

        forward_reward = 0.0
        abs_vel_reward = 0.0  # Prevent standing still
        if self.mode_idx == 0:
            forward_reward = self._forward_reward_weight * x_velocity
            abs_vel_reward = self._forward_reward_weight * np.fabs(x_velocity)
        elif self.mode_idx == 1:
            forward_reward = self._forward_reward_weight * -x_velocity
            abs_vel_reward = self._forward_reward_weight * np.fabs(x_velocity)
        elif self.mode_idx == 2:
            forward_reward = self._forward_reward_weight * -np.fabs(x_velocity)

        observation = self._get_obs()

        truncated = False
        if self.t == self._max_episode_steps:
            truncated = True

        terminated = self.terminated
        if terminated:
            healthy_reward = -100.0

        if self.train_expert:
            reward = forward_reward + healthy_reward - ctrl_cost
        else:
            reward = abs_vel_reward*0.6 + healthy_reward*0.3

        info = dict(reward_eval=forward_reward, reward_train=abs_vel_reward*0.6 + healthy_reward*0.3)

        self.renderer.render_step()
        return observation, reward, terminated, truncated, info
