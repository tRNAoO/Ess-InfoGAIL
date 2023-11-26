import copy
import gym
import numpy as np
from gym import spaces
from gym.envs.mujoco.pusher_v4 import PusherEnv


class MultimodalPusher(PusherEnv):
    """A multi-modal variant of the Pusher environment.

    https://github.com/openai/gym/blob/master/gym/envs/mujoco/pusher_v4.py

    The goal information in _get_obs() is removed. The goal position is generated according to mode_idx.

    reward_eval is the mode specific reward only for evaluation.
    reward_train is the reward shared by all modes.
    """

    def __init__(self, num_modes=6, train_expert=False):
        super().__init__()
        self.num_modes = num_modes
        self.goals = [[0, 0], [0, -0.43], [0, -0.86], [0.2, 0], [0.2, -0.43], [0.2, -0.86]]
        self.train_expert = train_expert
        self._max_episode_steps = 100
        self.t = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23-3,), dtype=np.float64)

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
        qpos = self.init_qpos
        self.goal_pos = np.asarray(self.goals[mode_idx])
        self.cylinder_pos = np.array([0, -0.2])
        qpos[0] = np.pi / 2.7
        qpos[1] = np.pi / 2.5
        qpos[3] = -np.pi / 2.0
        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, a):
        self.t += 1
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")
        vec_3 = self.get_body_com("tips_arm") - self.get_body_com("goal")

        reward_near = -np.linalg.norm(vec_1)
        reward_dist = -np.linalg.norm(vec_2)
        reward_reach = -np.linalg.norm(vec_3)
        reward_ctrl = -np.square(a).sum()

        if self.train_expert:
            reward = reward_near + reward_dist + reward_reach + reward_ctrl
        else:
            reward = reward_near + reward_ctrl

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        self.renderer.render_step()

        truncated = False
        if self.t == self._max_episode_steps:
            truncated = True
        return (
            ob,
            reward,
            False,
            truncated,
            dict(reward_eval=reward_dist + reward_reach, reward_train=reward_near + reward_ctrl),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:7],
                self.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                # self.get_body_com("goal")
            ]
        )
