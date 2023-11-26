import os
import numpy as np
import torch
import random

from tqdm import tqdm
from .buffer import Buffer


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print('Random seed: {}'.format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def collect_demo(env, algo, buffer_size, device, std, p_rand, seed, obs_horizon, idx, rend_env=False):
    # env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=state_shape,
        action_shape=action_shape,
        device=device,
        obs_horizon=obs_horizon
    )

    state_his = np.zeros((obs_horizon, state_shape[0]))
    action_his = np.zeros((obs_horizon, action_shape[0]))
    state = env.reset(mode_idx=idx)
    num_episodes = 0
    t = 0
    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1
        state_his[:-1, :] = state_his[1:, :]
        state_his[-1, :] = state
        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit(state)
            action = add_random_noise(action, std)

        action_his[:-1, :] = action_his[1:, :]
        action_his[-1, :] = action

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.append(state, action, reward, done, terminated, next_state, state_his, action_his)

        if rend_env:
            env.render()

        if done:
            num_episodes += 1
            next_state = env.reset(mode_idx=idx)
            t = 0
            state_his[:, :] = 0.0
            action_his[:, :] = 0.0

        state = next_state

    return buffer
