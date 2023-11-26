import os
import numpy as np
import torch


class SerializedBuffer:

    def __init__(self, path, device, num_modes, im_ratio=None):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.terminated = []
        self.next_states = []
        self.state_his = []
        self.action_his = []
        self.labels = []
        tmp = torch.load(path[0])
        len_data = tmp['state'].shape[0]
        im_weight = np.array([len_data]*num_modes)
        if im_ratio:
            # A fixed imbalanced weight
            im_weight = []
            num_ratio = 3  # How many ratios to set
            ratio_step = im_ratio**(1/(num_ratio - 1))
            int_part = num_modes // num_ratio  # Integer part
            dec_part = num_modes % num_ratio  # Decimal part
            for i in range(num_ratio):
                n = int_part
                if dec_part:
                    dec_part -= 1
                    n += 1
                im_weight.extend([int(len_data // ratio_step**i)]*n)

            # # A random imbalanced weight
            # len_data_min = len_data // im_ratio
            # im_weight = np.random.randint(len_data_min, len_data, num_modes)

            # # For Reacher-v4 with 6 modes, im_ratio=100
            # im_weight = np.array([1000, 1000, 10000, 10000, 100000, 100000])
            print('Num mode: ', num_modes)
            print('Imbalanced weight: ', np.array(im_weight)/len_data)
        for i, p in enumerate(path):
            tmp = torch.load(p)
            self.labels.append(torch.tensor([i]*len_data).to(device)[:im_weight[i]])
            self.states.append(tmp['state'].clone().to(device)[:im_weight[i]])
            self.actions.append(tmp['action'].clone().to(device)[:im_weight[i]])
            self.rewards.append(tmp['reward'].clone().to(device)[:im_weight[i]])
            self.dones.append(tmp['done'].clone().to(device)[:im_weight[i]])
            self.terminated.append(tmp['terminated'].clone().to(device)[:im_weight[i]])
            self.next_states.append(tmp['next_state'].clone().to(device)[:im_weight[i]])
            self.state_his.append(tmp['state_his'].clone().to(device)[:im_weight[i]])
            self.action_his.append(tmp['action_his'].clone().to(device)[:im_weight[i]])

        self.states = torch.cat(self.states, dim=0)
        self.actions = torch.cat(self.actions, dim=0)
        self.rewards = torch.cat(self.rewards, dim=0)
        self.dones = torch.cat(self.dones, dim=0)
        self.terminated = torch.cat(self.terminated, dim=0)
        self.next_states = torch.cat(self.next_states, dim=0)
        self.state_his = torch.cat(self.state_his, dim=0)
        self.action_his = torch.cat(self.action_his, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

        self.buffer_size = self._n = self.states.shape[0]
        self.device = device

    def sample(self, batch_size, obs_his_steps):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.terminated[idxes],
            self.next_states[idxes],
            self.state_his[idxes, -obs_his_steps:, :],
            self.action_his[idxes, -obs_his_steps:, :],
            self.labels[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device, obs_horizon):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.terminated = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.state_his = torch.empty(
            (buffer_size, obs_horizon, *state_shape), dtype=torch.float, device=device)
        self.action_his = torch.empty(
            (buffer_size, obs_horizon, *action_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, terminated, next_state, state_his, action_his):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = int(done)
        self.terminated[self._p] = int(terminated)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.state_his[self._p].copy_(torch.from_numpy(state_his))
        self.action_his[self._p].copy_(torch.from_numpy(action_his))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.terminated[idxes],
            self.next_states[idxes],
            self.state_his[idxes],
            self.action_his[idxes]
        )

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'terminated': self.terminated.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
            'state_his': self.state_his.clone().cpu(),
            'action_his': self.action_his.clone().cpu(),
        }, path)


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1, obs_horizon=8):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.terminated = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.mus = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.sigmas = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.state_his = torch.empty(
            (self.total_size, obs_horizon, *state_shape), dtype=torch.float, device=device)
        self.action_his = torch.empty(
            (self.total_size, obs_horizon, *action_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, terminated, log_pi, next_state, mu, sigma, state_his, action_his):
        self.states[self._p].copy_(state)
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = int(done)
        self.terminated[self._p] = int(terminated)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.mus[self._p].copy_(torch.from_numpy(mu))
        self.sigmas[self._p].copy_(torch.from_numpy(sigma))
        self.state_his[self._p].copy_(torch.from_numpy(state_his))
        self.action_his[self._p].copy_(torch.from_numpy(action_his))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self, obs_his_steps=1):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.terminated[idxes],
            self.log_pis[idxes],
            self.next_states[idxes],
            self.mus[idxes],
            self.sigmas[idxes],
            self.state_his[idxes, -obs_his_steps:, :],
            self.action_his[idxes, -obs_his_steps:, :]
        )

    def sample(self, batch_size, obs_his_steps=1):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.terminated[idxes],
            self.log_pis[idxes],
            self.next_states[idxes],
            self.mus[idxes],
            self.sigmas[idxes],
            self.state_his[idxes, -obs_his_steps:, :],
            self.action_his[idxes, -obs_his_steps:, :]
        )

    def clear(self):
        self.states = self.states.detach()
        self.states[:, :] = 0
        self.actions[:, :] = 0
        self.rewards[:, :] = 0
        self.dones[:, :] = 0
        self.terminated[:, :] = 0
        self.log_pis[:, :] = 0
        self.next_states[:, :] = 0
        self.mus[:, :] = 0
        self.sigmas[:, :] = 0
        self.state_his[:, :, :] = 0
        self.action_his[:, :, :] = 0
