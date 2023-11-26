import os
import torch
from torch import nn
from torch.optim import Adam
import numpy as np

from .base import Algorithm
from gail.buffer import Buffer
from gail.utils import soft_update, disable_gradient
from gail.network import (
    StateDependentPolicy, TwinnedStateActionFunction
)


class SAC(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.99,
                 batch_size=256, buffer_size=10**6, lr_actor=3e-4,
                 lr_critic=3e-4, lr_alpha=3e-4, units_actor=(256, 256),
                 units_critic=(256, 256), start_steps=10000, tau=5e-3, obs_horizon=8, idx=0):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        self.idx = idx

        # Replay buffer.
        self.buffer = Buffer(
            buffer_size=buffer_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            obs_horizon=obs_horizon
        )

        # Actor.
        self.actor = StateDependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)

        # Critic.
        self.critic = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.critic_target = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device).eval()

        soft_update(self.critic_target, self.critic, 1.0)
        disable_gradient(self.critic_target)

        # Entropy coefficient.
        self.alpha = 1.0
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.target_entropy = -float(action_shape[0])

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.batch_size = batch_size
        self.start_steps = start_steps
        self.tau = tau
        self.obs_horizon = obs_horizon
        self.state_his = np.zeros((obs_horizon, state_shape[0]))
        self.action_his = np.zeros((obs_horizon, action_shape[0]))

    def is_update(self, steps):
        return steps >= max(self.start_steps, self.batch_size)

    def step(self, env, state, step):
        self.state_his[:-1, :] = self.state_his[1:, :]
        self.state_his[-1, :] = state
        if step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(state)[0]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        self.action_his[:-1, :] = self.action_his[1:, :]
        self.action_his[-1, :] = action

        self.buffer.append(state, action, reward, done, terminated, next_state, self.state_his, self.action_his)

        if done:
            next_state = env.reset(mode_idx=self.idx)
            self.state_his[:, :] = 0.0
            self.action_his[:, :] = 0.0

        return next_state

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, terminated, next_states, _, _ = \
            self.buffer.sample(self.batch_size)

        self.update_critic(
            states, actions, rewards, dones, terminated, next_states, writer)
        self.update_actor(states, writer)
        self.update_target()

    def update_critic(self, states, actions, rewards, _, terminated, next_states,
                      writer):
        curr_qs1, curr_qs2 = self.critic(states, actions)
        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards + (1.0 - terminated) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/critic1', loss_critic1.item(), self.learning_steps)
            writer.add_scalar(
                'loss/critic2', loss_critic2.item(), self.learning_steps)

    def update_actor(self, states, writer):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)
        loss_actor = self.alpha * log_pis.mean() - torch.min(qs1, qs2).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        entropy = -log_pis.detach_().mean()
        loss_alpha = -self.log_alpha * (self.target_entropy - entropy)

        self.optim_alpha.zero_grad()
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()

        with torch.no_grad():
            self.alpha = self.log_alpha.exp().item()

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'loss/alpha', loss_alpha.item(), self.learning_steps)
            writer.add_scalar(
                'stats/alpha', self.alpha, self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)

    def update_target(self):
        soft_update(self.critic_target, self.critic, self.tau)

    def save_models(self, save_dir, idx):
        super().save_models(save_dir, idx)
        # We only save actor to reduce workloads.
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, '{}.pth'.format(idx))
        )


class SACExpert(SAC):

    def __init__(self, state_shape, action_shape, device, path,
                 units_actor=(256, 256)):
        self.actor = StateDependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.actor.load_state_dict(torch.load(path))

        disable_gradient(self.actor)
        self.device = device
