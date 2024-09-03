import os
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable

from .base import Algorithm
from gail.buffer import RolloutBufferSAC
from gail.utils import soft_update, disable_gradient
from gail.network import (
    StateDependentPolicy, TwinnedStateActionFunction
)


class SAC(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.99,
                 batch_size=256, buffer_size=10 ** 6, lr_actor=3e-4,
                 lr_critic=3e-4, lr_alpha=3e-4, units_actor=(256, 256),
                 units_critic=(256, 256), start_steps=10000, tau=5e-3, obs_horizon=8,
                 dim_c=6, mix_buffer=1, multi_value_num=1, update_interval=1000, epoch_sac=20):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        self.multi_value_num = multi_value_num
        self.update_interval = update_interval
        self.learning_steps_sac = 0
        self.epoch_sac = epoch_sac

        self.buffer = RolloutBufferSAC(
            buffer_size=buffer_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer,
            obs_horizon=obs_horizon,
        )

        # Actor.
        self.actor = StateDependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)

        # Critic.
        self.critic_set = []
        self.critic_target_set = []
        for i in range(self.multi_value_num):
            self.critic_set.append(TwinnedStateActionFunction(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=units_critic,
                hidden_activation=nn.ReLU(inplace=True)
            ).to(device))
            self.critic_target_set.append(TwinnedStateActionFunction(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=units_critic,
                hidden_activation=nn.ReLU(inplace=True)
            ).to(device).eval())

            soft_update(self.critic_target_set[i], self.critic_set[i], 1.0)
            disable_gradient(self.critic_target_set[i])

        self.dim_c = dim_c
        self.latent_eps = None
        self.latent_c = None
        self.prior_parameters = np.array(dim_c * [float(1) / dim_c])
        self.sample_latent_c()
        self.sample_latent_eps()

        # Entropy coefficient.
        self.alpha = 1.0
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.target_entropy = -float(action_shape[0])

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic_set = []
        for i in range(self.multi_value_num):
            self.optim_critic_set.append(Adam(self.critic_set[i].parameters(), lr=lr_critic))
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.batch_size = batch_size
        self.start_steps = start_steps
        self.tau = tau
        self.obs_horizon = obs_horizon
        self.state_his = np.zeros((obs_horizon, state_shape[0]))
        self.action_his = np.zeros((obs_horizon, action_shape[0]))

    def is_update(self, step):
        # return (step >= self.start_steps) and (step % self.update_interval == 0)
        return step % self.update_interval == 0

    def sample_latent_eps(self, batch_size=1):
        self.latent_eps = torch.tensor(np.random.rand(batch_size, 1) * 2. - 1., dtype=torch.float32, device=self.device)

    def sample_latent_c(self, batch_size=1, temperature=1.0):
        prior_parameters = torch.tensor(self.prior_parameters / temperature).to(self.device)
        prob = F.softmax(prior_parameters, dim=-1)
        self.latent_c = F.one_hot(prob.multinomial(batch_size), num_classes=self.dim_c)

    def step(self, env, state, step):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        state = torch.cat([state, self.latent_eps[0], self.latent_c[0]], dim=-1)
        self.state_his[:-1, :] = self.state_his[1:, :]
        self.state_his[-1, :] = state.detach().cpu().numpy()
        if step < self.start_steps:
            action = np.random.uniform(env.action_space.low, env.action_space.high, env.action_space.shape)
        else:
            action = self.explore(state)[0]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        self.action_his[:-1, :] = self.action_his[1:, :]
        self.action_his[-1, :] = action

        next_state_latent = np.hstack([next_state, self.latent_eps[0].detach().cpu().numpy(),
                                       self.latent_c[0].detach().cpu().numpy()])

        self.buffer.append(state, action, reward, done, terminated, next_state_latent,
                           self.state_his, self.action_his)

        if done:
            self.sample_latent_c()
            self.sample_latent_eps()
            next_state = env.reset(mode_idx=torch.argmax(self.latent_c).detach().cpu().numpy())
            self.state_his[:, :] = 0.0
            self.action_his[:, :] = 0.0

        return next_state

    def update_sac(self, states, actions, rewards_set, dones, terminated, next_states, writer):
        self.learning_steps_sac += 1
        self.update_critic(
            states.data, actions, rewards_set, dones, terminated, next_states.data, writer)
        self.update_actor(states.data, writer)
        self.update_target()

    def update_critic(self, states, actions, rewards_set, _, terminated, next_states,
                      writer):
        for i in range(len(self.critic_set)):
            curr_qs1, curr_qs2 = self.critic_set[i](states, actions)
            with torch.no_grad():
                next_actions, log_pis = self.actor.sample(next_states)
                next_qs1, next_qs2 = self.critic_target_set[i](next_states, next_actions)
                next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
            target_qs = rewards_set[i] + (1.0 - terminated) * self.gamma * next_qs

            loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
            loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

            self.optim_critic_set[i].zero_grad()
            (loss_critic1 + loss_critic2).backward(retain_graph=False)
            self.optim_critic_set[i].step()

            if self.learning_steps_sac % self.epoch_sac == 0:
                writer.add_scalar(
                    'Loss/critic1_{}'.format(i), loss_critic1.item(), self.learning_steps)
                writer.add_scalar(
                    'Loss/critic2_{}'.format(i), loss_critic2.item(), self.learning_steps)

    def update_actor(self, states, writer):
        actions, log_pis = self.actor.sample(states)
        qs1_set, qs2_set = [], []
        for i in range(len(self.critic_set)):
            qs1, qs2 = self.critic_set[i](states, actions)
            qs1_set.append(qs1)
            qs2_set.append(qs2)

        if self.multi_value_num <= 1:
            qs1_mean = qs1_set[0]
            qs2_mean = qs2_set[0]
        else:
            qs1_mean = torch.stack(qs1_set)
            qs2_mean = torch.stack(qs2_set)
            qs1_mean = (self.disc.reward_i_coef*qs1_mean[0, ...] + self.disc.reward_us_coef*qs1_mean[1, ...] +
                         self.disc.reward_ss_coef*qs1_mean[2, ...] + self.disc.reward_t_coef*qs1_mean[3, ...])
            qs2_mean = (self.disc.reward_i_coef*qs2_mean[0, ...] + self.disc.reward_us_coef*qs2_mean[1, ...] +
                         self.disc.reward_ss_coef*qs2_mean[2, ...] + self.disc.reward_t_coef*qs2_mean[3, ...])

        loss_actor = self.alpha * log_pis.mean() - torch.min(qs1_mean, qs2_mean).mean()

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

        if self.learning_steps_sac % self.epoch_sac == 0:
            writer.add_scalar(
                'Loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'Loss/alpha', loss_alpha.item(), self.learning_steps)
            # writer.add_scalar(
            #     'stats/alpha', self.alpha, self.learning_steps)
            # writer.add_scalar(
            #     'stats/entropy', entropy.item(), self.learning_steps)

    def update_target(self):
        for i in range(len(self.critic_set)):
            soft_update(self.critic_target_set[i], self.critic_set[i], self.tau)

    def save_models(self, save_dir, idx):
        super().save_models(save_dir, idx)
        # We only save actor to reduce workloads.
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, '{}.pth'.format(idx))
        )


# class SACExpert(SAC):
#
#     def __init__(self, state_shape, action_shape, device, path,
#                  units_actor=(256, 256)):
#         self.actor = StateDependentPolicy(
#             state_shape=state_shape,
#             action_shape=action_shape,
#             hidden_units=units_actor,
#             hidden_activation=nn.ReLU(inplace=True)
#         ).to(device)
#         self.actor.load_state_dict(torch.load(path))
#
#         disable_gradient(self.actor)
#         self.device = device
