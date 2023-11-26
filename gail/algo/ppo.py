import os
import time
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable

from .base import Algorithm
from gail.buffer import RolloutBuffer
from gail.network import StateIndependentPolicy, StateFunction


def calculate_gae(values, rewards, dones, tm_dones, next_values, gamma, lambd):
    """
        Calculate the advantage using GAE
        'tm_dones=True' means dead or win, there is no next state s'
        'dones=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps).
        When calculating the adv, if dones=True, gae=0
        Reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/5.PPO-continuous/ppo_continuous.py
    """
    with torch.no_grad():
        # Calculate TD errors.
        deltas = rewards + gamma * next_values * (1 - tm_dones) - values
        # Initialize gae.
        gaes = torch.empty_like(rewards)

        # Calculate gae recursively from behind.
        gaes[-1] = deltas[-1]
        for t in reversed(range(rewards.size(0) - 1)):
            gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995, rollout_length=2048,  mix_buffer=20,
                 learning_rate=1e-3, units_actor=(64, 64), units_critic=(64, 64), epoch_ppo=10, clip_eps=0.2,
                 lambd=0.97, max_grad_norm=1.0, desired_kl=0.01, surrogate_loss_coef=2., value_loss_coef=5.,
                 entropy_coef=0., bounds_loss_coef=10., dim_c=6, obs_horizon=8, lr_actor=1e-3, lr_critic=1e-3,
                 lr_prior=1e-3, auto_lr=True, epoch_prior=20):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        self.learning_rate = learning_rate
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_prior = lr_prior
        self.auto_lr = auto_lr
        self.epoch_prior = epoch_prior

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer,
            obs_horizon=obs_horizon,
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.max_grad_norm = max_grad_norm
        self.desired_kl = desired_kl
        self.surrogate_loss_coef = surrogate_loss_coef
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.bounds_loss_coef = bounds_loss_coef
        self.dim_c = dim_c
        self.latent_eps = None
        self.latent_c = None
        temp = torch.tensor(dim_c * [float(1) / dim_c]).to(device)
        # To ensure more stable gradient updates for the latent skill distribution and
        # avoid issues with values that are too small or too large, we use the logarithmic
        # form of the latent skill distribution for gradient updates.
        self.prior_parameters = Variable(temp, requires_grad=True)
        self.sample_latent_c()
        self.sample_latent_eps()

        self.optim_actor = Adam([{'params': self.actor.parameters()}], lr=lr_actor)
        self.optim_critic = Adam([{'params': self.critic.parameters()}], lr=lr_critic)
        self.optim_prior = Adam([{'params': self.prior_parameters}], lr=lr_prior)

        self.obs_horizon = obs_horizon
        self.state_his = np.zeros((obs_horizon, state_shape[0]))
        self.action_his = np.zeros((obs_horizon, action_shape[0]))

    def is_update(self, step):
        return step % self.rollout_length == 0

    def sample_latent_eps(self, batch_size=1):
        self.latent_eps = torch.tensor(np.random.rand(batch_size, 1) * 2. - 1., dtype=torch.float32, device=self.device)

    def sample_latent_c(self, batch_size=1):
        self.latent_c = self.approx_latent(torch.exp(self.prior_parameters), batch_size)

    def sample_gumbel(self, shape, eps=1e-20):
        u = torch.FloatTensor(shape, self.dim_c).to(self.device).uniform_(0, 1)
        return -torch.log(-torch.log(u + eps) + eps)

    def gumbel_softmax_sample(self, logits, temp, batch_size):
        y = logits + self.sample_gumbel(batch_size)
        return torch.nn.functional.softmax(y / temp, dim=-1)

    def approx_latent(self, params, batch_size):
        params = F.softmax(params, dim=-1)
        log_params = torch.log(params)
        c = self.gumbel_softmax_sample(log_params, temp=0.1, batch_size=batch_size)
        return c

    def step(self, env, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        state = torch.cat([state, self.latent_eps[0], self.latent_c[0]], dim=-1)
        self.state_his[:-1, :] = self.state_his[1:, :]
        self.state_his[-1, :] = state.detach().cpu().numpy()
        action, log_pi = self.explore(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        self.action_his[:-1, :] = self.action_his[1:, :]
        self.action_his[-1, :] = action

        means = self.actor.means.detach().cpu().numpy()[0]
        stds = (self.actor.log_stds.exp()).detach().cpu().numpy()[0]
        next_state_latent = np.hstack([next_state, self.latent_eps[0].detach().cpu().numpy(),
                                       self.latent_c[0].detach().cpu().numpy()])
        self.buffer.append(state, action, reward, done, terminated, log_pi, next_state_latent, means, stds,
                           self.state_his, self.action_his)

        if done:
            self.sample_latent_c()
            self.sample_latent_eps()
            next_state = env.reset(mode_idx=torch.argmax(self.latent_c).detach().cpu().numpy())
            self.state_his[:, :] = 0.0
            self.action_his[:, :] = 0.0

        return next_state

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, terminated, log_pis, next_states, mus, sigmas = \
            self.buffer.get()
        self.update_ppo(
            states, actions, rewards, dones, terminated, log_pis, next_states, mus, sigmas, writer)

    def update_ppo(self, states, actions, rewards, dones, terminated, log_pis, next_states, mus, sigmas,
                   writer):
        with torch.no_grad():
            values = self.critic(states.detach())
            next_values = self.critic(next_states.detach())

        targets, gaes = calculate_gae(
            values, rewards, dones, terminated, next_values, self.gamma, self.lambd)

        for i in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states.detach(), targets, writer)
            # To minimize computational time, we restrict the update of the latent skill distribution to
            # only the first iteration of policy updates.
            if i < self.epoch_prior:
                retain_graph = True
            else:
                retain_graph = False
                states = states.data

            self.update_actor(states, actions, log_pis, gaes, mus, sigmas, writer, retain_graph=retain_graph)

    def update_critic(self, states, targets, writer):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()
        loss_critic = loss_critic * self.value_loss_coef

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'Loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, mus_old, sigmas_old, writer, retain_graph=False):
        self.optim_actor.zero_grad()
        self.optim_prior.zero_grad()
        log_pis = self.actor.evaluate_log_pi(states, actions)
        mus = self.actor.means
        sigmas = (self.actor.log_stds.exp()).repeat(mus.shape[0], 1)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()
        loss_actor = loss_actor * self.surrogate_loss_coef
        if self.auto_lr:
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigmas / sigmas_old + 1.e-5) + (
                                torch.square(sigmas_old) + torch.square(mus_old - mus)) / (
                                2.0 * torch.square(sigmas)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.lr_actor = max(1e-5, self.lr_actor / 1.5)
                    self.lr_critic = max(1e-5, self.lr_critic / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.lr_actor = min(1e-2, self.lr_actor * 1.5)
                    self.lr_critic = min(1e-2, self.lr_critic * 1.5)

                for param_group in self.optim_actor.param_groups:
                    param_group['lr'] = self.lr_actor
                for param_group in self.optim_critic.param_groups:
                    param_group['lr'] = self.lr_critic
                for param_group in self.optim_d.param_groups:
                    param_group['lr'] = self.lr_actor

        loss = loss_actor

        loss.backward(retain_graph=retain_graph)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()
        self.optim_prior.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'Loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'Loss/entropy', entropy.item(), self.learning_steps)
            writer.add_scalar(
                'Loss/learning_rate', self.lr_actor, self.learning_steps)

    def save_models(self, save_dir, idx):
        pass
