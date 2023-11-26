import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

from .ppo import PPO
from gail.network import GAILDiscrim
from gail.network.utils import Normalizer


class EssInfoGAIL(PPO):

    def __init__(self, buffer_exp_lb, buffer_exp_ulb, buffer_exp_ulb_eval, state_shape, action_shape, device, seed,
                 gamma=0.99, rollout_length=50000, num_steps=2000000, mix_buffer=1,
                 batch_size=1000, learning_rate=1e-2, units_actor=(64, 64), units_critic=(64, 64),
                 units_disc=(100, 100), epoch_ppo=20, epoch_disc=50, clip_eps=0.2, lambd=0.95, max_grad_norm=1.0,
                 disc_coef=20.0, us_coef=1.0, ss_coef=10.0, disc_grad_penalty=0.1, disc_logit_reg=0.25,
                 disc_weight_decay=0.0005, dim_c=6, reward_i_coef=1.0, reward_us_coef=0.1, reward_ss_coef=0.1,
                 reward_t_coef=0.01, obs_horizon=8, klwt=10.0, surrogate_loss_coef=2.0, value_loss_coef=5.0,
                 info_max_coef1=0.5, info_max_coef2=1.0, info_max_coef3=0.01, lr_actor=1e-3, lr_critic=1e-3,
                 lr_prior=1e-3, lr_disc=1e-3, lr_q=1e-3, auto_lr=True, epoch_prior=20, use_obs_norm=True,
                 obs_his_steps=1, begin_weight=20):
        super().__init__(
            state_shape=state_shape, action_shape=action_shape, device=device, seed=seed, gamma=gamma,
            rollout_length=rollout_length, mix_buffer=mix_buffer, learning_rate=learning_rate, units_actor=units_actor,
            units_critic=units_critic, epoch_ppo=epoch_ppo, clip_eps=clip_eps, lambd=lambd,
            max_grad_norm=max_grad_norm, dim_c=dim_c, obs_horizon=obs_horizon, surrogate_loss_coef=surrogate_loss_coef,
            value_loss_coef=value_loss_coef, lr_actor=lr_actor, lr_critic=lr_critic, lr_prior=lr_prior,
            auto_lr=auto_lr, epoch_prior=epoch_prior
        )
        self.dim_c = dim_c
        self.rollout_length = rollout_length
        self.num_steps = num_steps
        self.disc_grad_penalty = disc_grad_penalty
        self.disc_coef = disc_coef
        self.us_coef = us_coef
        self.ss_coef = ss_coef
        self.info_max_coef1_on = 0
        self.info_max_coef2_on = 0
        self.info_max_coef3_on = 0
        self.info_max_coef1 = info_max_coef1
        self.info_max_coef2 = info_max_coef2
        self.info_max_coef3 = info_max_coef3
        self.disc_logit_reg = disc_logit_reg
        self.disc_weight_decay = disc_weight_decay
        self.obs_horizon = obs_horizon
        self.klwt = klwt
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_prior = lr_prior
        self.lr_disc = lr_disc
        self.lr_q = lr_q
        self.obs_his_steps = obs_his_steps
        self.begin_weight = begin_weight

        # Expert buffer
        self.buffer_exp_lb = buffer_exp_lb
        self.buffer_exp_ulb = buffer_exp_ulb
        self.buffer_exp_ulb_eval = buffer_exp_ulb_eval

        obs_shape_norm = (state_shape[0] - dim_c - 1) * obs_his_steps

        # Observation normalizer
        self.normalizer = None
        if use_obs_norm:
            self.normalizer = Normalizer(obs_shape_norm)

        # Discriminator
        self.disc = GAILDiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            dim_c=dim_c,
            device=device,
            hidden_units=units_disc,
            hidden_activation=nn.Tanh(),
            reward_i_coef=reward_i_coef,
            reward_us_coef=reward_us_coef,
            reward_ss_coef=reward_ss_coef,
            reward_t_coef=reward_t_coef,
            normalizer=self.normalizer,
            obs_his_steps=obs_his_steps
        ).to(device)

        # Loss function for semi-supervised encoder
        self.CE_loss = nn.CrossEntropyLoss().to(device)
        # Loss function for unsupervised encoder
        self.L1Loss = nn.L1Loss().to(device)

        self.learning_steps_disc = 0
        params_d = [
            {'params': self.disc.trunk.parameters(),
             'weight_decay': 10e-4, 'name': 'trunk'},
            {'params': self.disc.linear.parameters(),
             'weight_decay': 10e-4, 'name': 'head'},
            {'params': self.disc.encoder_eps.parameters(),
             'weight_decay': 10e-4, 'name': 'encoder_eps'},
        ]
        params_q = [
            {'params': self.disc.trunk.parameters(),
             'weight_decay': 10e-4, 'name': 'trunk'},
            {'params': self.disc.classifier.parameters(),
             'weight_decay': 10e-4, 'name': 'classifier'}
        ]
        self.optim_d = Adam(params_d, lr=self.lr_disc)
        self.optim_q = Adam(params_q, lr=self.lr_q)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer):
        self.learning_steps += 1

        # Gradually increase RIM coefficients with training
        # Do not use RIM at the beginning of training
        learning_num = self.num_steps//self.rollout_length
        if self.learning_steps >= self.begin_weight:
            self.info_max_coef1_on = min(self.info_max_coef1 * (self.learning_steps - self.begin_weight) * 1 /
                                         learning_num, self.info_max_coef1)
            self.info_max_coef2_on = min(self.info_max_coef2 * (self.learning_steps - self.begin_weight) * 1 /
                                         learning_num, self.info_max_coef2)
            self.info_max_coef3_on = min(self.info_max_coef3 * (self.learning_steps - self.begin_weight) * 1 /
                                         learning_num, self.info_max_coef3)

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories
            states, actions = self.buffer.sample(self.batch_size, self.obs_his_steps)[-2:]
            # Samples from labeled expert's demonstrations
            states_exp_lb, actions_exp_lb, label_exp_lb = self.buffer_exp_lb.sample(self.batch_size, self.obs_his_steps)[-3:]
            # Samples from unlabeled expert's demonstrations
            states_exp_ulb, actions_exp_ulb, label_exp_ulb = self.buffer_exp_ulb.sample(self.batch_size, self.obs_his_steps)[-3:]
            # Samples from unlabeled expert's demonstrations for evaluation
            states_exp_ulb_eval, actions_exp_ulb_eval, label_exp_ulb_eval = self.buffer_exp_ulb_eval.sample(self.batch_size, self.obs_his_steps)[-3:]

            label_eps = states[:, -1, -self.dim_c - 1].clone().unsqueeze(-1)
            label_c = states[:, -1, -self.dim_c:].clone()

            states = torch.reshape(states[:, :, :-self.dim_c - 1], (states.shape[0], -1)).clone()
            states_exp_lb = torch.reshape(states_exp_lb, (states_exp_lb.shape[0], -1)).clone()
            states_exp_ulb = torch.reshape(states_exp_ulb, (states_exp_ulb.shape[0], -1)).clone()
            states_exp_ulb_eval = torch.reshape(states_exp_ulb_eval, (states_exp_ulb_eval.shape[0], -1)).clone()

            actions = torch.reshape(actions, (actions.shape[0], -1)).clone()
            actions_exp_lb = torch.reshape(actions_exp_lb, (actions_exp_lb.shape[0], -1)).clone()
            actions_exp_ulb = torch.reshape(actions_exp_ulb, (actions_exp_ulb.shape[0], -1)).clone()
            actions_exp_ulb_eval = torch.reshape(actions_exp_ulb_eval, (actions_exp_ulb_eval.shape[0], -1)).clone()

            if self.normalizer is not None:
                with torch.no_grad():
                    states = self.normalizer.normalize_torch(states, self.device)
                    states_exp_lb = self.normalizer.normalize_torch(states_exp_lb, self.device)
                    states_exp_ulb = self.normalizer.normalize_torch(states_exp_ulb, self.device)
                    states_exp_ulb_eval = self.normalizer.normalize_torch(states_exp_ulb_eval, self.device)

            # Update semi-supervised encoder
            self.update_q(states_exp_lb, actions_exp_lb, label_exp_lb, states_exp_ulb, actions_exp_ulb,
                          states_exp_ulb_eval, actions_exp_ulb_eval, label_exp_ulb_eval, writer)

            # Update discriminator and unsupervised encoder
            self.update_disc(states, label_eps, label_c, actions, states_exp_ulb, actions_exp_ulb, writer)

            # Calculate the running mean and std of a data stream
            if self.normalizer is not None:
                self.normalizer.update(states.cpu().numpy())
                self.normalizer.update(states_exp_lb.cpu().numpy())
                self.normalizer.update(states_exp_ulb.cpu().numpy())
                self.normalizer.update(states_exp_ulb_eval.cpu().numpy())

        states, actions, rewards, dones, terminated, log_pis, next_states, mus, sigmas, state_his, action_his = self.buffer.get(self.obs_his_steps)

        # Calculate rewards
        rewards, rewards_i, rewards_us, rewards_ss, rewards_t = self.disc.calculate_reward(rewards, state_his, action_his)

        writer.add_scalar('Reward/rewards', rewards.mean().item(), self.learning_steps)
        writer.add_scalar('Reward/rewards_i', rewards_i.mean().item(), self.learning_steps)
        writer.add_scalar('Reward/rewards_us', rewards_us.mean().item(), self.learning_steps)
        writer.add_scalar('Reward/rewards_ss', rewards_ss.mean().item(), self.learning_steps)
        writer.add_scalar('Reward/rewards_t', rewards_t.mean().item(), self.learning_steps)

        # Update PPO
        self.update_ppo(states.clone(), actions, rewards, dones, terminated, log_pis, next_states, mus, sigmas, writer)

        # Clear buffer
        self.buffer.clear()

    def update_q(self, states_exp_lb, actions_exp_lb, label_exp_lb, states_exp_ulb, actions_exp_ulb, states_exp_ulb_eval, actions_exp_ulb_eval, label_exp_ulb_eval, writer, eps=1e-20):
        _, _, pred_c_lb = self.disc(torch.cat([states_exp_lb, actions_exp_lb], dim=-1))
        ss_loss = self.CE_loss(pred_c_lb, label_exp_lb)
        _, _, pred_c_ulb = self.disc(torch.cat([states_exp_ulb, actions_exp_ulb], dim=-1))

        pred_c_ulb_mean = torch.mean(pred_c_ulb, dim=0)
        info_max1 = torch.mean(-torch.sum(pred_c_ulb * torch.log(pred_c_ulb+eps), dim=-1))
        info_max2 = torch.sum(pred_c_ulb_mean * torch.log(pred_c_ulb_mean+eps))
        info_max3 = -torch.sum(pred_c_ulb_mean * torch.log(torch.softmax(torch.exp(self.prior_parameters.detach()), dim=-1)+eps))
        info_max_loss = (self.info_max_coef1_on * info_max1 + self.info_max_coef2_on * info_max2 +
                         self.info_max_coef3_on * info_max3)

        loss = self.ss_coef * ss_loss + info_max_loss

        self.optim_q.zero_grad()
        loss.backward()
        self.optim_q.step()
        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar('Loss/ss_loss', ss_loss.item(), self.learning_steps)
            writer.add_scalar('Loss/info_max_loss', info_max1.item()+info_max2.item()+info_max3.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                _, _, pred_c_ulb_eval = self.disc(torch.cat([states_exp_ulb_eval, actions_exp_ulb_eval], dim=-1))
                pred_c_lb = torch.argmax(pred_c_lb, dim=-1)
                pred_c_ulb_eval = torch.argmax(pred_c_ulb_eval, dim=-1)
                acc_class = torch.mean((pred_c_lb == label_exp_lb).float())
                acc_semi = torch.mean((pred_c_ulb_eval == label_exp_ulb_eval).float())
            writer.add_scalar('Acc/acc_lb', acc_class, self.learning_steps)
            writer.add_scalar('Acc/acc_semi', acc_semi, self.learning_steps)

    def update_disc(self, states, label_eps, label_c, actions, states_exp, actions_exp, writer):
        label_c = F.one_hot(torch.argmax(label_c, dim=-1), num_classes=self.dim_c)

        logits_pi, eps, pred_c = self.disc(torch.cat([states, actions], dim=-1))
        logits_exp, _, _ = self.disc(torch.cat([states_exp, actions_exp], dim=-1))

        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = 0.5 * (loss_pi + loss_exp)

        us_loss = self.L1Loss(eps, label_eps)

        # logit reg
        logit_weights = self.disc.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))

        # grad penalty
        sample_expert = torch.cat([states_exp, actions_exp], dim=-1)
        sample_expert.requires_grad = True
        disc = self.disc.linear(self.disc.trunk(sample_expert))
        ones = torch.ones(disc.size(), device=disc.device)
        disc_demo_grad = torch.autograd.grad(disc, sample_expert,
                                             grad_outputs=ones,
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        grad_pen_loss = torch.mean(disc_demo_grad)

        # weight decay
        disc_weights = self.disc.get_disc_weights()
        disc_weights = torch.cat(disc_weights, dim=-1)
        disc_weight_decay = torch.sum(torch.square(disc_weights))

        loss = self.disc_coef * loss_disc + self.us_coef * us_loss + self.disc_grad_penalty * grad_pen_loss + \
               self.disc_logit_reg * disc_logit_loss + self.disc_weight_decay * disc_weight_decay  # + self.klwt*kl_loss

        self.optim_d.zero_grad()
        loss.backward()
        self.optim_d.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar('Loss/disc', loss_disc.item(), self.learning_steps)
            writer.add_scalar('Loss/us_loss', us_loss.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
                pred_c = torch.argmax(pred_c, dim=-1)
                label_c = torch.argmax(label_c, dim=-1)
                acc_ulb = torch.mean((pred_c == label_c).float())
            writer.add_scalar('Acc/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('Acc/acc_exp', acc_exp, self.learning_steps)
            writer.add_scalar('Acc/acc_ulb', acc_ulb, self.learning_steps)

    def save_models(self, path, idx=0):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'disc': self.disc.state_dict(),
            'optim_actor': self.optim_actor.state_dict(),
            'optim_critic': self.optim_critic.state_dict(),
            'optim_d': self.optim_d.state_dict(),
            'optim_q': self.optim_q.state_dict(),
        }, os.path.join(path, 'model.pth'))

    def load_models(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location='cuda:0')
        self.actor.load_state_dict(loaded_dict['actor'])
        self.critic.load_state_dict(loaded_dict['critic'])
        self.disc.load_state_dict(loaded_dict['disc'])
        if load_optimizer:
            self.optim_actor.load_state_dict(loaded_dict['optim_actor'])
            self.optim_critic.load_state_dict(loaded_dict['optim_critic'])
            self.optim_d.load_state_dict(loaded_dict['optim_d'])
            self.optim_q.load_state_dict(loaded_dict['optim_q'])

