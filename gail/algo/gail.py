import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

from .ppo import PPO
from .sac import SAC
from gail.network import GAILDiscrim
from gail.network.utils import Normalizer


class EssInfoGAIL(SAC):

    def __init__(self, buffer_exp_lb, buffer_exp_ulb, buffer_exp_ulb_eval, state_shape, action_shape, device, seed,
                 gamma=0.99, update_interval=5000, num_steps=2000000, mix_buffer=1,
                 batch_size=1000, start_steps=10000, units_actor=(64, 64), units_critic=(64, 64),
                 units_disc=(100, 100), epoch_sac=20, epoch_disc=50,
                 disc_coef=20.0, us_coef=1.0, ss_coef=10.0, disc_grad_penalty=0.1, disc_logit_reg=0.25,
                 disc_weight_decay=0.0005, dim_c=6, reward_i_coef=1.0, reward_us_coef=0.1, reward_ss_coef=0.1,
                 reward_t_coef=0.01, obs_horizon=8, klwt=10.0, surrogate_loss_coef=1.0, value_loss_coef=1.0,
                 info_max_coef=0.5, prior_soft_coef=0.99, lr_actor=1e-3, lr_critic=1e-3,
                 lr_disc=1e-3, lr_q=1e-3, use_obs_norm=True,
                 obs_his_steps=1, begin_rim=20, multi_value_num=4):
        super().__init__(
            state_shape=state_shape, action_shape=action_shape, device=device, seed=seed, gamma=gamma,
            mix_buffer=mix_buffer, units_actor=units_actor, units_critic=units_critic, dim_c=dim_c,
            obs_horizon=obs_horizon, lr_actor=lr_actor, lr_critic=lr_critic, multi_value_num=multi_value_num,
            batch_size=batch_size, update_interval=update_interval, epoch_sac=epoch_sac, start_steps=start_steps
        )
        self.dim_c = dim_c
        self.update_interval = update_interval
        self.num_steps = num_steps
        self.disc_grad_penalty = disc_grad_penalty
        self.disc_coef = disc_coef
        self.us_coef = us_coef
        self.ss_coef = ss_coef
        self.info_max_coef_on = 0
        self.info_max_coef = info_max_coef
        self.prior_soft_coef = prior_soft_coef
        self.disc_logit_reg = disc_logit_reg
        self.disc_weight_decay = disc_weight_decay
        self.obs_horizon = obs_horizon
        self.klwt = klwt
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_disc = lr_disc
        self.lr_q = lr_q
        self.obs_his_steps = obs_his_steps
        self.begin_rim = begin_rim
        self.multi_value_num = multi_value_num

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
            obs_his_steps=obs_his_steps,
            multi_value_num=multi_value_num
        ).to(device)

        # Loss function for semi-supervised encoder
        self.CE_loss = nn.CrossEntropyLoss().to(device)
        # Loss function for unsupervised encoder
        self.MSELoss = nn.MSELoss().to(device)
        # self.L1Loss = nn.L1Loss().to(device)

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
        total_learning_steps = self.num_steps // self.update_interval
        if self.learning_steps >= self.begin_rim:
            self.info_max_coef_on = min(self.info_max_coef * (self.learning_steps - self.begin_rim) * 1 /
                                        total_learning_steps, self.info_max_coef)

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Sample from current policy trajectories
            states, actions = self.buffer.sample(self.batch_size, self.obs_his_steps)[-2:]
            # Sample from labeled expert demonstrations
            states_exp_lb, actions_exp_lb, label_exp_lb = self.buffer_exp_lb.sample(self.batch_size, self.obs_his_steps)[-3:]
            # Sample from unlabeled expert demonstrations
            states_exp_ulb, actions_exp_ulb, label_exp_ulb = self.buffer_exp_ulb.sample(self.batch_size, self.obs_his_steps)[-3:]
            # Sample from unlabeled expert demonstrations for evaluation
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

        for n in range(self.epoch_sac):
            states, actions, rewards, dones, terminated, next_states, state_his, action_his = (
                self.buffer.sample(self.batch_size, self.obs_his_steps))

            # Calculate rewards
            rewards, rewards_i, rewards_us, rewards_ss, rewards_t = self.disc.calculate_reward(rewards, state_his, action_his)

            if self.multi_value_num > 1:
                rewards_set = [rewards_i, rewards_us, rewards_ss, rewards_t]
            else:
                rewards_set = [rewards]

            # Update PPO
            self.update_sac(states.clone(), actions, rewards_set, dones, terminated, next_states, writer)

            if n == self.epoch_sac - 1:
                writer.add_scalar('Reward/rewards', rewards.mean().item(), self.learning_steps)
                writer.add_scalar('Reward/rewards_i', rewards_i.mean().item(), self.learning_steps)
                writer.add_scalar('Reward/rewards_us', rewards_us.mean().item(), self.learning_steps)
                writer.add_scalar('Reward/rewards_ss', rewards_ss.mean().item(), self.learning_steps)
                writer.add_scalar('Reward/rewards_t', rewards_t.mean().item(), self.learning_steps)

        lr_actor = max(self.lr_actor - self.learning_steps * (self.lr_actor / (self.num_steps // self.update_interval)), 1e-5)
        lr_critic = max(self.lr_critic - self.learning_steps * (self.lr_critic / (self.num_steps // self.update_interval)), 1e-5)
        lr_disc = max(self.lr_disc - self.learning_steps * (self.lr_disc / (self.num_steps // self.update_interval)), 1e-5)
        for param_group in self.optim_actor.param_groups:
            param_group['lr'] = lr_actor
        for optim_critic in self.optim_critic_set:
            for param_group in optim_critic.param_groups:
                param_group['lr'] = lr_critic
        for param_group in self.optim_d.param_groups:
            param_group['lr'] = lr_disc

    def update_q(self, states_exp_lb, actions_exp_lb, label_exp_lb, states_exp_ulb, actions_exp_ulb, states_exp_ulb_eval, actions_exp_ulb_eval, label_exp_ulb_eval, writer, eps=1e-20):
        _, _, pred_c_lb = self.disc(torch.cat([states_exp_lb, actions_exp_lb], dim=-1))
        ss_loss = self.CE_loss(pred_c_lb, label_exp_lb)
        _, _, pred_c_ulb = self.disc(torch.cat([states_exp_ulb, actions_exp_ulb], dim=-1))

        # estimate the prior
        pred_c_ulb_mean = torch.mean(pred_c_ulb, dim=0)
        self.prior_parameters = (pred_c_ulb_mean.detach().cpu().numpy() * self.prior_soft_coef +
                                 self.prior_parameters * (1 - self.prior_soft_coef))

        # normalized information maximization
        info_max = torch.mean(-torch.sum(pred_c_ulb * torch.log(pred_c_ulb+eps), dim=-1))
        info_max_loss = self.info_max_coef_on * info_max

        loss = self.ss_coef * ss_loss + info_max_loss

        self.optim_q.zero_grad()
        loss.backward()
        self.optim_q.step()
        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar('Loss/ss_loss', ss_loss.item(), self.learning_steps)
            writer.add_scalar('Loss/info_max_loss', info_max.item(), self.learning_steps)

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

        disc_pi_loss = -F.logsigmoid(-logits_pi).mean()
        disc_exp_loss = -F.logsigmoid(logits_exp).mean()
        disc_loss = 0.5 * (disc_pi_loss + disc_exp_loss)

        us_loss = self.MSELoss(eps, label_eps)

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

        loss = self.disc_coef * disc_loss + self.us_coef * us_loss + self.disc_grad_penalty * grad_pen_loss + \
               self.disc_logit_reg * disc_logit_loss + self.disc_weight_decay * disc_weight_decay

        self.optim_d.zero_grad()
        loss.backward()
        self.optim_d.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar('Loss/disc', disc_loss.item(), self.learning_steps)
            writer.add_scalar('Loss/us_loss', us_loss.item(), self.learning_steps)

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
            'critic': [self.critic_set[i].state_dict() for i in range(len(self.critic_set))],
            'disc': self.disc.state_dict(),
            'optim_actor': self.optim_actor.state_dict(),
            'optim_critic': [self.optim_critic_set[i].state_dict() for i in range(len(self.optim_critic_set))],
            'optim_d': self.optim_d.state_dict(),
            'optim_q': self.optim_q.state_dict(),
        }, os.path.join(path, 'model.pth'))

    def load_models(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location='cuda:0')
        self.actor.load_state_dict(loaded_dict['actor'])
        [self.critic_set[i].load_state_dict(loaded_dict['critic'][i]) for i in range(len(self.critic_set))]
        self.disc.load_state_dict(loaded_dict['disc'])
        if load_optimizer:
            self.optim_actor.load_state_dict(loaded_dict['optim_actor'])
            [self.optim_critic_set[i].load_state_dict(loaded_dict['optim_critic'][i]) for i in range(len(self.optim_critic_set))]
            self.optim_d.load_state_dict(loaded_dict['optim_d'])
            self.optim_q.load_state_dict(loaded_dict['optim_q'])

