import torch
from torch import nn
import torch.nn.functional as F

DISC_LOGIT_INIT_SCALE = 1.0


class GAILDiscrim(nn.Module):

    def __init__(self, state_shape, action_shape, dim_c=6, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh(), reward_i_coef=1.0, reward_us_coef=0.1,
                 reward_ss_coef=0.1, reward_t_coef=0.01, device=None, obs_his_steps=1,
                 normalizer=None):
        super().__init__()
        self.device = device
        self.dim_c = dim_c
        self.reward_i_coef = reward_i_coef
        self.reward_us_coef = reward_us_coef
        self.reward_ss_coef = reward_ss_coef
        self.reward_t_coef = reward_t_coef
        self.input_dim = (state_shape[0] - dim_c - 1 + action_shape[0])*obs_his_steps
        self.normalizer = normalizer

        layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(hidden_activation)
            curr_in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_units[-1], 1)
        self.classifier = nn.Linear(hidden_units[-1], dim_c)
        self.encoder_eps = nn.Linear(hidden_units[-1], 1)

        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                pass
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        torch.nn.init.uniform_(self.linear.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        torch.nn.init.zeros_(self.linear.bias)

        torch.nn.init.uniform_(self.classifier.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        torch.nn.init.zeros_(self.classifier.bias)

        self.trunk.train()
        self.linear.train()
        self.classifier.train()
        self.encoder_eps.train()
        self.L1Loss = nn.L1Loss(reduction='none').to(device)

    def forward(self, x):
        x = self.trunk(x)
        d = self.linear(x)
        eps = self.encoder_eps(x)
        c = torch.softmax(self.classifier(x), -1)
        return d, eps, torch.clamp(c, 1e-20, torch.inf)

    def calculate_reward(self, reward_t, states, actions):
        label_eps = states[:, -1, -self.dim_c-1].clone().unsqueeze(-1)
        label_c = states[:, -1, -self.dim_c:].clone()
        label_c = F.one_hot(torch.argmax(label_c, dim=-1), num_classes=self.dim_c)
        states = torch.reshape(states[:, :, :-self.dim_c-1], (states.shape[0], -1)).clone()
        actions = torch.reshape(actions[:, :, :], (states.shape[0], -1)).clone()
        with torch.no_grad():
            if self.normalizer is not None:
                states = self.normalizer.normalize_torch(states, self.device)

            d, eps, c = self.forward(torch.cat([states, actions], dim=-1))
            prob = 1 / (1 + torch.exp(-d))

            # Imitation reward
            reward_i = self.reward_i_coef * (-torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device))))

            # unsupervised reward
            reward_us = -self.reward_us_coef * self.L1Loss(eps, label_eps)

            log_c = torch.log(c)
            # Semi-supervised reward
            reward_ss = self.reward_ss_coef * torch.sum(label_c * log_c, dim=-1, keepdim=True)  # skill reward

            # Total reward
            reward = reward_i + reward_us + reward_ss + self.reward_t_coef * reward_t

        return reward, reward_i / (self.reward_i_coef + 1e-10), reward_us / (self.reward_us_coef + 1e-10), \
            reward_ss / (self.reward_ss_coef + 1e-10), reward_t

    def get_disc_logit_weights(self):
        return torch.flatten(self.linear.weight)

    def get_disc_weights(self):
        weights = []
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self.linear.weight))
        return weights

