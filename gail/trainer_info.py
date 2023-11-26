import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gail.eval_metrics_online import EvalMetrics
from train_classifier import Net
from tqdm import trange


class Trainer:

    def __init__(self, env, env_test, algo, log_dir, num_steps=10 ** 5,
                 eval_interval=10 ** 3, num_eval_episodes=30, idx=0, rend_env=False, classifier_dir=None):
        super().__init__()

        self.idx = idx

        self.env = env
        # self.env.seed(seed)

        self.rend_env = rend_env

        self.env_test = env_test

        # Load the pre-trained classifier to calculate ENT and NMI metrics
        self.pretrained_classifier = Net(state_shape=env.observation_space.shape[0],
                                         action_shape=env.action_space.shape[0], output_dim=algo.dim_c).to(algo.device)
        state_dict = torch.load(classifier_dir, map_location=algo.device)
        self.pretrained_classifier.load_state_dict(state_dict)
        self.eval_metrics = EvalMetrics(len_discrete_code=algo.dim_c, device=algo.device)

        self.algo = algo
        self.log_dir = log_dir

        # Use tensorboard to record data
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        pbar = trange(self.num_steps)
        # Initialize the environment using a specific mode
        state = self.env.reset(mode_idx=torch.argmax(self.algo.latent_c).detach().cpu().numpy())
        for step in pbar:
            step = step + 1
            # Run a step
            state = self.algo.step(self.env, state)

            # Update the model
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Render the environment
            if step != 1:
                if self.rend_env:
                    self.env.render()

            # Evaluate the model
            if step % self.eval_interval == 0 or step == 1:
                mean_entropy, mean_nmi, avg_task_reward = self.evaluate_ess_info_gail()
                self.algo.save_models(self.model_dir)
                self.writer.add_scalar('Eval/mean_entropy', mean_entropy, step)
                self.writer.add_scalar('Eval/mean_nmi', mean_nmi, step)
                self.writer.add_scalar('Reward/rewards_eval', avg_task_reward, step)

            # Print the information
            if step % (self.eval_interval // 100) == 0 or step == 1:
                pbar.set_description(
                    f'mean_entropy: {mean_entropy:<6.4f}   '
                    f'mean_nmi: {mean_nmi:<6.4f}   '
                    f'avg_task_reward: {avg_task_reward:<6.4f}   '
                )

    def evaluate_ess_info_gail(self):
        # Calculate ENT， NMI and average task reward
        mean_entropy, mean_nmi, avg_task_reward = self.eval_metrics.evaluate(self.pretrained_classifier,
                                                                             self.algo, self.env_test)
        return mean_entropy, mean_nmi, avg_task_reward
