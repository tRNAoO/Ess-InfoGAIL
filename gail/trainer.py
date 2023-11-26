import os
from time import time, sleep
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


class Trainer:

    def __init__(self, env, env_test, algo, log_dir, seed=0, idx=1, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=30):
        super().__init__()

        self.env = env
        self.idx = idx

        self.env_test = env_test

        self.algo = algo
        self.log_dir = log_dir

        self.summary_dir = os.path.join(log_dir, 'summary', str(self.idx))
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = log_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        self.start_time = time()
        state = self.env.reset(mode_idx=self.idx)
        pbar = trange(self.num_steps)
        for step in pbar:
            state = self.algo.step(self.env, state, step)

            if self.algo.is_update(step):
                self.algo.update(self.writer)

            if step % self.eval_interval == 0:
                self.evaluate(step, pbar)
        self.algo.save_models(self.model_dir, self.idx)

    def evaluate(self, step, pbar):
        episode_return = 0.0
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset(mode_idx=self.idx)
            truncated, terminated = False, False
            while not (truncated or terminated):
                action = self.algo.exploit(state)
                state, reward, terminated, truncated, _ = self.env_test.step(action)
                episode_return += reward

        mean_return = episode_return / self.num_eval_episodes

        self.writer.add_scalar('return/test', mean_return, step)
        pbar.set_description(f'Num steps: {step:<6}   '
                             f'Return: {mean_return:<5.1f}   '
                             f'Time: {self.time}')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
