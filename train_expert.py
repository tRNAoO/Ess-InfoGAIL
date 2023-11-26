import os
import argparse
import torch

from envs.env_norm import make_env
from gail.algo import SAC
from gail.trainer import Trainer
from envs import MultimodalEnvs
from gail.utils import set_seed


def run(args):
    # Environment for training
    env = make_env(MultimodalEnvs[args.env_id](num_modes=args.num_modes, train_expert=True))
    # Environment for evaluation
    env_test = make_env(MultimodalEnvs[args.env_id](num_modes=args.num_modes, train_expert=True))

    # Set random seed
    set_seed(args.seed)
    env.seed(args.seed)
    env_test.seed(args.seed)

    # Redefine log_dir
    log_dir = args.log_dir + args.env_id + '_' + str(args.num_modes) + '_modes'

    # Initialize SAC algorithm
    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        idx=args.idx
    )

    # Initialize a trainer
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        idx=args.idx
    )

    # Start training
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--idx', type=int, default=0)
    p.add_argument('--num_steps', type=int, default=10**5)
    p.add_argument('--eval_interval', type=int, default=10**4)
    p.add_argument('--env_id', type=str, default='Pusher-v4')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num_modes', type=int, default=6)
    p.add_argument('--log_dir', type=str, default='weights/')
    args = p.parse_args()
    run(args)
