import os
import argparse
import torch

from envs.env_norm import make_env
from gail.algo import SACExpert
from gail.utils import collect_demo
from envs import MultimodalEnvs
from gail.utils import set_seed


def run(args):
    env = make_env(MultimodalEnvs[args.env_id](num_modes=args.num_modes))

    # Set random seed
    set_seed(args.seed)
    env.seed(args.seed)

    weight_dir = args.weight_dir + args.env_id + '_' + str(args.num_modes) + '_modes/' + '{}.pth'.format(args.idx)
    if args.labeled:
        buffer_dir = args.buffer_dir + args.env_id + '_' + str(args.num_modes) + '_modes_lb'
    else:
        buffer_dir = args.buffer_dir + args.env_id + '_' + str(args.num_modes) + '_modes_ulb'

    algo = SACExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=weight_dir
    )

    buffer = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed,
        obs_horizon=args.obs_horizon,
        idx=args.idx,
        rend_env=args.rend_env
    )
    buffer.save(os.path.join(
        buffer_dir,
        'size{}_{}.pth'.format(args.buffer_size, args.idx)
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--idx', type=int, default=0)
    p.add_argument('--weight_dir', type=str, default='weights/')
    p.add_argument('--buffer_dir', type=str, default='buffers/')
    p.add_argument('--env_id', type=str, default='Pusher-v4')
    p.add_argument('--buffer_size', type=int, default=10**6)
    p.add_argument('--rend_env', type=bool, default=False)
    p.add_argument('--std', type=float, default=0.0)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--labeled', type=int, default=0)
    p.add_argument('--num_modes', type=int, default=6)
    p.add_argument('--obs_horizon', type=int, default=8)
    args = p.parse_args()
    run(args)
