import os
import glob
import argparse
import torch
from envs import MultimodalEnvs
from envs.env_norm import make_env
from gail.buffer import SerializedBuffer
from gail.algo import ALGOS
from gail.trainer_info import Trainer
from gail.utils import set_seed


def run(args):
    # Environment for training
    env = make_env(MultimodalEnvs[args.env_id](num_modes=args.num_modes))
    # Environment for evaluation
    env_test = make_env(MultimodalEnvs[args.env_id](num_modes=args.num_modes))

    # Set random seed
    set_seed(args.seed)
    env.seed(args.seed)
    env_test.seed(args.seed)

    # Load dataset
    buffer_lb = args.buffer_dir + args.env_id + '_' + str(args.num_modes) + '_modes_lb/*'
    buffer_ulb = args.buffer_dir + args.env_id + '_' + str(args.num_modes) + '_modes_ulb/*'
    buffer_dir_lb = sorted(glob.glob(buffer_lb))  # sort by name
    buffer_dir_ulb = sorted(glob.glob(buffer_ulb))  # sort by name

    # Labeled dataset for training
    buffer_exp_lb = SerializedBuffer(
        path=buffer_dir_lb,
        device=torch.device("cuda" if args.cuda else "cpu"),
        num_modes=args.num_modes
    )

    # Unlabeled and imbalanced dataset for training
    buffer_exp_ulb = SerializedBuffer(
        path=buffer_dir_ulb,
        device=torch.device("cuda" if args.cuda else "cpu"),
        num_modes=args.num_modes,
        im_ratio=args.im_ratio
    )

    # Labeled dataset for evaluation
    buffer_exp_ulb_eval = SerializedBuffer(
        path=buffer_dir_ulb,
        device=torch.device("cuda" if args.cuda else "cpu"),
        num_modes=args.num_modes
    )

    # Initialize Ess-InfoGAIL algorithm
    algo = ALGOS[args.algo](
        buffer_exp_lb=buffer_exp_lb,
        buffer_exp_ulb=buffer_exp_ulb,
        buffer_exp_ulb_eval=buffer_exp_ulb_eval,
        state_shape=tuple([env.observation_space.shape[0] + args.num_modes + 1]),
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        epoch_ppo=args.epoch_ppo,
        epoch_disc=args.epoch_disc,
        surrogate_loss_coef=args.surrogate_loss_coef,
        disc_grad_penalty=args.disc_grad_penalty,
        value_loss_coef=args.value_loss_coef,
        disc_coef=args.disc_coef,
        us_coef=args.us_coef,
        ss_coef=args.ss_coef,
        reward_i_coef=args.reward_i_coef,
        reward_us_coef=args.reward_us_coef,
        reward_ss_coef=args.reward_ss_coef,
        reward_t_coef=args.reward_t_coef,
        info_max_coef1=args.info_max_coef1,
        info_max_coef2=args.info_max_coef2,
        info_max_coef3=args.info_max_coef3,
        dim_c=args.num_modes,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_prior=args.lr_prior,
        lr_disc=args.lr_disc,
        lr_q=args.lr_q,
        auto_lr=args.auto_lr,
        epoch_prior=args.epoch_prior,
        use_obs_norm=args.use_obs_norm,
        obs_horizon=args.obs_horizon,
        obs_his_steps=args.obs_his_steps,
        begin_weight=args.begin_weight
    )

    # Path to load model
    if args.model_dir:
        algo.load_models(args.model_dir)

    # Path to save log
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'{args.idx}')

    classifier_dir = args.classifier_dir + '{}_{}_modes_classifier/'.format(args.env_id, args.num_modes) + 'model.pth'

    # Initialize a trainer
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        idx=args.idx,
        rend_env=args.rend_env,
        classifier_dir=classifier_dir
    )

    # Start training
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--idx', type=int, default=-1, help='Training index')
    p.add_argument('--seed', type=int, default=9, help='Set a random seed')
    p.add_argument('--env_id', type=str, default='Walker2d-v4',
                   help='Environment ID: 2D-Trajectory, Reacher-v4, Pusher-v4, Walker2d-v4, Humanoid-v4')
    p.add_argument('--num_modes', type=int, default=3, help='Number of behavioral modes')
    p.add_argument('--rend_env', type=bool, default=False, help='Whether to render the environment')
    p.add_argument('--buffer_dir', type=str, default='buffers/', help='Path of the buffer')
    p.add_argument('--classifier_dir', type=str, default='weights/', help='Path of the weights')
    p.add_argument('--model_dir', type=str, default=None, help='Path of the pre-trained model')
    p.add_argument('--algo', type=str, default='Ess-InfoGAIL', help='Which algorithm to use')
    p.add_argument('--cuda', action='store_true', help='Whether to use GPU')
    p.add_argument('--use_obs_norm', action='store_true', help='Whether to normalize the observations')
    p.add_argument('--auto_lr', type=bool, default=True, help='Whether to use automatic learning rates')
    p.add_argument('--rollout_length', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=1000)
    p.add_argument('--num_steps', type=int, default=2000000)
    p.add_argument('--eval_interval', type=int, default=200000)
    p.add_argument('--obs_horizon', type=int, default=8)
    p.add_argument('--obs_his_steps', type=int, default=1)
    p.add_argument('--im_ratio', type=int, default=20)
    p.add_argument('--surrogate_loss_coef', type=float, default=4.0)
    p.add_argument('--disc_grad_penalty', type=float, default=0.1)
    p.add_argument('--value_loss_coef', type=float, default=5.0)
    p.add_argument('--disc_coef', type=float, default=20)
    p.add_argument('--us_coef', type=float, default=1.0)
    p.add_argument('--ss_coef', type=float, default=4.0)
    p.add_argument('--info_max_coef1', type=float, default=3.0)
    p.add_argument('--info_max_coef2', type=float, default=0.05)
    p.add_argument('--info_max_coef3', type=float, default=0.5)
    p.add_argument('--begin_weight', type=int, default=20)
    p.add_argument('--reward_i_coef', type=float, default=1.0)
    p.add_argument('--reward_us_coef', type=float, default=0.1)
    p.add_argument('--reward_ss_coef', type=float, default=0.1)
    p.add_argument('--reward_t_coef', type=float, default=0.005)
    p.add_argument('--epoch_ppo', type=int, default=20)
    p.add_argument('--epoch_disc', type=int, default=50)
    p.add_argument('--epoch_prior', type=int, default=1)
    p.add_argument('--lr_actor', type=float, default=3e-3)
    p.add_argument('--lr_critic', type=float, default=3e-3)
    p.add_argument('--lr_prior', type=float, default=3e-3)
    p.add_argument('--lr_disc', type=float, default=5e-3)
    p.add_argument('--lr_q', type=float, default=1e-2)
    args = p.parse_args()
    run(args)
