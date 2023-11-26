import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import random
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from gail.buffer import Buffer


def run(args):
    """The 2D-Trajectory expert.
    Expert trajectories are collected under 4 behavioral modes.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)

    scale = args.scale
    center = 0
    radius = 0
    order = 0
    theta_init = 0
    center_set = [[0, 5], [0, 10], [0, -5], [0, -10]]
    radius_set = [5, 10]
    order_set = [1, -1]
    theta_init_set = [0, 2 * np.pi]

    buffer_sizes = [int(2e4), int(2e2)]  # unlabeled data and labeled data
    modes = [0, 1, 2, 3]  # 4 modes
    for lb, buffer_size in enumerate(buffer_sizes):
        points_x_all = []
        points_y_all = []
        if lb:
            print('Collecting labeled demonstrations...')
            buffer_dir = args.buffer_dir + args.env_id + '_' + str(args.num_modes) + '_modes_lb'
        else:
            print('Collecting unlabeled demonstrations...')
            buffer_dir = args.buffer_dir + args.env_id + '_' + str(args.num_modes) + '_modes_ulb'
        for mode in modes:
            print('Collecting mode: ', mode)
            if mode == 0:
                center = center_set[0]
                radius = radius_set[0]
                order = order_set[0]
                theta_init = theta_init_set[0]
            elif mode == 1:
                center = center_set[1]
                radius = radius_set[1]
                order = order_set[0]
                theta_init = theta_init_set[0]
            elif mode == 2:
                center = center_set[2]
                radius = radius_set[0]
                order = order_set[1]
                theta_init = theta_init_set[1]
            elif mode == 3:
                center = center_set[3]
                radius = radius_set[1]
                order = order_set[1]
                theta_init = theta_init_set[1]

            noise_range = 0.3

            # Calculate the circumference of a circle
            circumference = 2 * math.pi * radius

            step = 1.0
            num_points = circumference // step

            points = []
            points_x = []
            points_y = []

            state_shape = tuple([3])
            action_shape = tuple([1])
            device = torch.device("cuda")
            obs_horizon = 8

            buffer = Buffer(
                buffer_size=buffer_size,
                state_shape=state_shape,
                action_shape=action_shape,
                device=device,
                obs_horizon=obs_horizon
            )

            state_his = np.zeros((obs_horizon, state_shape[0]))
            action_his = np.zeros((obs_horizon, action_shape[0]))

            k = 0
            flag = True
            while flag:
                i = 0
                last_point = None
                state_his[:, :] = 0
                action_his[:, :] = 0
                theta = theta_init
                theta_state = 0
                while (i <= num_points) & flag:
                    angle = order*((i * step / radius) - np.pi / 2)
                    x = center[0] + radius * math.cos(angle)
                    y = center[1] + radius * math.sin(angle)
                    # Add noise
                    if i > 0:
                        x += random.uniform(-noise_range, noise_range)
                        y += random.uniform(-noise_range, noise_range)
                    point = [x, y]

                    if last_point is None or step - 0.005 <= np.sqrt(
                            (last_point[0] - point[0]) ** 2 + (last_point[1] - point[1]) ** 2) <= step + 0.005:
                        points.append(point)
                        points_x.append(x)
                        points_y.append(y)
                        last_point = point
                        k += 1
                        i += 1

                        if i >= 2:
                            state = np.array([points_x[k-2]/scale, points_y[k-2]/scale, theta])
                            action = np.array([np.arctan2(points_y[k-1]-points_y[k-2], points_x[k-1]-points_x[k-2])])
                            if action < 0:
                                action = 2*np.pi + action
                            action_d = action - theta
                            theta = list(action)[0]
                            reward = 0
                            done = False
                            terminated = False
                            next_state = np.array([points_x[k-1]/scale, points_y[k-1]/scale, theta])

                            state_his[:-1, :] = state_his[1:, :]
                            state_his[-1, :] = state
                            action_his[:-1, :] = action_his[1:, :]
                            action_his[-1, :] = action_d

                            buffer.append(state, action_d, reward, done, terminated, next_state, state_his, action_his)

                    if buffer._n >= buffer_size:
                        flag = False
                        points_x_all.append(points_x)
                        points_y_all.append(points_y)
                        buffer.save(os.path.join(
                            buffer_dir,
                            'size{}_{}.pth'.format(buffer_size, mode)
                        ))
        if args.rend:
            # Plot the trajectory
            style = ['bo', 'ro', 'go', 'mo']
            for px, py, sy in zip(points_x_all, points_y_all, style):
                plt.plot(np.array(px)/scale, np.array(py)/scale, sy)
                plt.axis('equal')
            plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--idx', type=int, default=0)
    p.add_argument('--buffer_dir', type=str, default='../buffers/')
    p.add_argument('--env_id', type=str, default='2D-Trajectory')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--obs_horizon', type=int, default=8)
    p.add_argument('--num_modes', type=int, default=4)
    p.add_argument('--scale', type=float, default=10)
    p.add_argument('--rend', type=bool, default=False)
    args = p.parse_args()
    run(args)
