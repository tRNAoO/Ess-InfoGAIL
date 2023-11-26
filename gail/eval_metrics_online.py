import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy.stats import entropy


def compute_entropy(arr):
    rows = arr.shape[0]
    cols = arr.shape[1]
    l = []
    for i in range(rows):
        l.append(entropy(arr[i, :]))

    for i in range(cols):
        l.append(entropy(arr[:, i]))

    return np.mean(np.asarray(l))


class EvalMetrics:
    def __init__(self, z_dim=62, len_discrete_code=6, output_dim=1, input_size=28,
                 n_exp=20, sample_num=100, repeat_checks=100, num_eval_steps=1000, device=None):
        self.z_dim = z_dim
        self.len_discrete_code = len_discrete_code
        self.output_dim = output_dim
        self.input_size = input_size
        self.n_exp = n_exp
        self.sample_num = sample_num
        self.repeat_checks = repeat_checks
        self.num_eval_steps = num_eval_steps

        self.temp_y = torch.zeros((sample_num, 1)).to(device)
        for i in range(sample_num):
            self.temp_y[i] = self.temp_y[i] + (i / (sample_num / len_discrete_code))
        self.sample_y_ = (torch.zeros((sample_num, len_discrete_code)).
                          scatter_(1, self.temp_y.type(torch.LongTensor), 1).to(device))

        self.final_matrix = torch.zeros([n_exp, len_discrete_code, len_discrete_code]).to(device)

    def evaluate(self, classifier=None, algo=None, env_test=None):
        avg_task_reward = []
        for i in range(self.n_exp):
            states = []
            actions = []
            temp_res = torch.zeros([self.len_discrete_code, self.len_discrete_code])
            for j in range(self.len_discrete_code):
                latent_eps = np.random.rand() * 2. - 1.
                latent_c = np.zeros(self.len_discrete_code)
                latent_c[j] = 1
                state = env_test.reset(mode_idx=j)
                t = 0
                for _ in range(self.num_eval_steps):
                    t += 1
                    state = np.hstack([state, latent_eps, latent_c])
                    action = algo.exploit(state)
                    state, _, terminated, truncated, info = env_test.step(action)
                    done = terminated or truncated
                    avg_task_reward.append(info['reward_eval'])
                    states.append(state)
                    actions.append(action)
                    if done:
                        t = 0
                        state = env_test.reset(mode_idx=j)

            states = torch.tensor(np.vstack(states), dtype=torch.float32).to(algo.device)
            actions = torch.tensor(np.vstack(actions), dtype=torch.float32).to(algo.device)
            preds_gen = classifier(torch.cat([states, actions], dim=-1))
            top_ind_c1 = torch.argmax(preds_gen, dim=1)
            top_ind_c1 = torch.reshape(top_ind_c1, [self.len_discrete_code, self.num_eval_steps])
            for v in range(self.len_discrete_code):
                for w in range(self.num_eval_steps):
                    temp_res[v][top_ind_c1[v][w]] = temp_res[v][top_ind_c1[v][w]] + 1

            temp_res = temp_res / 10
            self.final_matrix[i] = temp_res

        avg_task_reward = np.mean(avg_task_reward) * env_test._max_episode_steps

        final_matrix = self.final_matrix.cpu().numpy() + 1e-10

        # computation of ENT
        a = final_matrix
        res = []
        for j in range(self.n_exp):
            if np.sum(a[j, :, :]) > 0:
                res.append(compute_entropy(a[j, :, :]))

        entropy_mean = np.mean(np.asarray(res))
        # entropy_std = np.std(np.asarray(res))

        # computation of NMI
        num_samples = 1000
        mat = final_matrix
        all_nmi = []
        for run_it in range(self.n_exp):
            curr_run = mat[run_it] * 10
            pred_label = np.zeros((num_samples * mat.shape[1]))
            gt_labels = np.zeros((num_samples * mat.shape[1]))
            tmp_offset = 0
            for class_it in range(mat.shape[1]):
                pred_label[class_it * num_samples:(class_it + 1) * num_samples] = class_it
                my_offset = 0
                for tmp_it in range(mat.shape[1]):
                    num_pred = int(round(curr_run[class_it][tmp_it]))
                    if num_pred > 0:
                        gt_labels[
                        class_it * num_samples + my_offset:class_it * num_samples + my_offset + num_pred] = tmp_it
                        my_offset = my_offset + num_pred
                tmp_offset = my_offset
            curr_nmi = nmi(gt_labels, pred_label)
            if (tmp_offset > 0):
                all_nmi.append(curr_nmi)
        nmi_mean = np.mean(np.asarray(all_nmi))
        # nmi_std = np.std(np.asarray(all_nmi))

        return entropy_mean, nmi_mean, avg_task_reward



