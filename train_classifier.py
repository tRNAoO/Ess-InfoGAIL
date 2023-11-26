import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
from gail.utils import set_seed
from envs import MultimodalEnvs
from envs.env_norm import make_env

DISC_LOGIT_INIT_SCALE = 1.0


class Net(nn.Module):
    def __init__(self, state_shape=6, action_shape=2, output_dim=6, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh(), device=None):
        super(Net, self).__init__()
        self.device = device

        self.input_dim = state_shape + action_shape

        layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(hidden_activation)
            curr_in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_units[-1], output_dim)

        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                pass
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        torch.nn.init.uniform_(self.linear.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        torch.nn.init.zeros_(self.linear.bias)

        self.trunk.train()
        self.linear.train()

    def forward(self, x):
        x = self.trunk(x)
        x = torch.softmax(self.linear(x), -1)
        return x


class Buffer:
    def __init__(self, path, device):
        self.states = []
        self.actions = []
        self.labels = []
        for i, p in enumerate(path):
            tmp = torch.load(p)
            state = tmp['state'].clone().to(device)
            action = tmp['action'].clone().to(device)
            self.states.append(state)
            self.actions.append(action)
            self.labels.append(torch.ones(state.shape[0], dtype=torch.int64).to(device)*i)

        self.states = torch.cat(self.states, dim=0)
        self.actions = torch.cat(self.actions, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

        self.buffer_size = self._n = self.states.shape[0]
        self._n_train = int(self._n*0.8)
        self._n_test = self._n - self._n_train
        idxes = np.array(list(range(self._n)))
        self.train_idxes = np.random.choice(idxes, size=self._n_train, replace=False)
        self.test_idxes = np.delete(idxes, self.train_idxes)
        self.device = device

    def train_sample(self, batch_size):
        idxes = np.random.choice(self.train_idxes, batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.labels[idxes]
        )

    def test_sample(self, batch_size):
        idxes = np.random.choice(self.test_idxes, batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.labels[idxes]
        )


def train(args, model, device, buffer, optimizer, epoch):
    model.train()
    CE_loss = nn.CrossEntropyLoss().cuda()
    num_updata = buffer._n_train // args.batch_size
    for _ in range(num_updata):
        states, actions, labels = buffer.train_sample(args.batch_size)
        optimizer.zero_grad()
        output = model(torch.cat([states, actions], dim=-1))
        loss = CE_loss(output, labels)
        loss.backward()
        optimizer.step()


def test(args, model, device, buffer, epoch):
    model.eval()
    CE_loss = nn.CrossEntropyLoss().cuda()
    test_loss = 0
    correct = 0
    num_test = buffer._n_test // args.test_batch_size
    with torch.no_grad():
        for _ in range(num_test):
            states, actions, labels = buffer.test_sample(args.test_batch_size)
            output = model(torch.cat([states, actions], dim=-1))
            test_loss = test_loss + CE_loss(output, labels).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = correct + pred.eq(labels.view_as(pred)).sum().item() / labels.shape[0]

    test_loss = test_loss / num_test

    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        epoch, test_loss, correct, num_test,
        100. * correct / num_test))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--env_id', type=str, default='2D-Trajectory')
    parser.add_argument('--num_modes', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    env = make_env(MultimodalEnvs[args.env_id](num_modes=args.num_modes))

    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]

    # Set random seed
    set_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    data_dir = sorted(glob.glob('buffers/{}_{}_modes_ulb/*.pth'.format(args.env_id, args.num_modes)))  # sort by name
    buffer = Buffer(data_dir, device)

    model = Net(state_shape, action_shape, output_dim=args.num_modes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, buffer, optimizer, epoch)
        test(args, model, device, buffer, epoch)

    if args.save_model:
        save_path = 'weights/{}_{}_modes_classifier/'.format(args.env_id, args.num_modes)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))


if __name__ == '__main__':
    main()
