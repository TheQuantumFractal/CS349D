"""
file: mnist.py
----------------
Evaluating trained model quality on MNIST dataset
"""

import sys
import os
import argparse
import logging
import wandb
import gzip
import numpy as np
import torch
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wrapper
from simulation import FaultSimulator, train


class Net(nn.Module):
    # from https://github.com/pytorch/examples/blob/main/mnist/main.py
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MNISTDataLoader:
    def __init__(self, train_x, train_y, val_x, val_y, num_splits, batch_size, shuffle=True):
        if shuffle:
            indices = np.random.permutation(len(train_x))
            train_x = train_x[indices]
            train_y = train_y[indices]
            indices = np.random.permutation(len(val_x))
            val_x = val_x[indices]
            val_y = val_y[indices]

        self.world_size = num_splits
        self.train_x = self._split_data(train_x, num_splits)
        self.train_y = self._split_data(train_y, num_splits)
        self.val_x = self._split_data(val_x, num_splits)
        self.val_y = self._split_data(val_y, num_splits)
        self.train_len = self.train_x[0].shape[0] // batch_size
        self.val_len = self.val_x[0].shape[0] // batch_size
        self.iter = 0
        self.batch_size = batch_size

    def _split_data(self, data, world_size):
        # splits data into world_size chunks
        split_data = np.array_split(data, world_size)
        return split_data

    def reshape_data(self, dead_nodes):
        # redistributes data to remaining nodes
        # TODO: this
        pass

    def get_batch(self, split, device):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if world_size != self.world_size:
            new_train_x = []
            old_train_x = []
            new_train_y = []
            old_train_y = []
            for i in self.train_x:
                new_train_x.append(i[self.iter :])
                old_train_x.append(i[: self.iter])
            for i in self.train_y:
                new_train_y.append(i[self.iter :])
                old_train_y.append(i[: self.iter])
            new_train_x = self._split_data(np.concatenate(new_train_x), world_size)
            old_train_x = self._split_data(np.concatenate(old_train_x), world_size)
            self.train_x = [np.concatenate((a, b)) for a, b in zip(old_train_x, new_train_x)]
            new_train_y = self._split_data(np.concatenate(new_train_y), world_size)
            old_train_y = self._split_data(np.concatenate(old_train_y), world_size)
            self.train_y = [np.concatenate((a, b)) for a, b in zip(old_train_y, new_train_y)]

            new_val_x = []
            old_val_x = []
            new_val_y = []
            old_val_y = []
            for i in self.val_x:
                new_val_x.append(i[self.iter :])
                old_val_x.append(i[: self.iter])
            for i in self.val_y:
                new_val_y.append(i[self.iter :])
                old_val_y.append(i[: self.iter])
            new_val_x = self._split_data(np.concatenate(new_val_x), world_size)
            old_val_x = self._split_data(np.concatenate(old_val_x), world_size)
            self.val_x = [np.concatenate((a, b)) for a, b in zip(old_val_x, new_val_x)]
            new_val_y = self._split_data(np.concatenate(new_val_y), world_size)
            old_val_y = self._split_data(np.concatenate(old_val_y), world_size)
            self.val_y = [np.concatenate((a, b)) for a, b in zip(old_val_y, new_val_y)]
            self.iter = 0
            self.train_len = self.train_x[0].shape[0] // self.batch_size
            self.val_len = self.val_x[0].shape[0] // self.batch_size

        if split == "train":
            x = self.train_x
            y = self.train_y
        elif split == "val":
            x = self.val_x
            y = self.val_y
        else:
            raise ValueError(f"Invalid split: {split}")
        if self.iter + self.batch_size > x[rank].shape[0]:
            self.iter = 0

        x = torch.tensor(x[rank][self.iter : self.iter + self.batch_size], dtype=torch.float32).to(
            device
        )
        y = torch.tensor(y[rank][self.iter : self.iter + self.batch_size], dtype=torch.long).to(
            device
        )
        self.iter += self.batch_size
        return x, y


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--p-fail",
        default=0.0,
        type=float,
        help="Probability of a fault occurring in simulator",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=10,
        help="Number of nodes to simulate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,  # MNIST is small and simple enough where this is fine
        help=("Batch size to use during training."),
    )
    parser.add_argument(
        "--num-epochs",
        type=float,
        default=14,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=1000,
        help="Number of evaluation batches to use for calculating validation loss",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Measure validation loss every `eval-interval` trainig steps",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to folder to write model configuration and trained model checkpoint",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="If set, log results to the specified wandb project",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        help="If set, log results to the specified wandb run",
    )
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST("../data", train=True, download=True, transform=transform)
    val_set = datasets.MNIST("../data", train=False, transform=transform)
    train_x, train_y = train_set.data, train_set.targets
    val_x, val_y = val_set.data, val_set.targets
    train_x = train_x.unsqueeze(1)  # for single channel
    val_x = val_x.unsqueeze(1)
    train_x = train_x.type(torch.float32)
    train_y = train_y.type(torch.LongTensor)
    val_x = val_x.type(torch.float32)
    val_y = val_y.type(torch.LongTensor)

    dataloader = MNISTDataLoader(
        train_x,
        train_y,
        val_x,
        val_y,
        num_splits=args.world_size,
        batch_size=args.batch_size,
    )

    model = Net()
    faultsim = FaultSimulator(args.p_fail, seed=0xDEADBEEF)

    logging.info("starting training")
    mp.spawn(
        train,
        args=(
            args.world_size,
            deepcopy(dataloader),
            deepcopy(model),
            nn.NLLLoss(reduction="sum"),
            "Adadelta",
            "StepLR",
            faultsim,
            args.num_epochs,
            dataloader.val_len,
            args.eval_interval,
            args.output_dir,
            args.wandb_project,
            args.wandb_name,
        ),
        nprocs=args.world_size,
        join=True,
    )
    logging.info("finished running %s", sys.argv[0])
