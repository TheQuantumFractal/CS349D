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
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wrapper
from simulation import DistDataLoader, FaultSimulator, get_adam_optimizer, train


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

    dataloader = DistDataLoader(
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
