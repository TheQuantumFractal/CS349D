import sys
import os
import argparse
import logging
import wandb
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wrapper
from simulation import DataLoader, FaultSimulator, get_adam_optimizer, train


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


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
        default=10,
        help=("Batch size to use during training."),
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=2000,
        help="Number of training steps to perform",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=10,
        help="Number of evaluation batches to use for calculating validation loss",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
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

    mnist_dataset = {}
    data_sources = {
        "training_images": "train-images-idx3-ubyte.gz",  # 60,000 training images.
        "test_images": "t10k-images-idx3-ubyte.gz",  # 10,000 test images.
        "training_labels": "train-labels-idx1-ubyte.gz",  # 60,000 training labels.
        "test_labels": "t10k-labels-idx1-ubyte.gz",  # 10,000 test labels.
    }
    for key in ("training_images", "test_images"):
        with gzip.open(
            os.path.join(os.path.join(os.path.dirname(__file__), "_data", data_sources[key])), "rb"
        ) as mnist_file:
            mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=16).reshape(
                -1, 28 * 28
            )
    for key in ("training_labels", "test_labels"):
        with gzip.open(
            os.path.join(os.path.join(os.path.dirname(__file__), "_data", data_sources[key])), "rb"
        ) as mnist_file:
            mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)

    dataloader = DataLoader(
        mnist_dataset["training_images"],
        mnist_dataset["training_labels"],
        mnist_dataset["test_images"],
        mnist_dataset["test_labels"],
        world_size=args.world_size,
        batch_size=args.batch_size,
    )

    model = AlexNet()
    optimizer = get_adam_optimizer(model)
    faultsim = FaultSimulator(args.p_fail, seed=0xDEADBEEF)

    logging.info("starting training")
    mp.spawn(
        train,
        args=(
            args.world_size,
            dataloader,
            model,
            optimizer,
            faultsim,
            args.train_iters,
            args.eval_iters,
            args.eval_interval,
            args.output_dir,
            args.wandb_project,
            args.wandb_name,
        ),
        nprocs=args.world_size,
        join=True,
    )
    logging.info("finished running %s", sys.argv[0])
