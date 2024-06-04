"""
file: simulation.py
----------------
Training loop with simulated faults
"""

import os
import logging
import json
from datetime import timedelta
import wandb
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import TensorDataset, Subset, DataLoader
from tqdm import tqdm

from wrapper import DDPNoStop


# logging
logging.basicConfig(level=logging.INFO)


import torch
from torch.utils.data import DataLoader, Subset


import torch
from torch.utils.data import TensorDataset, Subset, DataLoader


class DistDataLoader:
    def __init__(self, train_x, train_y, val_x, val_y, num_splits, batch_size):
        self.num_splits = num_splits
        self.batch_size = batch_size
        self.train_len = len(train_x)
        self.val_len = len(val_x)

        # Create datasets
        train_dataset = TensorDataset(train_x, train_y)
        val_dataset = TensorDataset(val_x, val_y)

        # Split the train dataset into num_splits groups
        train_indices = list(range(len(train_dataset)))
        shuffled_indices = torch.randperm(len(train_indices))
        train_indices = [train_indices[i] for i in shuffled_indices]

        split_sizes = [len(train_indices) // num_splits] * num_splits
        split_sizes = [
            size + (1 if i < len(train_indices) % num_splits else 0)
            for i, size in enumerate(split_sizes)
        ]
        split_indices = [sum(split_sizes[:i]) for i in range(num_splits + 1)]

        self.train_split_datasets = [
            Subset(train_dataset, train_indices[start:end])
            for start, end in zip(split_indices[:-1], split_indices[1:])
        ]
        self.train_split_loaders = [
            DataLoader(split_dataset, batch_size=batch_size)
            for split_dataset in self.train_split_datasets
        ]

        # Create a single DataLoader for the validation dataset
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)

    def get_batch(self, split, rank, device):
        if split == "train":
            loader = self.train_split_loaders[rank]
        elif split == "val":
            loader = self.val_loader
        else:
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'val'.")

        for batch in loader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            return X, y


class FaultSimulator:
    def __init__(self, p_fail, seed=None, max_faults=None):
        assert 0 <= p_fail <= 1, "p_fail must be between 0 and 1"
        self.p_fail = p_fail
        self.fault_counter = 0
        self.max_faults = max_faults if max_faults is not None else float("inf")
        if seed is None:  # None - random seed
            seed = torch.randint(0, 2**32, (1,)).item()
        self.seed = seed  # 0 - no faults
        np.random.seed(seed)
        logging.info(f"Fault simulator initialized with p_fail={p_fail} and seed={seed}")

    def can_fail(self):
        return self.fault_counter < self.max_deaths

    def __call__(self, iter):
        if self.can_fail and np.random.rand() < self.p_fail:
            self.fault_counter += 1
            return True
        return False


def _setup_dist(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "31500"

    timeout = timedelta(seconds=200)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timeout)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    return device


def get_adam_optimizer(
    model, lr=1e-3, weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.98, adam_eps=1e-9
):
    # set up and return Adam optimizer
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
    params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": params_to_decay, "weight_decay": weight_decay},
        {"params": params_to_not_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )
    return optimizer


def train(
    rank,
    world_size,
    dataloader,
    model,
    optimizer,
    scheduler,
    criterion,
    fault_sim,
    num_epochs,
    eval_iters,
    eval_interval,
    output_dir,
    wandb_project,
    wandb_name,
):
    """
    boilerplate training loop

    Assumes dataloader is a DataLoader object with get_batch method
    """
    device = _setup_dist(rank, world_size)
    model = DDPNoStop(model).to(device)
    is_master_process = rank == 0

    torch.manual_seed(rank)  # for reproducibility

    if is_master_process and wandb_project:
        wandb.login()
        wandb.init(
            project=wandb_project,
            config={
                "world_size": world_size,
                "num_epochs": num_epochs,
                "eval_iters": eval_iters,
                "eval_interval": eval_interval,
                "p_fail": fault_sim.p_fail,
            },
            name=wandb_name,
        )

    for epoch in range(num_epochs):
        for iter in tqdm(range(dataloader.train_len)):
            batch_x, batch_y = dataloader.get_batch(
                "train",
                rank,
                device=device,
            )
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            model.finish_gradient_synchronization()
            optimizer.step()

            if rank != 0 and fault_sim(iter):
                # simulate a fault
                fault_sim.fault_counter += 1
                logging.error(f"Simulated fault in rank {rank} iteration {iter}.")
                os._exit(0)

            if is_master_process and iter % 10 == 0 and wandb_project:
                wandb.log({"train_loss": loss.item()}, step=iter)

        if is_master_process and wandb_project:
            val_loss = eval_val_loss(
                rank, world_size, model, dataloader, criterion, eval_iters, device
            )
            wandb.log({"val_loss": val_loss}, step=iter)

        scheduler.step()

    if is_master_process:
        # save model
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.module.state_dict(), os.path.join(output_dir, "model.pth"))


@torch.no_grad()
def eval_val_loss(rank, world_size, model, dataloader, criterion, eval_iters, device):
    # assumes that model is already wrapped in DDPNoStop
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        batch_x, batch_y = dataloader.get_batch(
            "val",
            rank,
            device=device,
        )
        logits = model(batch_x)
        loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        losses[i] = loss.item()
    model.train()
    return losses.mean()
