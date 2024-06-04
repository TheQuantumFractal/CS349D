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
from tqdm import tqdm

from wrapper import DDPNoStop


# logging
logging.basicConfig(level=logging.INFO)


class DataLoader:
    # wrapper class for data loading to our format
    def __init__(self, train_x, train_y, val_x, val_y, world_size, batch_size):
        self.train_x = self._split_data(train_x, world_size)
        self.train_y = self._split_data(train_y, world_size)
        self.val_x = self._split_data(val_x, world_size)
        self.val_y = self._split_data(val_y, world_size)
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

    def get_batch(self, split, rank, device):
        if split == "train":
            x = self.train_x
            y = self.train_y
        elif split == "val":
            x = self.val_x
            y = self.val_y
        else:
            raise ValueError(f"Invalid split: {split}")

        x = torch.tensor(x[rank][self.iter : self.iter + self.batch_size], dtype=torch.float32).to(
            device
        )
        y = torch.tensor(y[rank][self.iter : self.iter + self.batch_size], dtype=torch.long).to(
            device
        )
        self.iter += self.batch_size
        return x, y


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
    os.environ["MASTER_PORT"] = "29500"

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
    fault_sim,
    train_iters,
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
                "train_iters": train_iters,
                "eval_iters": eval_iters,
                "eval_interval": eval_interval,
                "p_fail": fault_sim.p_fail,
            },
            name=wandb_name,
        )

    # fetch first batch
    batch_x, batch_y = dataloader.get_batch(
        "train",
        rank,
        device=device,
    )

    for iter in tqdm(range(train_iters)):
        optimizer.zero_grad()
        output = model(batch_x)
        # async prefetch next batch during forward pass on GPU
        next_batch_x, next_batch_y = dataloader.get_batch(
            "train",
            rank,
            device=device,
        )
        loss = F.cross_entropy(output.view(-1, output.size(-1)), batch_y.view(-1))
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()

        if rank != 0 and fault_sim(iter):
            # simulate a fault
            fault_sim.fault_counter += 1
            logging.error(f"Simulated fault in rank {rank} iteration {iter}.")
            os._exit(0)

        if is_master_process and wandb_project:
            wandb.log({"train_loss": loss.item()}, step=iter)

        if is_master_process and iter != 0 and iter % eval_interval == 0 and wandb_project:
            val_loss = eval_val_loss(rank, world_size, model, dataloader, eval_iters, device)
            wandb.log({"val_loss": val_loss}, step=iter)

        # update batch
        batch_x, batch_y = next_batch_x, next_batch_y

    if is_master_process:
        # save model
        torch.save(model.module.state_dict(), os.path.join(output_dir, "model.pth"))


@torch.no_grad()
def eval_val_loss(rank, world_size, model, dataloader, eval_iters, device):
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
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        losses[i] = loss.item()
    model.train()
    return losses.mean()
