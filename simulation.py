"""
file: simulation.py
----------------
Training loop with simulated faults
"""

import os
import logging
from datetime import timedelta
import wandb
import numpy as np
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, Subset, DataLoader
from tqdm import tqdm

from wrapper import DDPNoStop


class DistDataLoader:
    def __init__(self, data_dict, world_size, batch_size):
        self.world_size = world_size
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.init_from_dict(world_size)

    def init_from_dict(self, num_splits):
        # resets to saved original data
        train_x, train_y, val_x, val_y = (
            # make copies
            self.data_dict["train_x"].clone().detach(),
            self.data_dict["train_y"].clone().detach(),
            self.data_dict["val_x"].clone().detach(),
            self.data_dict["val_y"].clone().detach(),
        )
        self.train_len = len(train_x)
        self.eval_len = len(val_x)
        self.train_x = self._split_data(train_x, num_splits)
        self.train_y = self._split_data(train_y, num_splits)
        # only leader runs on eval set
        self.val_x = val_x
        self.val_y = val_y

        self.last_iter = 0  # last iter we split data on
        self.reset_loaders()

    def get_train_len(self):
        return len(self.train_x[0])

    def get_eval_len(self):
        return self.eval_len

    def _split_data(self, data, world_size):
        # splits data into world_size chunks
        split_data = torch.split(data, data.shape[0] // world_size)
        return split_data

    def reset_loaders(self):
        self.train_loaders = [
            DataLoader(
                TensorDataset(torch.tensor(self.train_x[rank]), torch.tensor(self.train_y[rank])),
                batch_size=self.batch_size,
            )
            for rank in range(self.world_size)
        ]
        self.val_loader = DataLoader(
            TensorDataset(torch.tensor(self.val_x), torch.tensor(self.val_y)),
            batch_size=self.batch_size,
        )

    def resplit(self, curr_iter):
        # checks if the world size has changed and re-splits data if necessary
        world_size = dist.get_world_size()
        if world_size < self.world_size:
            iter_diff = curr_iter - self.last_iter
            # re-split data
            new_train_x = []
            new_train_y = []
            for i in range(world_size):
                # get unprocessed data
                new_train_x.append(self.train_x[i][iter_diff:])
                new_train_y.append(self.train_y[i][iter_diff:])
            # split the remaining data evenly among the new ranks
            for i in range(self.world_size, world_size):
                new_train_x_split = torch.split(self.train_x[i][iter_diff:], world_size)
                new_train_y_split = torch.split(self.train_y[i][iter_diff:], world_size)
                # concat the new data with the old data
                for j in range(world_size):
                    new_train_x[j] = torch.cat((self.train_x[j], new_train_x_split[j]))
                    new_train_y[j] = torch.cat((self.train_y[j], new_train_y_split[j]))

            self.train_x = new_train_x
            self.train_y = new_train_y
            self.world_size = world_size
            self.last_iter = curr_iter
            return True  # if True, data was resplit and should be reinitialized
        return False

    def get_batch(self, split, rank, device):
        if split == "train":
            train_x, train_y = next(iter(self.train_loaders[rank]))
            return train_x.to(device), train_y.to(device)
        elif split == "val":
            val_x, val_y = next(iter(self.val_loader))
            return val_x.to(device), val_y.to(device)
        else:
            raise ValueError("Invalid split")


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


def train(
    rank,
    world_size,
    dataloader,
    model,
    criterion,
    fault_sim,
    num_epochs,
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
    # optimizer has references to model so need to init within process
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    torch.manual_seed(rank)  # for reproducibility

    # master has rank 0
    if (rank == 0) and wandb_project:
        wandb.login()
        wandb.init(
            project=wandb_project,
            config={
                "world_size": world_size,
                "num_epochs": num_epochs,
                "p_fail": fault_sim.p_fail,
            },
            name=wandb_name,
        )

    model.train()

    for epoch in tqdm(range(num_epochs)):
        dataloader.init_from_dict(world_size)
        for iter in range(dataloader.get_train_len()):
            batch_x, batch_y = dataloader.get_batch("train", rank, device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()

            for param in model.parameters():
                if param.grad is not None:
                    model._sync_gradients_hook(param)
            model.finish_gradient_synchronization()
            optimizer.step()

            # if faulted and world size changes, re-split train loader
            if dataloader.resplit(iter):
                rank = dist.get_rank()  # rank may have changed
                world_size = dist.get_world_size()
                dataloader.reset_loaders()

            if rank != 0 and fault_sim(iter):
                # simulate a fault
                fault_sim.fault_counter += 1
                logging.info(f"Simulated fault in rank {rank} iteration {iter}.")
                # log fault so it plot
                os._exit(0)

            if (rank == 0) and iter % 10 == 0 and wandb_project:
                wandb.log(
                    {"train_loss": loss.item()}, step=iter + epoch * dataloader.get_train_len()
                )

        if (rank == 0) and wandb_project:
            val_loss = eval_val_loss(rank, world_size, model, dataloader, criterion, device)
            logging.info(f"epoch {epoch+1} val_loss: {val_loss}")
            wandb.log({"val_loss": val_loss}, step=epoch + 1)

        scheduler.step()

    if rank == 0:
        # save model
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.module.state_dict(), os.path.join(output_dir, "model.pth"))


@torch.no_grad()
def eval_val_loss(rank, world_size, model, dataloader, criterion, device):
    # assumes that model is already wrapped in DDPNoStop
    model.eval()
    losses = torch.zeros(dataloader.get_eval_len())
    for i in range(dataloader.get_eval_len()):
        batch_x, batch_y = dataloader.get_batch("val", rank, device)
        logits = model(batch_x)
        loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        losses[i] = loss.item()
    model.train()
    return losses.mean()
