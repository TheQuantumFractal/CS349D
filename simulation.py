"""
file: simulation.py
----------------
Training loop with simulated faults
"""

import os
import logging
import wandb
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from wrapper import DDPNoStop


# logging
logging.basicConfig(level=logging.INFO)


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
    optimizer_str,
    scheduler_str,
    fault_sim,
    num_epochs,
    eval_iters,
    eval_interval,
    output_dir,
    wandb_project,
    wandb_name,
    memory_efficient=False,
):
    """
    boilerplate training loop

    Assumes dataloader is a DataLoader object with get_batch method
    """
    device = _setup_dist(rank, world_size)
    os.makedirs(output_dir, exist_ok=True)
    model = DDPNoStop(model).to(device)

    if optimizer_str == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
    elif optimizer_str == "AdamW":
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
        params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": params_to_decay, "weight_decay": 0.1},
            {"params": params_to_not_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    elif optimizer_str == "RMSProp":
        # less memory than Adam
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    else:
        raise ValueError(f"Unrecognized optimizer class: {optimizer_str}")

    if scheduler_str == "StepLR":
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    elif scheduler_str == "Cosine":
        # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        pass

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

    model.train()
    real_iteration = 1

    for epoch in range(num_epochs):
        for iter in range(dataloader.train_len):
            rank = dist.get_rank()
            batch_x, batch_y = dataloader.get_batch(
                "train",
                device=device,
            )
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.view(-1, output.size(-1)), batch_y.view(-1))
            loss.backward()

            for param in model.parameters():
                if param.grad is not None:
                    model._sync_gradients_hook(param)
            model.finish_gradient_synchronization()
            if memory_efficient:
                torch.cuda.empty_cache()
            optimizer.step()

            if fault_sim(iter):
                # simulate a fault
                fault_sim.fault_counter += 1
                logging.error(f"Simulated fault in rank {rank} iteration {iter+1}.")
                os._exit(0)

            if is_master_process and (iter + 1) % 10 == 0 and wandb_project:
                logging.info(f"Epoch {epoch}, iteration {iter+1}, loss: {loss.item()}")
                wandb.log({"train_loss": loss.item()}, step=real_iteration)

            if is_master_process and (iter + 1) % eval_interval == 0:
                val_loss = eval_val_loss(
                    rank, world_size, model, dataloader, criterion, eval_iters, device
                )
                torch.save(model.module.state_dict(), os.path.join(output_dir, "model.pth"))

                if wandb_project:
                    logging.info(f"Epoch {epoch}, iteration {iter+1}, val_loss: {val_loss}")
                    wandb.log({"val_loss": val_loss}, step=real_iteration)

            if memory_efficient:
                torch.cuda.empty_cache()
                del batch_x, batch_y, output, loss

            real_iteration += 1

        # scheduler.step()

    if is_master_process:
        # save model
        torch.save(model.module.state_dict(), os.path.join(output_dir, "model.pth"))


@torch.no_grad()
def eval_val_loss(rank, world_size, model, dataloader, criterion, eval_iters, device):
    val_loss = 0
    for _ in tqdm(range(eval_iters)):
        batch_x, batch_y = dataloader.get_batch("val", device=device)
        output = model(batch_x)
        loss = criterion(output.view(-1, output.size(-1)), batch_y.view(-1))
        val_loss += loss.item()
    val_loss /= eval_iters
    return val_loss
