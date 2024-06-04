"""
file: simulation.py 
----------------
simulates training loop and faults
"""
import os
import logging
import timeit # more accurate than time
from datetime import timedelta
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm

from wrapper import DDPNoStop

WORLD_SIZE = 5
MAX_ITERS = 100
GLOBAL_P_FAIL = 0.1
FAULT_SEED = None   # 0 for no faults, None for random seed


# logging
if 1:
    logging.basicConfig(level=logging.INFO)

class FaultSimulator:
    def __init__(self, p_fail, seed=None):
        assert 0 <= p_fail <= 1, "p_fail must be between 0 and 1"
        self.p_fail = p_fail
        self.fault_counter = 0
        if seed is None:    # None - random seed
            seed = torch.randint(0, 2**32, (1,)).item()
        self.seed = seed    # 0 - no faults
        np.random.seed(seed)
        logging.info(f"Fault simulator initialized with p_fail={p_fail} and seed={seed}")

    def __call__(self, iter):
        if self.seed != 0 and np.random.rand() < self.p_fail:
            self.fault_counter += 1
            return True
        return False


def _setup_dist(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "20420"

    timeout = timedelta(seconds=2)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timeout)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    return device


def train(rank, world_size, model, fault_sim):
    """
    boilerplate training loop
    """
    device = _setup_dist(rank, world_size)
    model = DDPNoStop(model).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # keep the same data for benchmarking 
    x = torch.randn(10, 10).to(device)
    y= torch.randint(0, 10, (10,)).to(device)

    def _train_step():
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()

    bench_times = []
    for iter in range(MAX_ITERS):
        start = timeit.default_timer()
        _train_step()

        if rank != 0 and fault_sim(iter):
            # simulate a fault
            fault_sim.fault_counter += 1
            logging.error(f"Simulated fault in rank {rank} iteration {iter}.")
            os._exit(0)
        
        bench_times.append(timeit.default_timer() - start)
    
    logging.info(f"Rank {rank} time per iteration: {torch.tensor(bench_times).mean()} ± {torch.tensor(bench_times).std()}")
    


if __name__ == "__main__":
    world_size = WORLD_SIZE
    model = torch.nn.Linear(10, 10)
    fault_sim = FaultSimulator(GLOBAL_P_FAIL, FAULT_SEED)
    mp.spawn(train, args=(world_size, model, fault_sim), nprocs=world_size)
    print("Total faults: ", fault_sim.fault_counter)