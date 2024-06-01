"""
file: simulation.py 
----------------
simulates training loop and faults
"""
import os
import hashlib
import torch
import timeit # more accurate than time
from datetime import timedelta
import torch.multiprocessing as mp
import torch.distributed as dist

from wrapper import DDPNoStop

GLOBAL_P_FAIL = 0 # 1e-5

class FaultSimulator:
    def __init__(self, p_fail, rank):
        assert 0 <= p_fail <= 1, "p_fail must be between 0 and 1"
        self.p_fail = p_fail
        self.rank = rank
        # seed hash fn with rank
        self.sha256 = hashlib.sha256(str(rank).encode())

    def __call__(self, iter):
        self.sha256.update(str(iter).encode())
        return int(self.sha256.hexdigest(), 16) % 100 < self.p_fail * 100


def _setup_dist(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "32420"

    timeout = timedelta(seconds=2)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timeout)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    fault_sim = FaultSimulator(GLOBAL_P_FAIL, rank)

    return device, fault_sim


def train(rank, world_size, model):
    """
    boilerplate training loop
    """
    device, fault_sim = _setup_dist(rank, world_size)
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
    for iter in range(100):
        start = timeit.default_timer()
        _train_step()

        if fault_sim(iter):
            # simulate a fault
            logging.error(f"Rank {rank} experienced a fault at iteration {iter}.")
            os._exit(0)
        
        bench_times.append(timeit.default_timer() - start)
    
    print(f"Rank {rank} time per iteration: {torch.tensor(bench_times).mean()} ± {torch.tensor(bench_times).std()}")
    


if __name__ == "__main__":
    world_size = 3
    model = torch.nn.Linear(10, 10)
    mp.spawn(train, args=(world_size, model), nprocs=world_size)