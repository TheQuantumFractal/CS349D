"""
file: wrapper.py
----------------
basically the code in timeout.py but as a torch.nn.Module wrapper
"""

import os
import time
import logging
import timeit # more accurate than time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class DDPNoStop(torch.nn.Module):
    """
    Wrapper around torch.nn.module that syncs gradient in a fault-tolerant manner.
    """
    def __init__(self, module):
        super(DDPNoStop, self).__init__()
        self.module = module
        self.leader = 0
        self.broadcast_params(async_op=True)

        # register hooks to sync gradients
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._sync_gradients_hook)
        self.handles = []

        dist.barrier()

    def broadcast_params(self, async_op):
        for param in self.module.parameters():
            dist.broadcast(param.data, self.leader, async_op=async_op)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def _sync_gradients_hook(self, param):
        if param.grad is not None:
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((handle, param.grad))
    
    def finish_gradient_synchronization(self):
        try:
            for handle, grad in reversed(self.handles):
                handle.wait()
                grad /= self.world_size
            
            self.handles.clear()
            dist.barrier()

        except: # timeout
            logging.error("The system experienced a fault. Attempting to recover.")
            self.fault_recovery()

    def fault_recovery(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        process_group = dist.distributed_c10d._get_default_group()
        ranks = dist.get_process_group_ranks(process_group)
        # dist.distributed_c10d._set_pg_timeout(timedelta(milliseconds=200), process_group)

        alive = torch.zeros(len(ranks))
        alive[rank] = 1

        if rank != self.leader:
            try:
                dist.send(alive, dst=self.leader)
                time.sleep(2)
                dist.recv(alive, src=self.leader)
                indices = torch.where(alive == 1)[0].tolist()
                my_rank = indices.index(rank)
                print(f"I was previously {rank} but I am now {my_rank}.")
                dist.distributed_c10d.destroy_process_group()
                dist.init_process_group("gloo", rank=my_rank, world_size=world_size-1, timeout=timedelta(seconds=2))
            except:
                # Leader has died. Create new leader
                ranks.remove(self.leader)
                my_rank = ranks.index(rank)
                print(f"The leader has died. I was previously {rank} but I am now {my_rank}.")
                dist.distributed_c10d.destroy_process_group()
                dist.init_process_group("gloo", rank=my_rank, world_size=world_size-1, timeout=timedelta(seconds=2))
        else:
            for i in range(world_size):
                if i != self.leader:
                    try:
                        tmp = torch.zeros(world_size)
                        dist.recv(tmp, src=i)
                        alive += tmp
                    except:
                        continue
            indices = torch.where(alive == 1)[0].tolist()
            my_rank = indices.index(rank)
            for i in indices:
                if i != self.leader:
                    dist.send(alive, dst=i)
            print(f"I am the leader. I was previously {rank} but I am now {my_rank}.")
            dist.distributed_c10d.destroy_process_group()
            dist.init_process_group("gloo", rank=my_rank, world_size=world_size-1, timeout=timedelta(seconds=2))
        
            logging.info("The system experienced a fault and successfully recovered.")
        self.broadcast_params(async_op=True)

def _setup_dist(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "32420"

    timeout = timedelta(seconds=2)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timeout)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    return device

def train(rank, world_size, model):
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
    for iter in range(100):
        start = timeit.default_timer()
        _train_step()

        if rank == 2 and iter == 20:
            # simulate a fault
            os._exit(0)
        
        bench_times.append(timeit.default_timer() - start)
    
    print(f"Rank {rank} time per iteration: {torch.tensor(bench_times).mean()} ± {torch.tensor(bench_times).std()}")
    


if __name__ == "__main__":
    world_size = 3
    model = torch.nn.Linear(10, 10)
    mp.spawn(train, args=(world_size, model), nprocs=world_size)