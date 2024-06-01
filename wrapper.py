"""
file: wrapper.py
----------------
basically the code in timeout.py but as a torch.nn.Module wrapper
"""

import os
import time
import logging
from datetime import timedelta

import torch
import torch.distributed as dist



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
