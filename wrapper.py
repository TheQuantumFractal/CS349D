"""
file: wrapper.py
----------------
basically the code in timeout.py but as a torch.nn.Module wrapper
"""

import os
import time
import logging
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist

CHECK_LEADER_HEALTH_TOPIC = 0
CHECK_FOLLOWER_HEALTH_TOPIC = 1
CHECK_NEW_NODE_TOPIC = 2

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

        # adjusts sensitivy of timeout (sec)
        self.comm_time = 25 * self.benchmark_comm() # multiply by arbitrary constant
        self.comm_time = 5
        print(self.comm_time)

        dist.barrier()

    def broadcast_params(self, async_op):
        for param in self.module.parameters():
            dist.broadcast(param.data, self.leader, async_op=async_op)

    def benchmark_comm(self):
        comm_times = []
        for _ in range(100):
            t = time.time()
            for param in self.module.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            dist.barrier()
            comm_times.append(time.time() - t)
        return np.mean(comm_times) * 1000

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def _sync_gradients_hook(self, param):
        if param.grad is not None:
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((handle, param.grad))
    
    def finish_gradient_synchronization(self, leader_end):     
        # as the leader, check if we've received a message from the newly awaken node
        new_node = torch.ones(1)
        if dist.get_rank() == self.leader:
            if leader_end.poll(timeout=1):
                # if we have, receive the message, expand the world size, and signal to other nodes to update their dist. 
                test = leader_end.recv()
                print(test)
                print('finished receiving')
                world_size = dist.get_world_size()
                leader_end.send([world_size])
                                
                for i in range(world_size):
                    if i != self.leader:
                        dist.send(new_node, dst=i, tag=CHECK_NEW_NODE_TOPIC)
                self._update_process_group(dist.get_rank(), world_size + 1)
        
        # synchronize gradients    
        try:
            for handle, grad in reversed(self.handles):
                handle.wait()
                grad /= self.world_size

            self.handles.clear()            
            dist.barrier()

        except: # timeout 
            # one reason for the timeout is if we have a new node:
            if dist.get_rank() != self.leader:
                handle = dist.irecv(new_node, src=self.leader, tag=CHECK_NEW_NODE_TOPIC)
                try:    
                    handle.wait(timeout=timedelta(seconds=self.comm_time))
                    self._update_process_group(dist.get_rank(), world_size + 1)
                except:
                    # only if unsuccessful, something is wrong and run fault recovery
                    self.fault_recovery()
            # otherwise, a node has failed and we need to start our fault recovery process.
            else:
                self.fault_recovery()

    def _update_process_group(self, my_rank, world_size):
        dist.distributed_c10d.destroy_process_group()
        os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 1)
        dist.init_process_group("gloo", rank=my_rank, world_size=world_size, timeout=timedelta(seconds=self.comm_time * world_size))
        # dist.monitored_barrier(timeout=timedelta(seconds=self.comm_time))
        dist.barrier()

    def fault_recovery(self):
        rank = dist.get_rank()
        logging.error(f"Rank {rank} detected a fault. Attempting to recover...")
        world_size = dist.get_world_size()
        process_group = dist.distributed_c10d._get_default_group()
        ranks = dist.get_process_group_ranks(process_group)

        alive = torch.zeros(len(ranks))
        alive[rank] = 1

        # while len(ranks) > 1:   # repeat leader election until undead leader is found
        if rank != self.leader:
            try:
                dist.send(alive, dst=self.leader, tag=CHECK_LEADER_HEALTH_TOPIC)
                handle = dist.irecv(alive, src=self.leader, tag=CHECK_FOLLOWER_HEALTH_TOPIC)
                handle.wait(timeout=timedelta(seconds=world_size * self.comm_time))
                indices = torch.where(alive == 1)[0].tolist()

                # world_size-len(indices) processes have died
                for i in range(world_size-len(indices)):
                    ranks.remove(len(ranks)-1)
                    world_size -= 1
                my_rank = indices.index(rank)
                logging.info(f"I was previously {rank} but I am now {my_rank} with world size {world_size}.")
                self._update_process_group(my_rank, world_size)
            except RuntimeError:    # timeout
                # Leader has died. Create new leader
                world_size -= 1
                ranks.remove(len(ranks)-1)
                my_rank = rank - 1
                logging.info(f"The leader has died. I was previously {rank} but I am now {my_rank} with world size {world_size}.")
                self._update_process_group(my_rank, world_size)
        else:   # leader branch
            arr_op = [torch.zeros(world_size) for _ in range(world_size)]
            handles = []
            for i in range(dist.get_world_size()):
                if i != self.leader:
                    try:
                        handle = dist.irecv(arr_op[i], src=i, tag=CHECK_LEADER_HEALTH_TOPIC)
                        handles.append(handle)
                    except:
                        # RuntimeError: Connection closed by peer
                        continue
            for handle in handles:
                try:
                    handle.wait(timeout=timedelta(seconds=self.comm_time))
                except:
                    continue
            for op in arr_op:
                alive += op
            indices = torch.where(alive == 1)[0].tolist()

            # world_size-len(indices) processes have died
            for i in range(world_size-len(indices)):
                ranks.remove(len(ranks)-1)
                world_size -= 1
            my_rank = 0   # leader is always rank 0
            logging.info(f"I am the leader. I was previously {rank} but I am now {my_rank} with world size {world_size}.")
            for i in indices:
                if i != self.leader:
                    dist.send(alive, dst=i, tag=CHECK_FOLLOWER_HEALTH_TOPIC)
            self._update_process_group(my_rank, world_size)
        
        logging.info("The system experienced a fault and successfully recovered.")
        self.broadcast_params(async_op=True)