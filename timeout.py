import sys
import os
import time
import argparse
import logging
import timeit
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def single_node_setup(backend, device, rank, world_size):
    if device == "cuda":
        torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "32420"
    timeout = timedelta(seconds=2)
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timeout)

size = 4
def eep(rank):
    single_node_setup("gloo", "cpu", rank, size)
    t = torch.ones(1)
    leader = 0
    process_group = dist.distributed_c10d._get_default_group()
    ranks = dist.get_process_group_ranks(process_group)
    process_group = dist.new_group(ranks=ranks, timeout=timedelta(milliseconds=100))

    if rank != size - 1:
        try:
            dist.all_reduce(t, group=process_group, op=dist.ReduceOp.SUM)
        except:
            a = torch.zeros(size)
            a[rank] = 1
            if rank != leader:
                dist.send(a, dst=leader, group=process_group)
                time.sleep(0.1)
                dist.recv(a, src=leader, group=process_group)
                indices = torch.where(a == 1)[0]
                process_group = dist.new_group(ranks=indices, timeout=timedelta(seconds=2))
                dist.all_reduce(t, group=process_group, op=dist.ReduceOp.SUM)
            else:
                for i in range(size):
                    if i != leader:
                        try:
                            tmp = torch.zeros(size)
                            dist.recv(tmp, src=i, group=process_group)
                            a += tmp
                        except:
                            continue
                indices = torch.where(a == 1)[0].tolist()
                for i in indices:
                    if i != leader:
                        dist.send(a, dst=i, group=process_group)
                process_group = dist.new_group(ranks=indices, timeout=timedelta(seconds=2))
                dist.all_reduce(t, group=process_group, op=dist.ReduceOp.SUM)
        print(t)
        logging.info(f"Rank {rank} great success!")
        return
    else:
        time.sleep(1)
        logging.info(f"Rank {rank} failed successfully.")
        return


if __name__ == "__main__":
    mp.spawn(
        fn=eep,
        args=(),
        nprocs=size,
        join=True,
    )