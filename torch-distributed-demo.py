"""
run::
    python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node=2 torch-distributed-demo.py

https://pytorch.org/docs/stable/notes/ddp.html
"""

import os
import io
import sys
import time
import subprocess as sp
import torch
from torch import nn
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def _debug_mem(msg):
    if local_rank == 1:
        print(msg)
        sp.call(f"nvidia-smi | grep {os.getpid()}", shell=True, stdout=sys.stdout)
        sys.stdout.flush()


dist.init_process_group(backend=None)  # nccl + gloo
local_rank = int(os.environ["LOCAL_RANK"])
local_size = int(os.environ["LOCAL_WORLD_SIZE"])
dev = torch.device(f"cuda:{local_rank}")
print(f"Start running torch distributed training on local rank {local_rank}/{local_size}.")
_debug_mem("start")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


model = Model()
model.to(dev)
_debug_mem("after model init")

ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
_debug_mem("after DDP wrapping")

# define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
_debug_mem("after optimizer init")

step = 0
while True:
    # forward pass
    outputs = ddp_model(torch.randn(20, 10, device=dev))
    labels = torch.randn(20, 10, device=dev)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

    print(f"[{local_rank}] step {step}")
    _debug_mem("step {step}")

    if step >= 3:
        break

    time.sleep(0.5)
    step += 1
