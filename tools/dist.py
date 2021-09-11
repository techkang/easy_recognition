from functools import partial

import torch
import torch.distributed as dist

from tools.comm import iterative_map
import logging


def gather_tensor(tensor):
    world_size = dist.get_world_size()
    tensor_placeholder = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.gather(tensor_placeholder, tensor, async_op=False)
    output_tensor = torch.cat(tensor_placeholder, dim=0)
    return output_tensor


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across devices.
    Args:
        tensors (list or tensor): tensors to perform all gather across all processes in
        all devices.
    """

    if not dist.is_initialized():
        return tensors
    func = iterative_map(gather_tensor, torch.Tensor, "raise")
    return func(tensors)


def reduce_tensor(tensor, average):
    dist.reduce(tensor, dst=0)
    if average:
        world_size = dist.get_world_size()
        tensor.mul_(1.0 / world_size)
    return tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list or tensor): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    if not dist.is_initialized():
        return tensors
    inner_func = partial(reduce_tensor, average=average)
    func = iterative_map(inner_func, torch.Tensor, "raise")
    return func(tensors)


def is_master_proc(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def get_world_size():
    """
    Get the size of the world.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get the rank of the current process.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
