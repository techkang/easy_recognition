"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import logging
from abc import ABC
from collections import defaultdict
from collections.abc import Hashable

import numpy as np
import torch.nn as nn
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""


class Counter:
    def __init__(self, iterable):
        self.counter = defaultdict(int)
        self.info = []
        for i in iterable:
            if isinstance(i, Hashable):
                self.counter[i] += 1
            else:
                self.info.append(i)

    def __repr__(self):
        return self.counter.__repr__() + "\n" + self.info.__repr__()


class selfdict(dict):
    def __missing__(self, key):
        return key


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


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


def rgb_to_ycbcr(img, only_y=True):
    dtype = img.dtype
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]
        ) / 255.0 + [16, 128, 128]
    if dtype == np.uint8:
        rlt = np.clip(rlt, 0, 255).astype(dtype)
    return rlt


def _convert_image(image, bolder, gray):
    image = image[bolder:-bolder, bolder:-bolder]
    image = image.astype(np.float32)
    if gray and image.shape[2] == 3:
        image = rgb_to_ycbcr(image)
    return image


def iterative_map(func, default_type=None, failed="raise"):
    assert failed in ("raise", "warning", "none")

    def res_func(data):

        if default_type is not None and isinstance(data, default_type):
            return func(data)
        elif isinstance(data, dict):
            return {k: res_func(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple, set)):
            return type(data)([res_func(i) for i in data])
        else:
            if failed == "raise":
                raise ValueError(f"unrecognized type: {type(data)} for {func}!")
            elif failed == "warning":
                logging.warning(f"unrecognized type: {type(data)} for {func}!")
            else:
                return data

    return res_func


class ToClass(ABC):
    @classmethod
    def __subclasshook__(cls, C):
        if cls is ToClass and any("to" in B.__dict__ for B in C.__mro__):
            return True
        return NotImplemented


def data_to(obj, device, failed="none"):
    def main_to(data):
        return data.to(device)

    return iterative_map(main_to, ToClass, failed)(obj)


def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad = False
            if hasattr(module, 'bias'):
                module.bias.requires_grad = False
            module.track_running_stats = False
