# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import os
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision.transforms import functional

import data.datasets as datasets
from tools.dist import get_rank, get_world_size


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (os.getpid() + int(datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, data_source, shuffle: bool = True):
        """
        Args:
            data_source (Dataset): the dataset to be sampled
            shuffle (bool): whether to shuffle the indices or not
        """
        self._size = len(data_source)
        assert self._size > 0
        self._shuffle = shuffle

        self._rank = get_rank()
        self._world_size = get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        while True:
            if self._shuffle:
                yield from np.random.permutation(self._size).tolist()
            else:
                yield from np.arange(self._size).tolist()


class InferenceSampler(Sampler):
    """
    Produce indices for inference.
    Inference needs to run on the __exact__ set of samples,
    therefore when the num_classes number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, data_source):
        """
        Args:
            data_source (Dataset): target dataset
        """
        self._size = len(data_source)
        assert self._size > 0
        # self._rank = get_rank()
        # self._world_size = get_world_size()
        self._rank = 0
        self._world_size = 1

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)


def resize_collector(batch, image_key="image", pad_uniform=False):
    images = [batch[i][image_key] for i in range(len(batch))]
    dest_h = max(i.shape[1] for i in images)
    dest_w = max(i.shape[2] for i in images)
    for i in range(len(batch)):
        image = batch[i][image_key]
        h, w = image.shape[-2:]
        if pad_uniform:
            left = random.randint(0, dest_w - w)
            top = random.randint(0, dest_h - h)
        else:
            left, top = 0, 0
        image = functional.pad(image, [left, top, dest_w - w - left, dest_h - h - top])
        batch[i][image_key] = image
    return default_collate(batch)


def build_dataloader(cfg, mode):
    name = cfg.dataset.name
    if mode == "test":
        name = cfg.dataset.test_dataset
    dataset = getattr(datasets, name)(cfg, mode=mode)
    if mode == "train":
        pin_memory = drop_last = True
        sampler = TrainingSampler(dataset, shuffle=True)
        batch_size = cfg.dataloader.batch_size
        num_workers = cfg.dataloader.num_workers
    else:
        pin_memory = drop_last = False
        sampler = InferenceSampler(dataset)
        if cfg.dataloader.eval_batch_size:
            batch_size = cfg.dataloader.eval_batch_size
        else:
            batch_size = cfg.dataloader.batch_size
        if cfg.dataloader.eval_num_workers >= 0:
            num_workers = cfg.dataloader.eval_num_workers
        else:
            num_workers = cfg.dataloader.num_workers
    if cfg.dataloader.collector == "default":
        collect_fn = default_collate
    elif cfg.dataloader.collector == "resize":
        collect_fn = partial(resize_collector, pad_uniform=mode == "train")
    else:
        raise ValueError("Unsupported collector!")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_reset_seed,
        collate_fn=collect_fn,
        prefetch_factor=cfg.dataloader.prefetch_factor if num_workers else 2,
    )
    return dataloader
