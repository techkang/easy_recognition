"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.
It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
from functools import partial

import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageOps


def build_transform(cfg, gray_scale=False):
    transform_list = []

    if gray_scale:
        transform_list.append(transforms.Grayscale(1))
    h = cfg.height
    transform_list.append(transforms.Lambda(partial(_scale_height, target_size=h)))
    transform_list.append(transforms.Lambda(partial(_scale_width, min_width=cfg.min_width, max_width=cfg.max_width,
                                                    pad_uniform=cfg.pad_position == "uniform")))

    if cfg.divisible:
        transform_list.append(transforms.Lambda(partial(_make_divisible, factor=cfg.divisible)))
    if cfg.blur:
        transform_list.append(transforms.GaussianBlur(5, sigma=3.))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(0.5, 0.5))
    return transforms.Compose(transform_list)


def _scale_height(img, target_size):
    origin_w, origin_h = img.size
    if origin_h == target_size:
        return img
    h = target_size
    w = int(origin_w * h / origin_h)
    return img.resize((w, h), Image.CUBIC)


def _scale_width(img, min_width=0, max_width=1e100, pad_uniform=False):
    width, height = img.size
    if width < min_width:
        left = 0
        if pad_uniform:
            left = np.random.randint(0, min_width - width + 1)
        img = ImageOps.expand(img, (left, 0, min_width - width - left, 0))
    if width > max_width:
        new_h = int(max_width / width * height)
        img = img.resize((max_width, new_h), Image.CUBIC)
        top = 0
        if pad_uniform:
            top = np.random.randint(0, height - new_h + 1)
        img = ImageOps.expand(img, (0, top, 0, height - new_h - top))
    return img


def _make_divisible(img, factor):
    w, h = img.size
    h = (h + factor - 1) // factor * factor
    w = (w + factor - 1) // factor * factor
    return img.resize((w, h), Image.CUBIC)


def denormalize(image, mean=(0.5,), std=(0.5,)):
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = np.clip(inv_tensor.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8).transpose((1, 2, 0))
    return inv_tensor
