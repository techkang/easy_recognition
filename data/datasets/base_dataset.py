import json
import logging
from pathlib import Path

import PIL.Image
import cv2
import numpy as np
from torch.utils.data import Dataset

import model.convertor.charset
from data.lmdb_loader import get_loader
from data.transform import build_transform


class OCRDataset(Dataset):
    def __init__(self, cfg, mode):
        if mode == 'train':
            dataset = cfg.dataset.train
        elif mode == 'eval':
            dataset = cfg.dataset.eval
        else:
            dataset = cfg.dataset.test

        self.base_path = Path(cfg.dataset.path)
        self.mode = mode
        self.target_length = cfg.charset.target_length
        self.dataset_names = dataset.dataset

        if mode == "train":
            logging.info("loading train annotations, this may take a while.")
        self.loader = None

        self.loader_name = dataset.loader
        loader = get_loader(self.loader_name)
        self.repeat = dataset.repeat if len(dataset.repeat) == len(dataset.dataset) else tuple(
            [1] * len(dataset.dataset))
        if self.loader_name == "caffe_loader":
            self.repeat = [True] * len(dataset.dataset)
        for name, repeat in zip(self.dataset_names, self.repeat):
            loader.load(str(Path(self.base_path, name)), repeat)
        self.total_length = loader.count()
        self.transform = build_transform(dataset, cfg.dataset.gray_scale)
        self.charset = getattr(model.convertor.charset, cfg.charset.name)(cfg)
        logging.info(f'{len(self)} images for {mode} loaded from {" ".join(dataset.dataset)}.')

    def __getitem__(self, item):
        if self.loader is None:
            self.loader = get_loader(self.loader_name)
            for name, repeat in zip(self.dataset_names, self.repeat):
                self.loader.load(str(Path(self.base_path, name)), repeat)
        if self.mode == "train":
            while True:
                item = item % self.total_length
                success, image_bytes, label, uri = self.loader.read(int(item))
                label = "".join(json.loads(label)["labels"])
                bytes = np.frombuffer(image_bytes, np.uint8)
                if len(bytes):
                    img = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
                else:
                    img = None
                if img is None or img.shape[0] < 4 or img.shape[1] < 4:
                    # logging.warning(f"broken image: {uri}, skip!")
                    pass
                elif len(label) + 2 > self.target_length:
                    pass
                else:
                    break
                item = (item + 19) % self.total_length
        else:
            success, image_bytes, label, uri = self.loader.read(item)
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            label = "".join(json.loads(label)["labels"])[:self.target_length - 2]
        uri = uri.decode("utf8")
        dataset_name = uri.split(".lmdb")[0].split("/")[-1]
        file_name = uri.split()[-1]
        h, w = img.shape[:2]
        img = self.transform(PIL.Image.fromarray(img))
        label = self.charset.filter_unknown(label)

        return {"image": img, "origin_shape": (h, w), "text": label, "dataset": dataset_name, "filename": file_name}

    def __len__(self):
        return self.total_length
