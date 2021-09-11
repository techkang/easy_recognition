import logging
from pathlib import Path

import numpy as np

import model
from .base_dataset import OCRDataset
from data.lmdb_loader import get_loader


class NLPDataset(OCRDataset):
    def __init__(self, cfg, mode):
        super().__init__(cfg, mode)

        self.dummy_img = np.zeros((1 if cfg.dataset.gray_scale else 3, 32, 256), dtype=np.float32)
        self.add_noise = True
        self.similar_word = self.read_similar("data/similar.txt")

    def read_similar(self, path):
        with open(path) as f:
            data = f.read().strip().split("\n")
        res = {}
        for line in data:
            res[line[0]] = list(line[2:])
        return res

    def __getitem__(self, item):
        if self.loader is None:
            self.loader = get_loader(self.loader_name)
            for name, repeat in zip(self.dataset_names, self.repeat):
                self.loader.load(str(Path(self.base_path, name)), repeat)
        while True:
            text, dataset = self.loader.read(item)
            if len(text) > self.target_length:
                start = np.random.randint(0, len(text) - self.target_length + 1)
                text = text[start:start + self.target_length]
            text = self.str_q2b(text.strip())
            if len(text) > 1:
                break
            item = (item + 19) % len(self)

        text = text[:np.random.randint(5, max(len(text) - 2, 6))]
        if self.add_noise:
            input_text = self.random_noise(text)
        else:
            input_text = text
        input_text = self.charset.filter_unknown(input_text)
        text = self.charset.filter_unknown(text)
        return {"image": self.dummy_img, "input": input_text, "label": text, "dataset": dataset}

    def random_noise(self, text):
        res = []
        prob = 1
        for i, char in enumerate(text):
            prob *= np.random.rand()
            if prob < 1 / 64:
                prob = 1
                if self.similar_word.get(char) and np.random.rand() < 0.5:
                    char = np.random.choice(self.similar_word[char])
                else:
                    char = np.random.choice(self.charset.single_char)
            res.append(char)
        return "".join(res)

    def __len__(self):
        return self.total_length

    def str_q2b(self, ustring):
        """全角转半角"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif inside_code >= 65281 and inside_code <= 65374:  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        return rstring

