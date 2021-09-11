import pickle

import lmdb


class LmdbLoader:
    def __init__(self, max_reader=4):
        self.loaders = []
        self.origin_size = []
        self.count_size = []
        self.dataset_names = []
        self.max_read = max_reader

    def load_index(self, dataset, repeat):
        txn = lmdb.open(dataset, readonly=True, lock=False, readahead=False, meminit=False).begin(write=False)
        self.loaders.append(txn)
        size = int(txn.get(b'total_number').decode("utf8"))
        self.origin_size.append(size)
        self.count_size.append(int(size * repeat))
        dataset_name = dataset.split(".lmdb")[0].split("/")[-1]
        self.dataset_names.append(dataset_name)

    def load(self, path: str):
        if path.endswith(".cfg"):
            with open(path) as f:
                for each_line in f:
                    self.load(each_line)
            return
        line = path.strip()
        if line:
            if line.startswith("data1") and "=" in line:
                raise ValueError('find "data1 = " in cfg file, please delete the "data1 = " in the cfg file to '
                                 'adapt to new lmdb loader!')
            if len(line.split(',')) > 1:
                dataset = ",".join(line.split(',')[:-1])
                repeat = float(line.split(',')[-1])
            else:
                dataset = line
                repeat = 1
            self.load_index(dataset, repeat)

    def read(self, index):
        origin_index = index
        for i in range(len(self.dataset_names)):
            if index < 0:
                raise ValueError(f"can not load data from index: {origin_index}")
            if index >= self.count_size[i]:
                index -= self.count_size[i]
            else:
                index = index % self.origin_size[i]
                image_bytes, label, file_name = pickle.loads(self.loaders[i].get(str(index).encode("utf8")))
                return image_bytes, label, file_name, self.dataset_names[i]

    def __len__(self):
        return sum(self.count_size)

    def count(self):
        return len(self)


class NlpLoader(LmdbLoader):
    def read(self, index):
        origin_index = index
        repeats = self.repeats[:]
        for i in range(len(self._count)):
            while repeats[i]:
                if index < 0:
                    raise ValueError(f"can not load data from index: {origin_index}")
                if index >= self._count[i]:
                    index -= self._count[i]
                    repeats[i] -= 1
                else:
                    label = self.loaders[i].get(str(index).encode("utf8")).decode("utf8")
                    return label, self.dataset_names[i]


def get_loader(name):
    if name == "caffe_loader":
        return DataLoader()
    elif name == "lmdb_loader":
        return LmdbLoader()
    elif name == "nlp_loader":
        return NlpLoader()
    else:
        raise ValueError
