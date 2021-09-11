import argparse
import pickle
import shutil
import sys
from pathlib import Path

import lmdb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ImageLoader(Dataset):
    def __init__(self, img_list, base_path):
        self.base_path = base_path
        with open(img_list) as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = self.lines[item].strip()
        file_path = line.split()[0]
        with open(Path(self.base_path, file_path), "rb") as f:
            img_bytes = f.read()

        label = line[len(file_path) + 1:].strip()
        if not label:
            print(f"warning: file {line} comes with no label.", file=sys.stderr)
        data_bytes = pickle.dumps((img_bytes, label, file_path))
        return data_bytes


def lmdb_converter(opt):
    img_list = opt.img_list
    output = opt.output
    base_path = opt.base_path
    batch_size = opt.batch_size
    coding = opt.coding
    size = opt.size
    num_workers = opt.thread

    if not output:
        print("no output folder is provided, using data.lmdb instead.")
        output = "data.lmdb"
    elif not output.endswith(".lmdb"):
        print('warning: output folder name should endswith ".lmdb", '
              'the ".lmdb" is appended to folder name automatically.')
        output += ".lmdb"
    if not base_path:
        base_path = Path(img_list).parent
        print(f'no base_path is given, using absolute path of image list: {str(base_path)} instead.')
    if num_workers > 0:
        print("using multi thread, if any bug occurs, please set --thread=0 and restart the scripts.")

    dataset = ImageLoader(img_list, base_path)
    dataloader = DataLoader(dataset, batch_size, num_workers=10)
    create_lmdb(output, dataloader, size, coding)



def create_lmdb(output, dataloader, size=1, coding="utf8"):
    # create lmdb database
    estimated_size = size * 1099511627776
    env = lmdb.open(output, map_size=estimated_size)

    # build lmdb
    count = 0
    if hasattr(dataloader, "__len__"):
        total = len(dataloader)
    else:
        total = None
    for i, data in tqdm(enumerate(dataloader), total=total):
        batch = [(str(count + j).encode(coding), data_bytes) for j, data_bytes in enumerate(data)]
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            cursor.putmulti(batch, dupdata=False, overwrite=True)
        count += len(batch)
    with env.begin(write=True) as txn:
        key = 'total_number'.encode(coding)
        value = str(count).encode(coding)
        txn.put(key, value)
    print(f'total {count} images are save to lmdb: {output}')
    print('done')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_list', '-i', required=True, help='input imglist path')
    parser.add_argument('--output', '-o', default="", help='output lmdb path')
    parser.add_argument('--base_path', '-b', default="", help='base_path for images')
    parser.add_argument('--size', '-strings', default=1, type=int, help='estimated size (GB)')
    parser.add_argument('--thread', default=10, type=int, help='default threads for ')
    parser.add_argument('--batch_size', type=int, default=1000, help='processing batch size, default 10000')
    parser.add_argument('--coding', '-c', default='utf8', help='bytes coding scheme, default utf8')
    opt = parser.parse_args()

    output = opt.output
    if Path(output).is_dir():
        while True:
            print(f'{output} already exist, delete or not? [Y/n]')
            flag = input().strip()
            if flag in ['Y', 'y']:
                shutil.rmtree(output)
                break
            elif flag in ['N', 'n']:
                return
    print(f'creating database {output}')
    Path(output).mkdir(parents=True, exist_ok=False)

    return opt


if __name__ == '__main__':
    lmdb_converter(get_args())
