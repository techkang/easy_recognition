from pathlib import Path

import cv2
import numpy as np
import scipy.io
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class SynthTextCropper(Dataset):
    def __init__(self):
        super().__init__()
        self.base_path = base_path = Path("dataset/SynthText")
        dest = "split"
        self.dest_dir = Path(base_path, dest)
        (base_path / dest).mkdir(exist_ok=True)
        self.data = scipy.io.loadmat(str(base_path / "SynthText/gt.mat"))

    def __len__(self):
        return len(self.data['txt'][0])

    def __getitem__(self, i):
        v = []
        for val in self.data['txt'][0][i]:  # 去掉换行符号和空格
            for x in val.strip().split(" "):
                v.extend(x.split("\n"))
        v = [i for i in v if i]
        rec = np.array(self.data['wordBB'][0][i], dtype=np.int32)
        if len(rec.shape) == 3:
            rec = rec.transpose([2, 1, 0])
        else:
            rec = rec.transpose([1, 0])[np.newaxis, :]
        img_name = Path(str(self.data["imnames"][0][i][0]))
        img = cv2.imread(str(self.base_path / "SynthText" / img_name))
        img_h, img_w = img.shape[:2]
        if not (self.dest_dir / img_name.parent).is_dir():
            (self.dest_dir / img_name.parent).mkdir(exist_ok=True)
        res_str = []
        for j in range(len(rec)):
            word_bbox = rec[j]
            min_x, max_x = int(min(word_bbox[:, 0])), int(max(word_bbox[:, 0]))
            min_y, max_y = int(min(word_bbox[:, 1])), int(max(word_bbox[:, 1]))
            cropped_img = img[min_y:max_y, min_x:max_x]
            crop_h, crop_w = cropped_img.shape[:2]
            if not (img_h > crop_h > 4 and img_w > crop_w > 4):
                continue

            if crop_h > crop_w * 5 or crop_w > crop_h * 15:
                continue

            crop_img_name = f"{str(img_name.parent)}/{img_name.stem}_{str(j)}.jpg"
            cv2.imwrite(str(self.dest_dir / crop_img_name), cropped_img)

            res_str.append(f"{str(crop_img_name)} {v[j]}\n")
        return "".join(res_str)


if __name__ == '__main__':
    dataset = SynthTextCropper()
    dataloader = DataLoader(dataset, 4, num_workers=10, shuffle=False)
    res = []
    for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        res.extend(data)
    with open("dataset/SynthText/label.txt", "w") as f:
        f.write("".join(res))
