# coding=utf-8
import json
import random
import re
import string
from argparse import ArgumentParser
from multiprocessing import cpu_count, Pool
from pathlib import Path

import cv2
import imgaug.augmenters as iaa
import lmdb
import msgpack
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont
from skimage import io
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tools.bezier import get_bezier_curve, bezier_prime


def available_cpu_count():
    cpu_info = Path('/proc/self/status')
    res = 0
    if cpu_info.is_file():
        with open(cpu_info) as f:
            m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$', f.read())
        res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
    res = cpu_count() if res == 0 else res
    return res


class FolderDataset(Dataset):
    def __init__(self, name, length):
        self.name = Path("dataset", "SynthSeal", name)
        self.length = length

    def __getitem__(self, item):
        image = io.imread(str(self.name / "image" / f"{str(int(item)).zfill(7)}.jpg"))
        with open(str(self.name / "label" / f"{str(int(item)).zfill(7)}.json"), 'rb') as f:
            config = f.read()
        return image, config

    def __len__(self):
        return self.length


class SynthSeal:
    def __init__(self, args):
        self.name = args.name
        self.total_images = args.length
        self.debug = args.debug
        self.job = args.job
        # 0: no adjust, 1: adjust with findContours, 2: switch adjust point by greedy strategy
        self.polygon_adjust_strategy = 0
        # 0: evenly distribute dst points, 1: set dst points according to their relative distance
        self.seed = args.seed
        self.crop = True

        self.show_image = True

        self.integral_total = 100000
        source_dir = Path("dataset", "synth_source")
        self.saving_path = Path("dataset", "SynthSeal", args.name)
        self.saving_path.mkdir(exist_ok=True, parents=True)
        self.img_list = [str(i) for i in (source_dir / "bg_img").iterdir() if i.suffix == '.jpg']
        for sub in ("image", "label"):
            (self.saving_path / sub).mkdir(exist_ok=True, parents=True)

        self.tokens = []
        frequency = []
        with open(source_dir / "charset" / "all_chinese.txt", encoding="utf8") as f:
            all_chinese = set(f.read().split('\n'))
        with open(source_dir / "charset" / "chinese_frequency.txt", encoding='utf8') as f:
            for line in f:
                res = line.strip().split()
                if len(res) == 3:
                    token, count, _ = res
                    if token in all_chinese:
                        self.tokens.append(token)
                        frequency.append(float(count))
        frequency = np.array(frequency)
        self.frequency = frequency / frequency.sum()

        self.random_param = {
            "font_path": [i for i in (source_dir / "font").iterdir() if i.suffix == '.ttf'],
            "font_size": [i for i in range(30, 60)],
            "radius": [i for i in range(40, 80)],
            # if read image here, multi processing will not work!
            "bg_img": [str(i) for i in Path(source_dir / "bg_img").iterdir()],
            "color": [(i, j, k) for i in range(200, 255) for j in range(200, 255) for k in range(200, 255)],
        }
        if self.debug:
            self.random_param = {
                "font_path": [str(i) for i in (source_dir / "font").iterdir() if i.suffix == '.ttf'],
                "font_size": [30],
                "radius": [100],
                "bg_img": [str(Path(source_dir / "bg_img" / "pink.jpg"))],
                "color": [(255, 255, 255)],
            }
            # load_path = "dataset/SynthSeal/debug/label/0000062.json"
            load_path = ""
            if load_path:
                with open(load_path, encoding='utf8') as f:
                    self.random_param = json.load(f)
                self.random_param.pop('polygon')
                for k, v in self.random_param.items():
                    self.random_param[k] = [v]
        else:
            self.random_param["font_path"] = [str(i) for i in self.random_param["font_path"] if
                                              not i.stem[-2:] in ('61', '65', '68')]

        self.default_items_config = {
            "text": "北京绿城怡景生态环境规划设计股份有限公司",
            "bg_img": [str(Path(source_dir / "bg_img" / "white_background.jpg"))]
        }
        self.default_items_config.update({k: v[0] for k, v in self.random_param.items()})

        self.text_aug_seq = iaa.Sequential(
            [
                iaa.MotionBlur(k=(3, 4)),
                iaa.GammaContrast((0.2, 1.)),
                iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.1)),
                iaa.Sometimes(0.3, [
                    iaa.imgcorruptlike.Snow(severity=(1, 2))
                ])
            ],
            random_order=True
        )
        self.img_aug_seq = iaa.Sequential(
            [
                iaa.GaussianBlur((0, 1.5)),
                iaa.GammaContrast((0.7, 1.3))
            ]
        )

        with open(Path("data", "chinese_charset.txt"), encoding="utf8") as f:
            self.all_chinese = tuple(f.readline().strip())

    def gen_text(self):
        text = ''.join(np.random.choice(self.tokens, size=np.random.randint(30, 64), p=self.frequency))
        return text

    def random_config(self):
        config = {}
        for k, v in self.random_param.items():
            config.update({k: random.choice(v)})
        if not self.debug:
            config["text"] = self.gen_text()
        else:
            config["text"] = "中国科学技术大学" if "text" not in self.random_param else self.random_param["text"][0]
        return config

    def filter_text_not_in_font(self, font_path, text):
        font = TTFont(font_path)
        unicode_map = font['cmap'].tables[0].ttFont.getBestCmap()
        if 'glyf' in font:
            glyf_map = font['glyf']
            in_font = [s for s in text if
                       ord(s) in unicode_map and len(glyf_map[unicode_map[ord(s)]].getCoordinates(0)[0]) > 0]
        else:
            in_font = list(text)
        return "".join(in_font)

    def get_random_string(self, length):
        return "".join(random.choice(self.all_chinese) for _ in range(length))

    def get_random_ascii_string(self, length):
        return "".join(random.choice(string.hexdigits) for _ in range(length))

    def split_points(self, points, num_split):
        distance = np.linalg.norm(points[1:] - points[:-1], axis=1)
        ys = np.cumsum(distance)
        target = np.linspace(ys[0], ys[-1], num_split)
        index = np.clip(np.searchsorted(ys, target), 0, len(points) - 1)
        return index

    def get_center_and_slop(self, text_num, length):

        x = np.linspace(0, length, 4)
        x += [0, np.random.normal(0, length / 20), np.random.normal(0, length / 10), np.random.normal(0, length / 20)]
        y = np.zeros_like(x)
        y += [np.random.normal(0, length / 20), np.random.normal(0, length / 20), np.random.normal(0, length / 8),
              np.random.normal(0, length / 20)]
        control = np.stack([x, y]).T
        curve = get_bezier_curve(control, text_num * 10)
        index = self.split_points(curve, text_num)
        curve = curve[index]
        slope = get_bezier_curve(control, text_num * 10, func=bezier_prime)
        slope = -np.arctan2(slope[:, 1], slope[:, 0])
        slope = slope[index]
        return curve, slope

    def gen_sample(self, config):
        text_size = config["font_size"]
        text = config["text"]
        font = config["font_path"]
        color = config["color"]

        res_label = []

        font = ImageFont.truetype(font, text_size)
        total_width = int(font.getsize(text)[0] * 1.02)

        text = self.filter_text_not_in_font(config["font_path"], text)
        config["text"] = text

        canvas_shape = (total_width * 3, total_width * 3, 3)
        canvas = np.zeros(canvas_shape, dtype=np.float32)

        ellipsis_center = np.array([[canvas.shape[1] // 2], [canvas.shape[0] // 2]])

        in_lines = []
        out_lines = []
        centers, slopes = self.get_center_and_slop(len(text), total_width)
        centers += ellipsis_center.T
        centers = centers.astype(np.int)

        for char, center, slop in zip(text, centers, slopes):
            width, height = font.getsize(char)
            diag = int(round(np.sqrt(width ** 2 + height ** 2)))
            x, y = (diag - width) // 2, (diag - height) // 2
            char_canvas = np.zeros((diag, diag, 3), dtype=np.uint8)

            img_pil = Image.fromarray(char_canvas)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x, y), char, font=font, fill=tuple(color))
            char_draw = np.asarray(img_pil)

            matrix = cv2.getRotationMatrix2D((diag // 2, diag // 2), slop * 180 / np.pi, 1.0)
            char_draw = cv2.warpAffine(char_draw, matrix, (diag, diag))
            char_center = center - diag // 2
            canvas[char_center[1]:char_center[1] + diag, char_center[0]:char_center[0] + diag] += char_draw

            pad = 0
            coordinate = np.array(
                [[x, y, 1], [x + width, y, 1], [x, y + height + pad, 1], [x + width, y + height + pad, 1]]).T

            center = center - (diag / 2, diag / 2)
            polygons = np.round(matrix @ coordinate + center.reshape(-1, 1)).astype(np.int).transpose(1, 0)
            out_lines.extend([polygons[0], polygons[1]])
            in_lines.extend([polygons[2], polygons[3]])

        canvas = np.clip(np.round(canvas), 0, 255).astype(np.uint8)
        in_lines = in_lines[::-1]
        contour = np.array(out_lines + in_lines)
        res_label.append({"text": text, "polygon": contour, "main": True})

        plotted = np.where(canvas.max(2) > 0)
        font_height = font.getsize(text)[1]
        x0, y0 = np.min(plotted, axis=1)
        x1, y1 = np.max(plotted, axis=1) + font_height
        x0 = max(0, x0 - font_height)
        y0 = max(0, y0 - font_height)
        canvas = canvas[x0:x1, y0:y1]

        for instance in res_label:
            instance["polygon"] -= np.array([[y0, x0]])

        return canvas, res_label

    def adjust_polygon(self, image, polygon):
        if self.polygon_adjust_strategy == 1:
            gray = cv2.fillPoly(image.copy(), [polygon], (0, 0, 0))
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            polygon, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            polygon = np.array(sorted(polygon, key=lambda x: len(x), reverse=True)[0])[:, 0]
        elif self.polygon_adjust_strategy == 2:
            polygon = [i for i in polygon]
            sorted_polygon = [polygon.pop(0)]
            while len(polygon) >= 2:
                point1 = polygon.pop(0)
                point2 = polygon.pop(0)
                last = sorted_polygon[-1]
                if np.linalg.norm(point1 - last) < np.linalg.norm(point2 - last):
                    sorted_polygon.append(point1)
                    sorted_polygon.append(point2)
                else:
                    sorted_polygon.append(point2)
                    sorted_polygon.append(point1)
            sorted_polygon.extend(polygon)
            polygon = np.array(sorted_polygon)
        return polygon

    def mix_with_bg(self, bg_img, image, res_label):
        mix = False
        if mix:
            bg_img = cv2.imread(bg_img)
        else:
            bg_img = None
        image_h, image_w, _ = image.shape
        mask = np.expand_dims(np.max(image, axis=2).astype(np.float32) / 255, 2)
        if bg_img is None or bg_img.shape[0] <= image_h or bg_img.shape[1] <= image_w:
            bg_img = np.ones((image_h * 2, image_w * 2, 3), dtype=image.dtype) * np.random.randint(200, 256, (1, 1, 3))
        bg_h, bg_w, _ = bg_img.shape
        crop_h, crop_w = np.random.randint(0, bg_h - image_h), np.random.randint(0, bg_w - image_w)
        cropped = bg_img[crop_h:crop_h + image_h, crop_w:crop_w + image_w]
        cropped = cropped * (1 - mask) + (255 - image) * mask
        cropped = np.clip(np.round(cropped), 0, 255).astype(np.uint8)
        if self.crop:
            bg_img = cropped
        else:
            bg_img[crop_h:crop_h + image_h, crop_w:crop_w + image_w] = cropped
            for instance in res_label:
                instance["polygon"] += np.array([crop_w, crop_h])
        return bg_img

    def aug_text(self, image):
        mask = np.max(image, axis=2) > 0
        image = self.text_aug_seq.augment_image(image)
        canvas = np.zeros_like(image)
        canvas[mask] = image[mask]
        return canvas

    def aug_image(self, image):
        image = self.img_aug_seq.augment_image(image)
        return image

    def visualize(self, image, res_label):
        polygon = [i["polygon"] for i in res_label]
        image = cv2.drawContours(image, polygon, -1, (0, 0, 0), 1)
        cv2.imshow("image with polygon", image)
        cv2.waitKey()

    def image_generator(self, i):
        if self.seed:
            np.random.seed(i)
            random.seed(i)
        config = self.random_config()
        crop_img, res_label = self.gen_sample(config)
        crop_img = self.aug_text(crop_img)

        crop_img = self.mix_with_bg(config["bg_img"], crop_img, res_label)
        crop_img = self.aug_image(crop_img)
        cv2.imwrite(str(self.saving_path / "image" / (str(i).zfill(7) + ".jpg")), crop_img)
        if self.debug:
            self.visualize(crop_img, res_label)
        with open(self.saving_path / "label" / (str(i).zfill(7) + ".json"), "w", encoding='utf8') as f:
            for instance in res_label:
                instance["polygon"] = instance["polygon"].tolist()
            json.dump(res_label, f, ensure_ascii=False)

    def collect(self):
        dataset = FolderDataset(self.name, self.total_images)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.job)
        env = lmdb.open(str(self.saving_path), map_size=1024 ** 2 * 8 * self.total_images // 200)
        tnx = env.begin(write=True)
        count = 0
        for batch in tqdm(dataloader, total=self.total_images):
            for image, config in zip(*batch):
                _, image = cv2.imencode('.jpg', image.numpy())
                config = json.loads(config)
                tnx.put(f"image-{str(count).zfill(7)}".encode("utf8"), image.tobytes())
                tnx.put(f"config-{str(count).zfill(7)}".encode("utf8"), msgpack.dumps(config))
                count += 1
        tnx.put("total_length".encode("utf8"), str(self.total_images).encode("utf8"))
        tnx.commit()
        env.close()

    def process(self):
        if self.debug:
            for i in range(100):
                self.image_generator(i)
            exit(0)

        print(f'using multiprocessing with {self.job} cores.')
        if self.job == 0:
            for i in tqdm(range(self.total_images)):
                self.image_generator(i)
        else:
            with Pool(self.job) as pool:
                for _ in tqdm(pool.imap(self.image_generator, range(self.total_images)), total=self.total_images):
                    pass
        with open(self.saving_path / "count.txt", "w", encoding="utf8") as f:
            f.write(str(self.total_images))


def arg_parse():
    parser = ArgumentParser("synth seal generator")
    parser.add_argument("-n", "--name", default="train", type=str, help="folder to save generated dataset")
    parser.add_argument("-l", "--length", default=10000, type=int, help="num_classes generated images")
    parser.add_argument("-d", "--debug", default=False, action="store_true", help="enable debug mode")
    parser.add_argument("-p", "--process", default=False, action="store_true", help="enable process new image")
    parser.add_argument("-a", "--aug", default=False, action="store_true", help="enable augment")
    parser.add_argument("-c", "--collect", default=False, action="store_true", help="enable collect image to lmdb")
    parser.add_argument("-j", "--job", default=available_cpu_count(), type=int, help="num of cpus for multiprocessing")
    parser.add_argument("--seed", default=1, type=int, help="seed for random, 0 for not fix seed")
    return parser.parse_args()


if __name__ == "__main__":
    arg = arg_parse()
    if arg.seed:
        random.seed(arg.seed)
        np.random.seed(arg.seed)
    synth_generator = SynthSeal(arg)
    if arg.process:
        synth_generator.process()
    if arg.collect:
        synth_generator.collect()
