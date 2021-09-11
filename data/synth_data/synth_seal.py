# coding=utf-8
import json
import random
import re
import string
from argparse import ArgumentParser
from functools import partial
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
        self.crop = True

        self.charset = args.charset

        self.show_image = True

        self.integral_total = 100000
        source_dir = Path("dataset", "synth_source")
        self.saving_path = Path("dataset", "SynthSeal", args.name)
        self.saving_path.mkdir(exist_ok=True, parents=True)
        self.img_list = [str(i) for i in (source_dir / "bg_img").iterdir() if i.suffix == '.jpg']
        for sub in ("image", "label"):
            (self.saving_path / sub).mkdir(exist_ok=True, parents=True)

        with open(source_dir / "charset" / (args.charset + ".txt"), encoding='utf8') as f:
            self.tokens = np.array([line.strip() for line in f])
        self.text_index = list(range(len(self.tokens)))

        self.random_param = {
            "font_path": [i for i in (source_dir / "font").iterdir() if i.suffix == '.ttf'],
            "font_size": [i for i in range(30, 60)],
            "ellipse_a": [1.] * 50 + [1.01, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            "radius": [i for i in range(40, 80)],
            # if read image here, multi processing will not work!
            "bg_img": [str(i) for i in Path(source_dir / "bg_img").iterdir()],
            "color": [(i, j, k) for i in range(51) for j in range(51) for k in range(200, 256)],
        }
        if self.debug:
            self.random_param = {
                "font_path": [str(i) for i in (source_dir / "font").iterdir() if i.suffix == '.ttf'],
                "font_size": [30],
                "ellipse_a": [2],
                "radius": [100],
                "bg_img": [str(Path(source_dir / "bg_img" / "pink.jpg"))],
                "color": [(0, 0, 255)],
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
        word_set = self.load_word_set()
        self.word_tuple = tuple(word_set) + self.get_specified_word() * int(len(word_set) * 0.1)

        with open(Path("data", "chinese_charset.txt"), encoding="utf8") as f:
            self.all_chinese = tuple(f.readline().strip())

    def get_specified_word(self):
        return "公司", "有限公司", "有限责任公司"

    def load_word_set(self):
        word_file = Path("dataset", "synth_source", "charset", "word_set.txt")
        if not word_file.is_file():
            word_set = self.gen_word_set()
            with open(word_file, "w", encoding="utf8") as f:
                f.write("\n".join(word_set))
        else:
            with open(word_file, encoding="utf8") as f:
                word_set = set(f.read().split("\n"))
        return word_set

    def gen_word_set(self):
        word_set = set()
        char_set = set(self.tokens)
        folder = Path("dataset", "synth_source", "extra_dict")
        for file in folder.iterdir():
            with open(file, encoding="utf8") as f:
                for line in f:
                    word = line.split()[0]
                    for char in word:
                        if char not in char_set:
                            break
                    else:
                        word_set.add(word)
        return word_set

    def gen_text(self):
        # word + char + char + word*n + char
        n = random.randint(1, 4)
        text_max_length = random.randint(8, 15)
        word_list = [random.choice(self.word_tuple) for _ in range(n + 1)]
        char = np.random.choice(self.tokens, 3)
        text = "".join([word_list[0], char[0], char[1], *word_list[1:], char[2]])
        text = text[:text_max_length]
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
        glyf_map = font['glyf']
        in_font = [
            s for s in text if ord(s) in unicode_map and len(glyf_map[unicode_map[ord(s)]].getCoordinates(0)[0]) > 0]
        return "".join(in_font)

    def theta_to_radius(self, theta, a, b):
        return np.sqrt(1 / (np.cos(theta) ** 2 / a ** 2 + np.sin(theta) ** 2 / b ** 2))

    def integral(self, func, start, end):
        xs = np.linspace(start, end, self.integral_total)
        delta = (end - start) / self.integral_total
        points = func(xs)
        delta_arc = delta * points
        int_result = np.cumsum(delta_arc)
        return xs, int_result

    def split_arc_by_theta(self, start, end, a, b, func, num_split):
        xs, ys = self.integral(func, start, end)
        target = np.linspace(ys[0], ys[-1], num_split)
        index = np.searchsorted(ys, target)
        target_thetas = xs[index.clip(0, len(xs) - 1)]
        all_radius = func(target_thetas)
        all_center = np.stack([np.cos(target_thetas), np.sin(target_thetas)]) * all_radius
        slope = np.arctan2(-all_center[0] / a ** 2, -all_center[1] / b ** 2)
        return target_thetas, all_center, slope

    def get_random_string(self, length):
        return "".join(random.choice(self.all_chinese) for _ in range(length))

    def get_random_ascii_string(self, length):
        return "".join(random.choice(string.hexdigits) for _ in range(length))

    def gen_sample(self, config):
        text_size = config["font_size"]
        text = config["text"]
        radius = config["radius"]
        ellipsis_a = config["ellipse_a"] * radius
        ellipsis_b = radius
        font = config["font_path"]
        color = config["color"]

        res_label = []
        start_angle = np.random.randint(-40, 41) if ellipsis_a != radius else np.random.randint(-40, 1)
        start_theta = (180 + start_angle) * np.pi / 180
        end_theta = (360 - start_angle) * np.pi / 180

        font = ImageFont.truetype(font, text_size)
        total_width = font.getsize(text)[0]
        func = partial(self.theta_to_radius, a=ellipsis_a, b=ellipsis_b)

        text = self.filter_text_not_in_font(config["font_path"], text)
        config["text"] = text
        while self.integral(func, start_theta, end_theta)[1][-1] < total_width:
            radius = int(radius * 1.1)
            ellipsis_a = config["ellipse_a"] * radius
            ellipsis_b = radius
            func = partial(self.theta_to_radius, a=ellipsis_a, b=ellipsis_b)

        canvas_shape = (round(ellipsis_b + text_size) * 3, round(ellipsis_a + text_size) * 3, 3)
        canvas = np.zeros(canvas_shape, dtype=np.float32)

        thetas, centers, slopes = self.split_arc_by_theta(start_theta, end_theta, ellipsis_a, ellipsis_b, func,
                                                          len(text))

        ellipsis_center = np.array([[canvas.shape[1] // 2], [canvas.shape[0] // 2]])
        centers += ellipsis_center
        centers = centers.transpose(1, 0)
        centers = np.round(centers).astype(np.int)

        in_lines = []
        out_lines = []

        for char, center, theta, slop in zip(text, centers, thetas, slopes):
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

        bottom_line_prob = 0.8
        bottom_line_angle = 45

        if np.random.rand() < bottom_line_prob or self.debug:
            bottom_line_angle = np.random.randint(max(0, -start_angle + 5), 46)
            _, bottom_centers, _ = self.split_arc_by_theta((180 - bottom_line_angle) * np.pi / 180,
                                                           (360 + bottom_line_angle) * np.pi / 180,
                                                           ellipsis_a, ellipsis_b, func, len(text))
            bottom_centers = (bottom_centers + ellipsis_center).T
            text = self.get_random_string(np.random.randint(4, 8))
            start = bottom_centers[0]
            end = bottom_centers[-1]
            while text and end[0] - start[0] <= font.getsize(text)[0]:
                text = text[:-1]
            start = start + (end - start - np.array([font.getsize(text)[0], 0])) / 2
            img_pil = Image.fromarray(canvas)
            draw = ImageDraw.Draw(img_pil)
            draw.text(start, text, font=font, fill=tuple(color))
            canvas = np.asarray(img_pil)
            width, height = font.getsize(text)
            y, x = start
            polygon = np.array([[y, x], [y + width, x], [y + width, x + height], [y, x + height]]).astype(np.int32)
            res_label.append({"text": text, "polygon": polygon, "main": False})

        top_line_prob = 0.8

        if (np.random.rand() < top_line_prob and start_angle < 0 and bottom_line_angle > 15) or self.debug:
            mid_line_angle = np.random.randint(-15, 10)
            _, mid_centers, _ = self.split_arc_by_theta((180 - mid_line_angle) * np.pi / 180,
                                                        (360 + mid_line_angle) * np.pi / 180,
                                                        ellipsis_a, ellipsis_b, func, len(text))
            mid_centers = (mid_centers + ellipsis_center).T
            text = self.get_random_ascii_string(np.random.randint(4, 18))
            start = mid_centers[0]
            end = mid_centers[-1]
            while text and end[0] - start[0] - 2 * text_size <= font.getsize(text)[0]:
                text = text[:-1]
            start = start + (end - start - np.array([font.getsize(text)[0], 0])) / 2
            img_pil = Image.fromarray(canvas)
            draw = ImageDraw.Draw(img_pil)
            draw.text(start, text, font=font, fill=tuple(color))
            canvas = np.asarray(img_pil)
            width, height = font.getsize(text)
            y, x = start
            polygon = np.array([[y, x], [y + width, x], [y + width, x + height], [y, x + height]]).astype(np.int32)
            res_label.append({"text": text, "polygon": polygon, "main": False})

        cv2.ellipse(canvas, tuple(ellipsis_center.reshape(-1)),
                    (int(ellipsis_a + text_size // 2 + 4), int(ellipsis_b + text_size // 2 + 4)), 0., 0., 360.,
                    tuple(color), 3)

        plotted = np.where(canvas.max(2) > 0)
        x0, y0 = np.min(plotted, axis=1)
        x1, y1 = np.max(plotted, axis=1)
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
        bg_img = cv2.imread(bg_img)
        image_h, image_w, _ = image.shape
        mask = np.expand_dims(np.max(image, axis=2).astype(np.float32) / 255, 2)
        bg_h, bg_w, _ = bg_img.shape
        if bg_h <= image_h or bg_w <= image_w:
            bg_img = np.ones((image_h * 2, image_w * 2, 3), dtype=image.dtype) * 255
            bg_h, bg_w, _ = bg_img.shape
        crop_h, crop_w = np.random.randint(0, bg_h - image_h), np.random.randint(0, bg_w - image_w)
        cropped = bg_img[crop_h:crop_h + image_h, crop_w:crop_w + image_w]
        cropped = cropped * (1 - mask) + image * mask
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
            for i in range(self.total_images):
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
    parser.add_argument("--charset", default="vocab", choices=("vocab", "all_chinese"), help="charset source")
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
