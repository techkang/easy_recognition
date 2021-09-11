import logging
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from PIL import Image, ImageFont, ImageDraw
from torch.utils import tensorboard
from tqdm import tqdm

import tools.dist as du
from data.dataloader import build_dataloader
from data.transform import denormalize
from lr_scheduler.build_lr_scheduler import build_lr_scheduler
from model import build_model
from tools.checkpointer import Checkpointer
from tools.comm import data_to


class Trainer:
    def __init__(self, cfg, resume):
        self.loss_thresh = 1000
        self.cfg = cfg
        self.device = t.device(cfg.device)
        self.model = build_model(cfg)
        self.output_dir = Path(cfg.output_dir)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.train_loader = None
        self.eval_loader = build_dataloader(cfg, "eval")
        self.test_loader = None
        self.scheduler = build_lr_scheduler(cfg, self.optimizer)
        if cfg.log_name:
            self.log_folder = cfg.log_name
        else:
            self.log_folder = f"{self.cfg.model.name}"
        self.check_pointer = Checkpointer(
            self.model,
            self.output_dir / "checkpoint" / self.log_folder,
            optimizer=self.optimizer,
        )
        self.writer = None
        self.start_iter = self.iter = self.resume_or_load(resume)
        if cfg.start_iter != -1:
            self.iter = self.start_iter = cfg.start_iter
        self.scheduler.last_epoch = self.iter
        self.max_iter = cfg.solver.max_iter
        self.score = None

        self.model.train()

        self._data_loader_iter = None
        self.all_metrics_list = []
        self.tqdm = None

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.

        Otherwise, load a model specified by the config.

        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there'strings no checkpoint).
        return (
                self.check_pointer.resume_or_load(self.cfg.model.weights, resume=resume).get(
                    "iteration", 0
                )
                + 1
        )

    def build_optimizer(self, cfg, net):
        optimizer = getattr(t.optim, cfg.optimizer.name)
        return optimizer(net.parameters(), cfg.lr_scheduler.base_lr)

    def _detect_anomaly(self, losses, info=""):
        if not t.isfinite(losses).all():
            error_info = f"Loss became infinite or NaN at iteration={self.iter}!\nlosses = {losses}"
            if info:
                error_info += " info=" + info
            raise FloatingPointError(error_info)

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {k: v.detach().cpu() if isinstance(v, t.Tensor) else float(v) for k, v in metrics_dict.items()}
        self.all_metrics_list.append(metrics_dict)

    def train(self):
        logging.info("Starting training from iteration {}".format(self.start_iter))

        self.before_train()
        for self.iter in range(self.start_iter, self.max_iter):
            self.before_step()
            self.run_step()
            self.after_step()
        self.after_train()

    def _init_writer(self):
        if self.cfg.tensorboard.enable and du.is_master_proc():
            if self.cfg.log_name:
                folder = self.output_dir / "tbfile" / self.log_folder
            elif self.cfg.tensorboard.name:
                folder = self.output_dir / "tbfile" / self.cfg.tensorboard.name
            else:
                folder = self.output_dir / "tbfile" / self.log_folder
            folder.mkdir(parents=True, exist_ok=True)
            if self.start_iter <= 1:
                shutil.rmtree(folder, ignore_errors=True)
            self.writer = tensorboard.SummaryWriter(folder)
            plt.switch_backend("agg")

    def before_train(self):
        # prepare for tensorboard
        self.train_loader = build_dataloader(self.cfg, "train")
        if du.is_master_proc():
            self._init_writer()

            total_num = sum(p.numel() for p in self.model.parameters())
            trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logging.info(f"total parameters:{total_num:.4e}, trainable parameters: {trainable_num:.4e}")
            model = self.model if self.cfg.num_gpus == 1 else self.model.module
            detail = {k: f"{sum(p.numel() for p in v.parameters()):.3e}" for k, v in model.named_children()}
            logging.info(f"parameters count of sub module: {detail}")

            self.tqdm = tqdm(total=self.cfg.solver.max_iter, initial=self.start_iter, disable=self.cfg.tqdm_disable)
        else:
            self.tqdm = tqdm(disable=True)

        logging.info("build iter")
        self._data_loader_iter = iter(self.train_loader)
        logging.info("begin to train model.")

    def before_step(self):
        if not self.iter % self.cfg.solver.eval_interval:
            self.eval()
            self.test()

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        batch = next(self._data_loader_iter)
        batch = data_to(batch, self.device)
        _unused_data_time = time.perf_counter() - start

        with t.cuda.amp.autocast(enabled=self.cfg.solver.amp):
            loss_out = self.model(batch)
            losses = sum(loss_out.values())

            self.optimizer.zero_grad()
            self._detect_anomaly(losses)
            losses.backward()
        self.optimizer.step()

        postfix_dict = {k: f"{v.detach().cpu():.5e}" for k, v in loss_out.items()}
        self.tqdm.set_postfix(postfix_dict)

        metrics_dict = {"lr": self.scheduler.get_last_lr()[0]}
        metrics_dict.update(loss_out)
        self._write_metrics(metrics_dict)

        self.scheduler.step(None)
        self.tqdm.update()
        if not (self.iter + 1) % 100:
            logging.info(f"time: {datetime.now()}, train iter: {self.iter + 1}, {postfix_dict}")

    def after_step(self):
        if du.is_master_proc() and not self.iter % self.cfg.tensorboard.save_freq and self.writer:
            if "data_time" in self.all_metrics_list[0]:
                data_time = np.max([x.pop("data_time") for x in self.all_metrics_list])
                self.writer.add_scalar("data_time", data_time, self.iter)

            # average the rest metrics
            metrics_dict = {k: np.mean([x[k] for x in self.all_metrics_list]) for k in self.all_metrics_list[0].keys()}

            for k, v in metrics_dict.items():
                self.writer.add_scalar("Train/" + k, v, self.iter)
        if not self.iter % self.cfg.tensorboard.save_freq:
            self.all_metrics_list = []
        if du.is_master_proc() and not (self.iter + 1) % self.cfg.solver.save_interval:
            self.check_pointer.save(f"{self.cfg.model.name}_{self.iter + 1}", iteration=self.iter)

    def collect_batch(self, pred_label_str_dict):
        acc = defaultdict(int)
        total = defaultdict(int)
        ignore_acc = defaultdict(int)
        for each in pred_label_str_dict:
            dataset, pred, label = each["dataset"], each["pred"], each["label"]
            for i in range(len(dataset)):
                acc[dataset[i]] += int(pred[i] == label[i])
                ignore_acc[dataset[i]] += int(pred[i].lower() == label[i].lower())
                total[dataset[i]] += 1
        average_precision = sum(ignore_acc.values()) / sum(total.values())
        return {"score": average_precision, **{"word/" + k: acc[k] / total[k] for k in acc},
                **{"ignore_case/" + k: ignore_acc[k] / total[k] for k in acc}}

    def save_best(self, score):
        if du.is_master_proc() and not self.cfg.eval_only:
            if self.score is None:
                self.score = score
            elif score > self.score:
                self.score = score
                logging.info(f"Find best model so far with max accuracy: {score:.3f} at iter: {self.iter}")
                self.check_pointer.save(f"{self.cfg.model.name}_max_accuracy", iteration=self.iter)

    @t.no_grad()
    def eval_core(self, data_loader):
        self.model.eval()

        all_result = []
        vis_count = 0
        if self.cfg.result_path:
            with open(self.cfg.result_path, "w") as f:
                f.write("")
        for batch in tqdm(data_loader, disable=self.iter != self.start_iter):
            batch = data_to(batch, self.device)
            eval_result = self.model(batch)
            if self.writer:
                vis_count = self.visualize(batch, eval_result, vis_count=vis_count)
            all_result.append(eval_result)
            self.save_result(batch, eval_result)
            # interface for visualize
        return all_result

    def save_result(self, batch, eval_result):
        if self.cfg.result_path:
            with open(self.cfg.result_path, "a") as f:
                for filename, pred, label in zip(batch["filename"], eval_result["pred"], eval_result["label"]):
                    f.write(f"pred: {pred}\nlabel: {label}\n")

    def eval(self, name="Eval"):
        # There is some problem when gathering results across gpus, so eval is only performed on main process.
        # For accurate measure, please use only 1 GPU.
        if du.is_master_proc():
            logging.info(f"iter {self.iter}: " + name.lower() + "ing...")
            if self.writer is None:
                self._init_writer()
            all_result = self.eval_core(self.eval_loader)
            analysed_info = self.collect_batch(all_result)
            loss_dict = {k: np.mean([i["loss"][k].cpu().numpy() for i in all_result]) for k in all_result[0]["loss"]}
            analysed_info.update(loss_dict)

            if name.lower() == "eval":
                self.save_best(analysed_info["score"])
            info_str = ""
            for j, (k, v) in enumerate(analysed_info.items()):
                if isinstance(v, dict):
                    format_v = {key: f"{v[key]:.5f}" for key in sorted(list(v.keys()))}
                else:
                    format_v = f"{v:.5f}"
                info_str += f"{k}:\t {format_v}\t\t"
            logging.info(info_str)

            if self.writer:
                for k, v in analysed_info.items():
                    if isinstance(v, dict):
                        writer_dict = {str(key): v[key] for key in sorted(list(v.keys()))}
                        self.writer.add_scalars(f"{name}/{k}", writer_dict, self.iter)
                    else:
                        self.writer.add_scalar(f"{name}/{k}", v, self.iter)
        self.model.train()

    def visualize(self, batch, eval_result, vis_count=0):
        for i in range(len(batch["image"])):
            if self.cfg.tensorboard.failed_only and eval_result["pred"][i].lower() == eval_result["label"][i].lower():
                continue
            vis_count += 1
            if vis_count > self.cfg.tensorboard.image_num:
                break

            text = eval_result["text"]

            image = denormalize(batch["image"][i])
            origin_h = image.shape[0]
            if image.shape[2] == 1:
                image = np.concatenate([image] * 3, -1)
            gray_shape = image.shape[0] * len(text) // 2, image.shape[1], image.shape[2]
            # using gray instead of white as background color
            image = np.ascontiguousarray(np.concatenate([image, np.ones(gray_shape, image.dtype) * 172]))
            image_pil = Image.fromarray(image)
            font = ImageFont.truetype("data/kaiti.ttf", size=15)
            draw = ImageDraw.Draw(image_pil)
            for j, line in enumerate(text):
                color = tuple(t.tensor([255, 0, 0]).roll(j).tolist())
                draw.text((0, origin_h + origin_h * j // 2), line[i], font=font, fill=color)
            image = np.asarray(image_pil)

            self.writer.add_image(f"eval/{vis_count}", image, self.iter, dataformats="HWC")
        return vis_count

    @t.no_grad()
    def test(self):
        if self.writer is None:
            self._init_writer()
        if not self.cfg.dataset.test:
            return
        if self.test_loader is None:
            self.test_loader = build_dataloader(self.cfg, "test")
        temp_loader, self.eval_loader = self.eval_loader, self.test_loader
        self.eval("Test")
        self.eval_loader = temp_loader

    def after_train(self):
        self.iter += 1
        if self.tqdm is not None:
            self.tqdm.disable = True
        self.after_step()
        if not self.iter % self.cfg.solver.eval_interval:
            self.eval()
        if self.writer is not None:
            self.writer.close()
        logging.info("\ntrain finished.")
