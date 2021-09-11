import argparse
from pathlib import Path
import builtins
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch as t
from yacs.config import CfgNode as CN
import yaml

import tools.dist as du

global_cfg = CN()


def setup(args):
    cfg = get_cfg()
    with open(args.config_file) as f:
        temp_cfg = yaml.load(f, Loader=yaml.FullLoader)
    for file in temp_cfg.get("base_cfg", ()):
        cfg.merge_from_file(Path(args.config_file).parent / file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg)
    cfg.freeze()
    return cfg


def default_argument_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume", action="store_true", help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--visible-gpus", type=str, default="")

    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,
    )
    return parser


def init_logging(arg, cfg):
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)
    if du.is_master_proc():
        if cfg.log_name:
            log_folder = cfg.log_name
        else:
            log_folder = f"{cfg.model.name}"
        Path(cfg.output_dir, "checkpoint", log_folder).mkdir(parents=True, exist_ok=True)
        Path(cfg.output_dir, "tbfile", log_folder).mkdir(parents=True, exist_ok=True)
        Path(cfg.output_dir, "logfile").mkdir(exist_ok=True)
        filename = Path(cfg.output_dir, "logfile", f"{log_folder}.log")
        if cfg.eval_only or cfg.test_only:
            logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        else:
            logging.getLogger().handlers.clear()
            logging.basicConfig(filename=str(filename), filemode="w", level=logging.INFO)
            logging.getLogger().addHandler(logging.StreamHandler())
        logging.info(f"Running with full config: \n{cfg}")
        logging.info("Command line arguments: " + str(arg))
    else:

        def print_dummy(*args, **kwargs):
            pass

        builtins.print = print_dummy


def default_setup(cfg):
    if cfg.random_seed >= 0:
        random.seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        t.manual_seed(cfg.random_seed)
        t.cuda.manual_seed_all(cfg.random_seed)
        t.backends.cudnn.deterministic = True

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if cfg.eval_only or cfg.test_only:
        t.backends.cudnn.benchmark = False
    else:
        t.backends.cudnn.benchmark = cfg.cudnn_benchmark


def get_cfg():
    from config.defaults import _C

    return _C.clone()


def set_global_cfg(cfg: CN):
    global global_cfg
    global_cfg.clear()
    global_cfg.update(cfg)
