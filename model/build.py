import logging

import torch as t

import model.recognizer


def build_model(cfg):
    net = getattr(model.recognizer, cfg.model.name)(cfg)
    logging.info("finish initialing model.")
    device = t.device(cfg.device)
    net.to(device)
    logging.info(f"load model to {device}")
    if cfg.device == "cuda" and cfg.num_gpus > 1:
        logging.info("using DistributedDataParallel")
        cur_device = t.cuda.current_device()
        net = t.nn.parallel.DistributedDataParallel(net, device_ids=[cur_device], output_device=cur_device)
    logging.info(f"model: {cfg.model.name} loaded.")
    return net
