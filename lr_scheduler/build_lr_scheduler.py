from math import cos, pi

import torch as t


def build_lr_scheduler(cfg, optim):
    """
    :param cfg: cfg.lr_scheduler: start_lr, warm_up_end_iter, cosine_start_iter, cosine_end_lr, solver: max_iter
    :param optim:
    :return:
    """
    return WCLR(optim, cfg)


class WCLR(t.optim.lr_scheduler.MultiStepLR):
    def __init__(self, optimizer, cfg):
        self.last_epoch = -1
        self.optimizer = optimizer
        self.start_lr = cfg.lr_scheduler.start_lr * cfg.lr_scheduler.base_lr
        self.base_lr = cfg.lr_scheduler.base_lr
        self.end_lr = cfg.lr_scheduler.end_lr * cfg.lr_scheduler.base_lr
        self.warm_up_end_iter = cfg.lr_scheduler.warm_up_end_iter * cfg.solver.max_iter
        self.cosine_start_iter = cfg.lr_scheduler.cosine_start_iter * cfg.solver.max_iter
        self.cosine_end_iter = cfg.solver.max_iter
        super().__init__(optimizer, [1])

    def get_lr(self):
        if self.last_epoch < self.warm_up_end_iter:
            lr = self.start_lr + (self.base_lr - self.start_lr) * self.last_epoch / self.warm_up_end_iter
        elif self.last_epoch > self.cosine_start_iter:
            lr = (
                self.end_lr
                + (self.base_lr - self.end_lr)
                * (
                    1
                    + cos(
                        (self.last_epoch - self.cosine_start_iter)
                        / (self.cosine_end_iter - self.cosine_start_iter)
                        * pi
                    )
                )
                / 2
            )
        else:
            lr = self.base_lr

        return [lr] * len(self.optimizer.param_groups)
