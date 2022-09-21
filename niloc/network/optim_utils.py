import math
from typing import Tuple, Any, Dict, Type, Union

import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities import AttributeDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


class WarmupCosineSchedule(LambdaLR):
    """
    Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    Source: Hugging Face https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/optimization.html
    """

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int,
            t_total: int,
            cycles: float = 0.5,
            last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer,
            [self.lr_lambda],
            last_epoch=last_epoch,
        )

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )


class WarmupConstantSchedule(LambdaLR):
    """
    Linear warmup and then constant.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Source: Hugging Face https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/optimization.html
    """

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int,
            last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(
            optimizer,
            [self.lr_lambda],
            last_epoch=last_epoch,
        )

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.0


class WarmupCosineWithHardRestartsSchedule(LambdaLR):
    """
    Linear warmup and then cosine cycles with hard restarts.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
    learning rate (with hard restarts).
    Source: Hugging Face https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/optimization.html
    """

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int,
            t_total: int,
            cycles: float = 1.0,
            last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(
            optimizer,
            self.lr_lambda,
            last_epoch=last_epoch,
        )

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        if progress >= 1.0:
            return 0.0
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0))),
        )


class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    """
    Adopted from https://raw.githubusercontent.com/ildoonet/pytorch-gradual-warmup-lr/master/warmup_scheduler
    /scheduler.py

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0,
        lr starts from 0 and ends up with the base_lr.
        warmup_epochs: target learning rate is reached at warmup_epoch, gradually
        **kwargs: arguments for ReduceonPlateu scheduler
    """

    def __init__(self, optimizer, multiplier, warmup_epochs, last_epoch=-1, **kwargs):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        super(WarmupReduceLROnPlateau, self).__init__(optimizer, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.finished = False

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))

        # initialize to 0th of warmup
        if self.multiplier == 1.0:
            warmup_lr = [base_lr / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            warmup_lr = [base_lr * ((self.multiplier - 1.) / self.warmup_epochs + 1.) for \
                         base_lr in self.base_lrs]
        for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
            param_group['lr'] = lr

    # @override
    def step(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        print(epoch)
        if epoch < self.warmup_epochs:
            self.last_epoch = epoch  # parent class does a similar process in else
            if self.multiplier == 1.0:
                warmup_lr = [base_lr * ((self.last_epoch + 1.) / self.warmup_epochs) for base_lr in self.base_lrs]
            else:
                warmup_lr = [base_lr * ((self.multiplier - 1.) * (self.last_epoch + 1.) / self.warmup_epochs + 1.) for \
                             base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            super(WarmupReduceLROnPlateau, self).step(metrics, None)


_optimizers = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
}

_schedulers = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "WarmupCosineWithHardRestartsSchedule": WarmupCosineWithHardRestartsSchedule,
    "WarmupCosineSchedule": WarmupCosineSchedule,
    "WarmupConstant": WarmupConstantSchedule,
    "WarmupReduceLROnPlateau": WarmupReduceLROnPlateau,
}


def get_optimizer(cfg: Union[DictConfig, AttributeDict]) -> Tuple[Type[Optimizer], Dict[str, Any]]:
    """
    Get optimizer class and the arguments from cfg.train_cfg.optimizer.
    To add new optimizers,
        1) create `name`.yaml file in config/train_cfg.optimizer.
            - requires `name`
            - (optional) any input parameters as key-value pairs
        2) add 'name`: NewClass to _optimizers
        3) specify in input as train_cfg/optimizer=`name`
    """
    optimizer_cfg = {}
    for key in cfg.train_cfg.optimizer:
        if key == "name":
            continue
        optimizer_cfg[key] = cfg.train_cfg.optimizer[key]

    return _optimizers[cfg.train_cfg.optimizer.name], optimizer_cfg


def get_scheduler(cfg: Union[DictConfig, AttributeDict]) -> Tuple[Type, Dict[str, Any]]:
    """
    Get scheduler class and the arguments from cfg.train_cfg.optimizer.
    To add new optimizers,
        1) create `name`.yaml file in config/train_cfg.scheduler.
            - requires `name`, monitor: bool (True, if scheduler depends on validation loss)
            - (optional) any input parameters as key-value pairs
            - add `t_total=True` if scheduler requires #epochs as input
        2) add 'name`: NewClass to _scheduler
        3) specify in input as train_cfg/scheduler=`name`
    """
    scheduler_cfg = {}
    for key in cfg.train_cfg.scheduler:
        if key in ["name", "monitor"]:
            continue
        if key == "t_total" and cfg.train_cfg.scheduler[key]:
            scheduler_cfg[key] = cfg.train_cfg.epochs
            continue
        scheduler_cfg[key] = cfg.train_cfg.scheduler[key]
    return _schedulers[cfg.train_cfg.scheduler.name], scheduler_cfg
