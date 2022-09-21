"""
This file includes the main libraries for training with reloc probability as target.
"""

import faulthandler
import logging
from typing import List, Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from niloc.models.model_factory import get_model
from niloc.network.optim_utils import get_optimizer, get_scheduler

# improve multiprocess debugging
faulthandler.enable(all_threads=True)


class MyLightningModule(pl.LightningModule):
    """
    Base class for all my Lightning Modules. Should be customized by overriding.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.save_hyperparameters(cfg)
        self.network = get_model(
            self.hparams.arch.name,
            self.hparams,
            self.hparams.network.input_dim,
            self.hparams.network.output_dim,
        )
        self.sample = self.hparams.arch.get('sample', 1)
        self.zero = self.hparams.train_cfg.get("with_zero", 0)  # 1 if include zero

    def configure_optimizers(self) -> Dict[str, Any]:
        # Return optimizer, scheduler, [metric to monitor]

        optimizer_cls, optimizer_args = get_optimizer(self.hparams)
        optimizer = optimizer_cls(
            self.network.parameters(),
            self.hparams.train_cfg.lr,
            **optimizer_args,
        )
        optimizer_info = {"optimizer": optimizer}

        if self.hparams.train_cfg.get("scheduler", False):
            scheduler_cls, scheduler_args = get_scheduler(self.hparams)
            scheduler = scheduler_cls(optimizer, **scheduler_args)
            optimizer_info["lr_scheduler"] = scheduler
            if self.hparams.train_cfg.scheduler.get("monitor", False):
                optimizer_info["monitor"] = self.hparams.train_cfg.scheduler.monitor
        else:
            logging.warn("No learning rate scheduler configured.")

        return optimizer_info

    @staticmethod
    def torch_to_numpy(torch_arr: torch.Tensor) -> np.ndarray:
        return torch_arr.cpu().detach().numpy()


class MemoryModule(MyLightningModule):
    """
    Base model for all models that use position priors.
    Specify:
        - loading position from gt
        - sampling/shifting memory according to config
    """

    def __init__(self, cfg: DictConfig) -> None:
        super(MemoryModule, self).__init__(cfg)
        self.zero = self.hparams.train_cfg.get("with_zero", 1)  # 1 if include zero (usually true with memory)

    def get_memory_from_gt(self, m: torch.Tensor) -> torch.Tensor:
        """
        :param m: [batch_size, seq_length] or [batch_size]
        :return:  [batch_size, feat, seq_length]
        """
        with torch.no_grad():
            if m.dim() < 2:
                m = m.reshape(-1, 1)
            m = m.reshape(m.size(0), 1, -1)  # batch, feat, seq_index
            memory = torch.zeros(m.size(0), self.hparams.grid.elements, m.size(-1), device=self.device).scatter(1, m, 1)
        return memory

    def shift_and_sample_memory(self, m: torch.Tensor, include_zero: int = 0):
        """
        :param
            m: [batch_size, feat, seq_length]
            include_zero: 1 (True) if data need not be shifted
        :return:  [batch_size, feat, seq_length]
        """
        with torch.no_grad():
            if self.sample > 1 and include_zero:
                m = m[:, :, ::self.sample]
            else:
                m = m[:, :, self.sample - 1::self.sample]
            if include_zero:
                m = m[:, :, :-1]
            else:
                m[:, :, 1:] = m[:, :, :-1]
                m[:, :, 0] = 1 / m.size(1)
        return m

    def shift_and_sample_from_gt(self, m: torch.Tensor, include_zero: int = 0):
        """
        :param
            :param m: [batch_size, seq_length] or [batch_size]
            include_zero: 1 (True) if data need not be shifted
        :return:  [batch_size, feat, seq_length]
        """
        with torch.no_grad():
            if m.dim() < 2:
                m = m.reshape(-1, 1)
            m = m.reshape(m.size(0), 1, -1)  # batch, feat, seq_index
            if self.sample > 1 and include_zero:
                m = m[:, :, ::self.sample]
            else:
                m = m[:, :, self.sample - 1::self.sample]

            m = torch.zeros(m.size(0), self.hparams.grid.elements, m.size(-1), device=self.device).scatter(1, m, 1)

            if include_zero:
                m = m[:, :, :-1]
            else:
                m[:, :, 1:] = m[:, :, :-1]
                m[:, :, 0] = 1 / m.size(1)
        return m


class ScheduledSamplingModule(MemoryModule):
    """
    Base model for all models that use scheduled sampling.
    Specify:
        - managing tr_ratio. (only applicable with scheduled sampling)
        - saving/loading from checkpoints
    """

    def __init__(self, cfg: DictConfig) -> None:
        super(ScheduledSamplingModule, self).__init__(cfg)

        self.val_ratio = self.hparams.train_cfg.get("val_ratio", 0)     # tr_ratio for validation
        self.tr_ratio = self.hparams.train_cfg.tr_ratio
        print(f"Teacher ratio {self.tr_ratio}")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint['tr_ratio'] = self.tr_ratio
        super(ScheduledSamplingModule, self).on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        super(ScheduledSamplingModule, self).on_load_checkpoint(checkpoint)
        if self.hparams.train_cfg.get("restore_tr_ratio", True):
            self.tr_ratio = checkpoint.get("tr_ratio", self.hparams.train_cfg.tr_ratio)
            logging.info(f"Restored tr_ratio {self.tr_ratio}")

    def training_epoch_end(self, outputs: List[Any]):
        self.log("tr_ratio", self.tr_ratio)

        if self.current_epoch > self.hparams.train_cfg.get('tr_warmup', 0) and self.current_epoch % \
                self.hparams.train_cfg.arre == 0:
            prev_tr = self.tr_ratio
            self.tr_ratio = max(0, self.tr_ratio - self.hparams.train_cfg.arrf)
            print(f"Changing teacher ratio {prev_tr}-->{self.tr_ratio}")
        super(ScheduledSamplingModule, self).on_train_epoch_end(outputs)
