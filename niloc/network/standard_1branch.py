"""
This file includes the main libraries for training with reloc probability as target.
"""

import logging
import time
from typing import List, Dict, Any

import numpy as np
import torch
from omegaconf import DictConfig

from niloc.network.base_models import MyLightningModule


class Standard1branchModule(MyLightningModule):
    """
    Predict position from velocity only.
    Compatible Models: transformer_encoder
    """

    def __init__(self, cfg: DictConfig) -> None:
        super(Standard1branchModule, self).__init__(cfg)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        :param feat : [batch, input_dim , sequence_length]
        :return output: output [batch, output_dim , sequence_length]
        """
        output = self.network(feat)
        if len(output.shape) > 2:
            output = output.permute(0, 2, 1)
        return output

    def training_step(
            self, train_batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        feat, targ, _, _ = train_batch
        pred = self(feat)
        if self.sample > 1:
            targ = targ[:, self.sample - 1::self.sample]
        loss = self.loss_func(pred, targ)
        loss = torch.mean(loss)
        self.log("train_loss", loss, on_epoch=True)
        return {
            "loss": loss,
            "train_loss": loss,
        }

    def validation_step(
            self, val_batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        feat, targ, _, _ = val_batch
        pred = self(feat)
        if self.sample > 1:
            targ = targ[:, self.sample - 1::self.sample]
        loss = self.loss_func(pred, targ)
        loss = torch.mean(loss)
        self.log("val_loss", loss, on_epoch=True)
        return {
            "val_loss": loss,
        }

    @torch.no_grad()
    def get_inference(self, data_loader: torch.utils.data.DataLoader,
                      cfg: DictConfig) -> List[Dict[str, np.ndarray]]:
        """
        Obtain attributes from a data loader given a network state
        Outputs all targets, predicts, softmax of prediction, and losses in numpy arrays
        Enumerates the whole data loader
        """
        targets_all, preds_softmax_all, losses_all, frame_ids_all = [], [], [], []
        loss_func = torch.nn.CrossEntropyLoss(reduction="none")
        t_start = time.time()

        for feat, targ, _, frame_id in data_loader:
            pred = self(feat.to(self.device))
            pred_softmax = torch.nn.functional.softmax(pred, 1).permute(0, 2, 1)
            if self.sample > 1:
                targ = targ[:, self.sample - 1::self.sample]
            loss = loss_func(pred, targ.to(self.device))

            targets_all.append(self.torch_to_numpy(targ))
            preds_softmax_all.append(self.torch_to_numpy(pred_softmax))
            losses_all.append(self.torch_to_numpy(loss))
            frame_ids_all.extend(self.torch_to_numpy(frame_id))

        targets_all = np.concatenate(targets_all, axis=0)  # [b, s, f]
        preds_softmax_all = np.concatenate(preds_softmax_all, axis=0)  # [b, s, f]
        losses_all = np.concatenate(losses_all, axis=0)  # [b, s, f]
        frame_ids_all = np.asarray(frame_ids_all)  # [b]
        attr_dict = {
            "targets": targets_all,
            "preds_softmax": preds_softmax_all,
            "losses": losses_all,
            "frame_ids": frame_ids_all,
            "time": time.time() - t_start,
        }
        logging.info(
            f"Inference done: {targets_all.shape}, {preds_softmax_all.shape}, {losses_all.shape}"
        )
        return [attr_dict]
