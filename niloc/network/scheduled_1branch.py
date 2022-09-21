"""
This file includes the main libraries for training with reloc probability as target.
"""

import logging
import time
from typing import List, Dict, Any

import numpy as np
import torch
from omegaconf import DictConfig

from niloc.network.base_models import ScheduledSamplingModule


class Scheduled1branchModule(ScheduledSamplingModule):
    """
    Models that has output from decoder. Trained with parallel scheduled sampling.
    Compatible models : transformer_full
    """

    def __init__(self, cfg: DictConfig) -> None:
        super(Scheduled1branchModule, self).__init__(cfg)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through entire network all memory is known.
        :params:
         - feat: [batch, input_dim , sequence_length]
         - memory: [batch, input_dim , sequence_length]
        :returns:
          - output : decoder output [batch, output_dim , sequence_length]
        """
        output = self.network(x, m)
        if len(output.shape) > 2:
            output = output.permute(0, 2, 1)
        return output

    def forward_enc(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Output from feature only branch
        :param feat: [batch, input_dim , sequence_length]
        :return mid_enc [sequence_length, batch, model_dim]
        """
        mid_enc = self.network.forward_encoder(feat)
        return mid_enc

    def forward_dec(self, enc_output: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Output from position prior branch with feature encoding given.
        :param enc_output: [sequence_length, batch, model_dim]
        :param memory: [batch, input_dim , sequence_length]
        :returns
        - output : decoder output [batch, output_dim , sequence_length]
        """
        out_dec = self.network.forward_decoder(enc_output, memory)
        if len(out_dec.shape) > 2:
            out_dec = out_dec.permute(0, 2, 1)
        return out_dec

    def forward_ss(self, feat: torch.Tensor, memory: torch.Tensor, tr_ratio: float,
                   itr_train: bool = True) -> torch.Tensor:
        """
        Forward with scheduled sampling during training/validation.
        :params:
         - feat: [batch, input_dim , sequence_length]
         - memory: [batch, input_dim , sequence_length]
         - tr_ratio: int
         - itr_train: True, if in training
        :returns:
          - output : decoder output [batch, output_dim , sequence_length]
        """
        enc_out = self.network.forward_encoder(feat)  # [s, b, f]
        s = max(0, int((memory.size(-1) - 1) * (1 - tr_ratio)))
        c = np.random.rand(2) if itr_train else [1, 1]
        if s > 0:
            with torch.no_grad():
                if c[0] > 0.5:
                    memory[:, :, 0] = 1 / memory.size(1)
                pass1 = self.network.forward_decoder(enc_out, memory)
                idx = np.random.choice(np.arange(1, memory.size(-1)), s, replace=False)
                memory[:, :, idx] = torch.nn.functional.softmax(pass1[:, (idx - 1), :], 1).permute(0, 2, 1)
        if c[1] > 0.5:
            memory[:, :, 0] = 1 / memory.size(1)
        output = self.network.forward_decoder(enc_out, memory)
        if len(output.shape) > 2:
            output = output.permute(0, 2, 1)
        return output

    def training_step(
            self, train_batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        feat, targ, _, _ = train_batch

        pred = self.forward_ss(feat[:, :, self.zero:],
                               self.shift_and_sample_memory(self.get_memory_from_gt(targ), self.zero),
                               self.tr_ratio)
        targ = targ[:, self.sample - 1 + self.zero::self.sample]
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
        # always use prediction
        pred = self.forward_ss(feat[:, :, self.zero:],
                               self.shift_and_sample_memory(self.get_memory_from_gt(targ), self.zero),
                               self.val_ratio, False)
        targ = targ[:, self.sample - 1 + self.zero::self.sample]
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
        :param
        - data_loader - Enumerates the whole data loader - ideally data loader should only contain one trajectory
        - cfg: Config. Passed separately, so that model can be loaded from checkpoint and test_cfg is specified here.
        """
        t_start = time.time()
        targets_all, frame_ids_all = [], []
        preds_softmax_all, losses_all, = [], []

        loss_func = torch.nn.CrossEntropyLoss(reduction="none")
        w, s = cfg.data_window_cfg.window_size, cfg.data_window_cfg.step_size
        overlap = int((w - self.zero - s) // self.sample)
        for i, (feat, targ, _, frame_id) in enumerate(data_loader):
            if cfg.test_cfg.get('with_gt', False):
                memory = self.shift_and_sample_memory(
                    self.get_memory_from_gt(targ.to(self.device)),
                    self.zero)
                pred = self(feat[:, :, self.zero:].to(self.device), memory)
                pred_softmax = torch.nn.functional.softmax(pred, 1)
                targ = targ[:, self.sample - 1 + self.zero::self.sample]
            elif cfg.test_cfg.get('individual', False):
                mid_enc = self.forward_enc(feat[:, :, self.zero:].to(self.device))
                targ_gt = targ[:, self.sample - 1 + self.zero::self.sample]
                length = targ_gt.size(1)
                memory = torch.ones(targ.size(0), self.hparams.grid.elements, length,
                                    device=self.device) / self.hparams.grid.elements
                if cfg.test_cfg.get('start_gt', False):
                    memory[:, :, :cfg.test_cfg.start_gt] = self.shift_and_sample_memory(
                        self.get_memory_from_gt(targ.to(self.device)),
                        self.zero)[:, :, :cfg.test_cfg.start_gt]
                for j in range(length):
                    if j > 0:
                        memory[:, :, 1:j + 1] = pred_softmax
                    pred = self.forward_dec(mid_enc, memory[:, :, :j + 1])
                    pred_softmax = torch.nn.functional.softmax(pred, 1)
                targ = targ_gt
            else:
                # propergate in time
                mid_enc = self.forward_enc(feat[:, :, self.zero:].to(self.device))

                targ_gt = targ[:, self.sample - 1 + self.zero::self.sample]
                length = targ_gt.size(1)
                ss = overlap if i > 0 else 0

                memory = torch.ones(targ.size(0), self.hparams.grid.elements, length,
                                    device=self.device) / self.hparams.grid.elements
                if i == 0 and cfg.test_cfg.get('start_gt', False):
                    memory[:, :, :cfg.test_cfg.start_gt] = self.shift_and_sample_memory(
                        self.get_memory_from_gt(targ.to(self.device)),
                        self.zero)[:, :, :cfg.test_cfg.start_gt]
                    ss = cfg.test_cfg.start_gt - 1
                elif i > 0:
                    memory[:, :, :overlap + 1] = pred_softmax[:, :, -overlap - 1:]  # from last iteration

                for j in range(ss, length):
                    if j > ss:
                        memory[:, :, 1:j + 1] = pred_softmax

                    pred = self.forward_dec(mid_enc, memory[:, :, :j + 1])
                    pred_softmax = torch.nn.functional.softmax(pred, 1)
                targ = targ_gt

            pred_softmax_p = pred_softmax.permute(0, 2, 1)  # bsf
            loss = loss_func(pred, targ.to(self.device))

            targets_all.append(self.torch_to_numpy(targ))
            preds_softmax_all.append(self.torch_to_numpy(pred_softmax_p))
            losses_all.append(self.torch_to_numpy(loss))
            frame_ids_all.extend(self.torch_to_numpy(frame_id))

        targets_all = np.concatenate(targets_all, axis=0)  # [b, s, f]
        preds_softmax_all = np.concatenate(preds_softmax_all, axis=0)  # [b, s, f]
        losses_all = np.concatenate(losses_all, axis=0)  # [b, s, f]
        frame_ids_all = np.asarray(frame_ids_all) + self.zero  # [b]
        decoder_attr_dict = {
            "targets": targets_all,
            "preds_softmax": preds_softmax_all,
            "losses": losses_all,
            "frame_ids": frame_ids_all,
            "time": time.time() - t_start,
        }
        logging.info(
            f"Inference done: {targets_all.shape}, {preds_softmax_all.shape}"
            f"{losses_all.shape} {frame_ids_all.shape}"
        )
        return [decoder_attr_dict]
