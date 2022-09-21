"""
This file includes the main libraries for training with reloc probability as target.
"""

import logging
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

from niloc.network.base_models import ScheduledSamplingModule, MemoryModule


class Scheduled2branchModule(ScheduledSamplingModule):
    """
    Models that has output from 2 branches. Trained with parallel scheduled sampling.
    Compatible models : transformer_2step
    """

    def __init__(self, cfg: DictConfig) -> None:
        super(Scheduled2branchModule, self).__init__(cfg)
        self.loss_func = torch.nn.CrossEntropyLoss()
        # weight from encoder only for training loss
        self.encoder_only_weight = self.hparams.train_cfg.get("encoder_only_weight", 0.5)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through entire network all memory is known.
        :params:
         - feat: [batch, input_dim , sequence_length]
         - memory: [batch, input_dim , sequence_length]
        :returns:
          - out_enc, out_dec : encoder & decoder output [batch, output_dim , sequence_length]
        """
        out_enc, out_dec = self.network(x, m)
        if len(out_enc.shape) > 2:
            out_enc = out_enc.permute(0, 2, 1)
            out_dec = out_dec.permute(0, 2, 1)
        return out_enc, out_dec

    def forward_enc(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Output from feature only branch
        :param feat: [batch, input_dim , sequence_length]
        :returns
         - out_enc : encoder output [batch, output_dim , sequence_length]
         - mid_enc : [sequence_length, batch, model_dim]
        """
        out_enc, mid_enc = self.network.forward_encoder(feat)
        if len(out_enc.shape) > 2:
            out_enc = out_enc.permute(0, 2, 1)
        return out_enc, mid_enc

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

    def forward_scheduled(self, feat: torch.Tensor,
                          memory: torch.Tensor,
                          tr_ratio: float,
                          itr_train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with scheduled sampling during training/validation.
        :params:
         - feat: [batch, input_dim , sequence_length]
         - memory: [batch, input_dim , sequence_length]
         - tr_ratio: int
         - itr_train: True, if in training
        :returns:
          - out_enc, out_dec : encoder & decoder output [batch, output_dim , sequence_length]
        """
        out_enc, encoder = self.network.forward_encoder(feat)  # [b, s, f], [s, b, f]

        # get output with gt and mix gt and predicted as decoder input
        s = max(0, int((memory.size(-1) - 1) * (1 - tr_ratio)))
        c = np.random.rand(2) if itr_train else [1, 1]
        if s > 0:
            with torch.no_grad():
                if c[0] > 0.5:
                    memory[:, :, 0] = 1 / memory.size(1)
                pass1 = self.network.forward_decoder(encoder, memory)  # [b, s, f]
                idx = np.random.choice(np.arange(1, memory.size(-1)), s, replace=False)
                memory[:, :, idx] = torch.nn.functional.softmax(pass1[:, (idx - 1), :].permute(0, 2, 1), 1)  # bsf->bfs
        if c[1] > 0.5:
            memory[:, :, 0] = 1 / memory.size(1)
        # actual pass
        out_dec = self.network.forward_decoder(encoder, memory)

        if len(out_enc.shape) > 2:
            out_enc = out_enc.permute(0, 2, 1)
            out_dec = out_dec.permute(0, 2, 1)
        return out_enc, out_dec

    def training_step(
            self, train_batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        feat, targ, _, _ = train_batch
        pred_enc, pred_dec = self.forward_scheduled(feat[:, :, self.zero:],
                                                    self.shift_and_sample_from_gt(targ, self.zero),
                                                    self.tr_ratio)

        targ = targ[:, self.sample - 1 + self.zero::self.sample]

        enc_loss = torch.mean(self.loss_func(pred_enc, targ))
        dec_loss = torch.mean(self.loss_func(pred_dec, targ))

        loss = enc_loss * self.encoder_only_weight + dec_loss * (1 - self.encoder_only_weight)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_enc_loss", enc_loss, on_epoch=True, on_step=True)
        self.log("train_dec_loss", dec_loss, on_epoch=True, on_step=True)
        return {
            "loss": loss,
            "train_loss": loss,
            "train_enc_loss": enc_loss,
            "train_dec_loss": dec_loss,
        }

    def validation_step(
            self, val_batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        feat, targ, _, _ = val_batch
        # use gt with probability self.val_ratio
        pred_enc, pred_dec = self.forward_scheduled(feat[:, :, self.zero:],
                                                    self.shift_and_sample_from_gt(targ, self.zero),
                                                    self.val_ratio, False)
        targ = targ[:, self.sample - 1 + self.zero::self.sample]

        enc_loss = torch.mean(self.loss_func(pred_enc, targ))
        dec_loss = torch.mean(self.loss_func(pred_dec, targ))
        loss = enc_loss * self.encoder_only_weight + dec_loss * (1 - self.encoder_only_weight)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_enc_loss", enc_loss, on_epoch=True)
        self.log("val_dec_loss", dec_loss, on_epoch=True)
        return {
            "val_loss": loss,
            "val_enc_loss": enc_loss,
            "val_dec_loss": dec_loss,
        }

    @torch.no_grad()
    def get_inference(self, data_loader: torch.utils.data.DataLoader,
                      cfg: DictConfig) -> List[Dict[str, np.ndarray]]:
        """
        Obtain attributes from a data loader given a network state
        Outputs all targets, predicts, softmax of prediction, and losses in numpy arrays for encoder and decoder
        :param
        - data_loader - Enumerates the whole data loader - ideally data loader should only contain one trajectory
        - cfg: Config. Passed separately, so that model can be loaded from checkpoint and test_cfg is specified here.
        """
        t_start = time.time()
        targets_all, frame_ids_all = [], []
        preds_enc_softmax_all, losses_enc_all, preds_dec_softmax_all, losses_dec_all = [], [], [], []

        loss_func = torch.nn.CrossEntropyLoss(reduction="none")
        w, s = cfg.data_window_cfg.window_size, cfg.data_window_cfg.step_size
        overlap = int((w - self.zero - s) // self.sample)
        for i, (feat, targ, _, frame_id) in enumerate(data_loader):
            if cfg.test_cfg.get('with_gt', False):
                memory = self.shift_and_sample_from_gt(targ.to(self.device),
                    self.zero)
                pred_enc, pred_dec = self(feat[:, :, self.zero:].to(self.device), memory)
                # print(pred_enc.shape, pred_dec.shape, memory.shape)
                pred_enc_softmax = torch.nn.functional.softmax(pred_enc, 1)  # bfs
                pred_dec_softmax = torch.nn.functional.softmax(pred_dec, 1)
                targ = targ[:, self.sample - 1 + self.zero::self.sample]
            elif cfg.test_cfg.get('individual', False):
                pred_enc, mid_enc = self.forward_enc(feat[:, :, self.zero:].to(self.device))
                pred_enc_softmax = torch.nn.functional.softmax(pred_enc, 1)
                targ_gt = targ[:, self.sample - 1 + self.zero::self.sample]
                length = targ_gt.size(1)
                memory = torch.ones(targ.size(0), self.hparams.grid.elements, length,
                                    device=self.device) / self.hparams.grid.elements
                ss = 0
                if cfg.test_cfg.get('start_gt', False):
                    memory[:, :, :cfg.test_cfg.start_gt] = self.shift_and_sample_from_gt(targ.to(self.device),
                        self.zero)[:, :, :cfg.test_cfg.start_gt]
                    ss = cfg.test_cfg.start_gt - 1
                for j in range(ss, length):
                    if j > ss:
                        # memory[:, :, 1:j + 1] = pred_dec_softmax
                        memory[:, :, j] = pred_dec_softmax[:, :, -1]
                    pred_dec = self.forward_dec(mid_enc, memory[:, :, :j + 1])
                    pred_dec_softmax = torch.nn.functional.softmax(pred_dec, 1)
                targ = targ_gt
            else:
                # propergate in time
                pred_enc, mid_enc = self.forward_enc(feat[:, :, self.zero:].to(self.device))
                pred_enc_softmax = torch.nn.functional.softmax(pred_enc, 1)

                targ_gt = targ[:, self.sample - 1 + self.zero::self.sample]
                length = targ_gt.size(1)
                ss = overlap if i > 0 else 0

                memory = torch.ones(targ.size(0), self.hparams.grid.elements, length,
                                    device=self.device) / self.hparams.grid.elements
                if i == 0 and cfg.test_cfg.get('start_gt', False):
                    memory[:, :, :cfg.test_cfg.start_gt] = self.shift_and_sample_from_gt(targ.to(self.device),
                        self.zero)[:, :, :cfg.test_cfg.start_gt]
                    ss = cfg.test_cfg.start_gt - 1
                elif i > 0:
                    memory[:, :, :overlap + 1] = pred_dec_softmax[:, :, -overlap - 1:]  # from last iteration

                for j in range(ss, length):
                    if j > ss:
                        # memory[:, :, 1:j + 1] = pred_dec_softmax
                        memory[:, :, j] = pred_dec_softmax[:, :, -1]

                    pred_dec = self.forward_dec(mid_enc, memory[:, :, :j + 1])
                    pred_dec_softmax = torch.nn.functional.softmax(pred_dec, 1)
                targ = targ_gt

            pred_dec_softmax_p = pred_dec_softmax.permute(0, 2, 1)  # bsf
            pred_enc_softmax = pred_enc_softmax.permute(0, 2, 1)
            loss_enc = loss_func(pred_enc, targ.to(self.device))
            loss_dec = loss_func(pred_dec, targ.to(self.device))

            targets_all.append(self.torch_to_numpy(targ))
            preds_enc_softmax_all.append(self.torch_to_numpy(pred_enc_softmax))
            preds_dec_softmax_all.append(self.torch_to_numpy(pred_dec_softmax_p))
            losses_enc_all.append(self.torch_to_numpy(loss_enc))
            losses_dec_all.append(self.torch_to_numpy(loss_dec))
            frame_ids_all.extend(self.torch_to_numpy(frame_id))

        targets_all = np.concatenate(targets_all, axis=0)  # [b, s, f]
        preds_enc_softmax_all = np.concatenate(preds_enc_softmax_all, axis=0)  # [b, s, f]
        preds_dec_softmax_all = np.concatenate(preds_dec_softmax_all, axis=0)  # [b, s, f]
        losses_enc_all = np.concatenate(losses_enc_all, axis=0)  # [b, s, f]
        losses_dec_all = np.concatenate(losses_dec_all, axis=0)  # [b, s, f]
        frame_ids_all = np.asarray(frame_ids_all) + self.zero  # [b]
        encoder_attr_dict = {
            "targets": targets_all,
            "preds_softmax": preds_enc_softmax_all,
            "losses": losses_enc_all,
            "frame_ids": frame_ids_all,
            "time": time.time() - t_start,
        }
        decoder_attr_dict = {
            "targets": targets_all,
            "preds_softmax": preds_dec_softmax_all,
            "losses": losses_dec_all,
            "frame_ids": frame_ids_all,
            "time": time.time() - t_start,
        }
        logging.info(
            f"Inference done: {targets_all.shape}, {preds_enc_softmax_all.shape}, {preds_dec_softmax_all.shape}, "
            f"{losses_enc_all.shape} {frame_ids_all.shape}"
        )
        return [encoder_attr_dict, decoder_attr_dict]

    @torch.no_grad()
    def get_inference_minimal(self, data_loader: torch.utils.data.DataLoader,
                              cfg: DictConfig, test_type:str = "dec") -> List[Dict[str, np.ndarray]]:
        """
        Minimal version for benchmarking
        Outputs all targets, predicts, softmax of prediction, and losses in numpy arrays for encoder or decoder
        :param
        - data_loader - Enumerates the whole data loader - ideally data loader should only contain one trajectory
        - cfg: Config. Passed separately, so that model can be loaded from checkpoint and test_cfg is specified here.
        """
        targets_all, frame_ids_all = [], []
        preds_softmax_all = []

        w, s = cfg.data_window_cfg.window_size, cfg.data_window_cfg.step_size
        overlap = int((w - self.zero - s) // self.sample)
        for i, (feat, targ, _, frame_id) in enumerate(data_loader):
            if test_type == "enc":
                pred, _ = self.network.forward_encoder(feat[:, :, self.zero:].to(self.device))
                pred_softmax = torch.nn.functional.softmax(pred, 2)
                targ = targ[:, self.sample - 1 + self.zero::self.sample]

            else:
                mid_enc = self.network.forward_encoder_minimal(feat[:, :, self.zero:].to(self.device))
                if cfg.test_cfg.get('with_gt', False):
                    memory = self.shift_and_sample_from_gt(targ.to(self.device), self.zero)
                    pred = self.network.forward_decoder(mid_enc, memory)
                    pred_softmax = torch.nn.functional.softmax(pred, 2)
                    targ = targ[:, self.sample - 1 + self.zero::self.sample]
                elif cfg.test_cfg.get('individual', False):
                    targ_gt = targ[:, self.sample - 1 + self.zero::self.sample]
                    length = targ_gt.size(1)
                    memory = torch.ones(targ.size(0), self.hparams.grid.elements, length,
                                        device=self.device) / self.hparams.grid.elements
                    ss = 0
                    if cfg.test_cfg.get('start_gt', False):
                        memory[:, :, :cfg.test_cfg.start_gt] = self.shift_and_sample_from_gt(targ.to(self.device),
                            self.zero)[:, :, :cfg.test_cfg.start_gt]
                        ss = cfg.test_cfg.start_gt - 1
                    for j in range(ss, length):
                        if j > ss:
                            # memory[:, :, 1:j + 1] = pred_dec_softmax
                            memory[:, :, j] = pred_softmax[:, :, -1]
                        pred_dec = self.network.forward_decoder(mid_enc, memory[:, :, :j + 1])
                        pred_softmax = torch.nn.functional.softmax(pred_dec, 2)
                    targ = targ_gt
                else:
                    # propergate in time
                    targ_gt = targ[:, self.sample - 1 + self.zero::self.sample]
                    length = targ_gt.size(1)
                    ss = overlap if i > 0 else 0

                    memory = torch.ones(targ.size(0), self.hparams.grid.elements, length,
                                        device=self.device) / self.hparams.grid.elements
                    if i == 0 and cfg.test_cfg.get('start_gt', False):
                        memory[:, :, :cfg.test_cfg.start_gt] = self.shift_and_sample_from_gt(targ.to(self.device),
                            self.zero)[:, :, :cfg.test_cfg.start_gt]
                        ss = cfg.test_cfg.start_gt - 1
                    elif i > 0:
                        memory[:, :, :overlap + 1] = pred_softmax[:, -overlap - 1:, :].permute(0, 2, 1)  # from last iteration

                    for j in range(ss, length-1):
                        if j > ss:
                            # memory[:, :, 1:j + 1] = pred_dec_softmax
                            memory[:, :, j] = pred_softmax[:, -1, :]

                        pred_dec = self.network.forward_decoder_last(mid_enc, memory[:, :, :j + 1])
                        pred_softmax = torch.nn.functional.softmax(pred_dec, 2)
                    if length - 1 > ss:
                        memory[:, :, -1] = pred_softmax[:, -1, :]
                    pred_dec = self.network.forward_decoder(mid_enc, memory)
                    pred_softmax = torch.nn.functional.softmax(pred_dec, 2)
                    targ = targ_gt

            targets_all.append(self.torch_to_numpy(targ))
            preds_softmax_all.append(self.torch_to_numpy(pred_softmax))
            frame_ids_all.extend(self.torch_to_numpy(frame_id))

        targets_all = np.concatenate(targets_all, axis=0)  # [b, s, f]
        preds_softmax_all = np.concatenate(preds_softmax_all, axis=0)  # [b, s, f]
        frame_ids_all = np.asarray(frame_ids_all) + self.zero  # [b]

        logging.info(
            f"Inference done: {targets_all.shape}, {preds_softmax_all.shape},"
        )
        attr_dict = {
            "targets": targets_all,
            "preds_softmax": preds_softmax_all,
            "frame_ids": frame_ids_all,
        }
        return attr_dict
