from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from niloc.models.io_factory import InputTCN, PositionalEncodingNone, output_embeddings, output_activation_type


class MyTCNEncoder(nn.Module):
    def __init__(
            self,
            d_input: int,
            d_model: int,
            num_channels: List[int],
            kernel_size: int,
            enc_out_embedding: nn.Module,
            activation: str = "ReLU",
            sample: int = 1,
            dropout: float = 0.5,
    ) -> None:
        super(MyTCNEncoder, self).__init__()

        dict_activation = {"ReLU": nn.ReLU, "GELU": nn.GELU}

        self.tcn = InputTCN(d_input, d_model, num_channels, kernel_size, PositionalEncodingNone(), sample, dropout,
                            activation=dict_activation[activation])
        self.enc_out_embedding = enc_out_embedding

    def get_num_params(self) -> int:
        """ Returns number of trainable paarameters """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, feat: Tensor) -> Tensor:
        """
        Feat shape see enc_in_embedding layer for details
        * Input to transformer is (seq_length, batch_size, d_input)
        Output shape see enc_out_embedding for details
        """
        te_out = self.tcn(feat)
        return self.enc_out_embedding(te_out)


class MyLSTMEncoder(nn.Module):
    def __init__(
            self,
            d_input: int,
            d_model: int,
            lstm_layers: int,
            enc_out_embedding: nn.Module,
            sample: int = 1,
            dropout: float = 0.2,
    ) -> None:
        super(MyLSTMEncoder, self).__init__()

        self.sample = sample
        self.lstm = torch.nn.LSTM(d_input, d_model, lstm_layers, batch_first=False, dropout=dropout)
        self.enc_out_embedding = enc_out_embedding

    def get_num_params(self) -> int:
        """ Returns number of trainable paarameters """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, feat: Tensor) -> Tensor:
        """"
        :param x: shape (batch_size, d_input, seq_length)  - for consistency with ConvNet models
        :return: y: shape (sequence_length, batch_size, d_model) where d_model = d_output (+ d_pos_encoding)
        """
        x = feat.permute(2, 0, 1)  # b, s, f
        y, _ = self.lstm(x)
        if self.sample > 1:
            # every nth element from n-1
            y = y[self.sample - 1::self.sample]
        return self.enc_out_embedding(y)


network_models = {
    "tcn_net": MyTCNEncoder,
    "lstm_net": MyLSTMEncoder,
}


def get_args_as_dict(cfg: Dict[str, Any], skip: Optional[List[str]] = None) -> Dict[str, Any]:
    if skip is None:
        skip = ["name"]
    args = {}

    for key in cfg:
        if key in skip:
            continue
        args[key] = cfg[key]

    return args


def build_seq2seq(arch: str, cfg: DictConfig, input_dim: int = 6, output_dim: int = 3) -> nn.Module:
    # build input/output layers
    global _output_activation
    _output_activation = output_activation_type[cfg.arch.get('output_activation', 'prelu')]
    io_layers = {}

    def build_position_output(layer_args: Dict[str, Any]) -> nn.Module:
        args = get_args_as_dict(layer_args)
        args["dropout"] = cfg.arch.dropout
        args["d_input"], args["d_output"] = cfg.arch.d_model, output_dim
        if "grid_dim" in args:
            args["grid_dim"] = cfg.grid.size
        return output_embeddings[layer_args["name"]](**args)

    if cfg.arch.get("encoder_output", False):
        io_layers["enc_out_embedding"] = build_position_output(cfg.arch.encoder_output)

    model_args = get_args_as_dict(cfg.arch, ["name", "build_model", "encoder_input", "decoder_input", "encoder_output",
                                             "decoder_output"])
    model_args['d_input'] = input_dim
    model = network_models[arch](**model_args, **io_layers)
    return model
