from typing import Optional, List, Tuple, Dict, Any

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    Transformer
)

from niloc.models.io_factory import positional_encoders, input_embeddings, output_embeddings, output_activation_type


class MyTransformerEncoder(nn.Module):
    """
    Parameters
    ----------
    d_model: No. of input fratures for encoder/decoder.
    nhead: Number of heads.
    d_feedforward:
        The dimension of the feedforward network model
    num_layers: Number of encoder layers to stack.
    dropout: Dropout probability.
    enc_in_embedding: module for feature extraction from input
    enc_out_embedding: module for processing output
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            d_feedforward: int,
            num_layers: int,
            enc_in_embedding: nn.Module,
            enc_out_embedding: nn.Module,
            dropout: float = 0.5,
            **kwargs
    ) -> None:
        super(MyTransformerEncoder, self).__init__()

        self.enc_in_embedding = enc_in_embedding
        self.enc_out_embedding = enc_out_embedding

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def get_num_params(self) -> int:
        """ Returns number of trainable paarameters """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, feat: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Feat shape see enc_in_embedding layer for details
        * Input to transformer is (seq_length, batch_size, d_input)
        Output shape see enc_out_embedding for details
        """
        te_in = self.enc_in_embedding(feat)
        te_out = self.transformer_encoder(te_in, mask=src_mask)
        return self.enc_out_embedding(te_out)


class MyTransformer(nn.Module):
    """
    Full transformer architecture

    Parameters
    ----------
    d_model: No. of input fratures for encoder/decoder.
    nhead: Number of heads.
    d_feedforward:
        The dimension of the feedforward network model
    encoder_layers: Number of encoder layers to stack.
    decoder_layers: Number of decoder layers to stack.
    dropout: Dropout probability.
    dec_in_embedding: module for feature extraction from decoder input
    dec_out_embedding: module for processing output
    enc_in_embedding: module for feature extraction from encoder input
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            d_feedforward: int,
            encoder_layers: int,
            decoder_layers: int,
            dec_in_embedding: nn.Module,
            dec_out_embedding: nn.Module,
            enc_in_embedding: nn.Module,
            dropout: float = 0.5,
            **kwargs
    ) -> None:
        super(MyTransformer, self).__init__()

        self.transformer = Transformer(d_model, nhead, encoder_layers, decoder_layers, d_feedforward, dropout)

        self.enc_in_embedding = enc_in_embedding
        self.dec_in_embedding = dec_in_embedding
        self.dec_out_embedding = dec_out_embedding

    def get_num_params(self) -> int:
        """ Returns number of trainable paarameters """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, feat_enc: Tensor, feat_dec: Tensor, ) -> Tensor:
        """
        params:
            feat_enc - input for encoder. see enc_in_embedding layer for details
            feat_dec - input for decoder. see dec_in_embedding layer for details
        * Input to transformer is (seq_length, batch_size, d_input)
        Output shape see dec_out_embedding for details
        """
        encoder_output = self.forward_encoder(feat_enc)
        return self.forward_decoder(encoder_output, feat_dec)

    def forward_encoder(self, feat_enc: Tensor) -> Tensor:
        """
        params:
            feat_enc - input for encoder. see enc_in_embedding layer for details
        * Input to transformer is (seq_length, batch_size, d_input)

        Output shape (sequence_length, batch_size, d_model)
        """
        encoder_input = self.enc_in_embedding(feat_enc)
        encoder_output = self.transformer.encoder(encoder_input)
        return encoder_output

    def forward_decoder(self, encoder_output: Tensor, feat_dec: Tensor, ) -> Tensor:
        """
        params:
            encoder_output -  memory from encoder. (seq_length1, batch_size, d_input)
            feat_dec - input for decoder. see dec_in_embedding layer for details
        Output shape see dec_out_embedding for details
        """

        decoder_input = self.dec_in_embedding(feat_dec)

        decoder_output = self.transformer.decoder(decoder_input, encoder_output,
                                                  tgt_mask=self.transformer.generate_square_subsequent_mask(
                                                      decoder_input.size(0)).to(decoder_input.device)
                                                  )

        return self.dec_out_embedding(decoder_output)

    def forward_decoder_last(self, encoder_output: Tensor, feat_dec: Tensor, ) -> Tensor:
        """
        Only returns the value of last position (to reduce no of opertaions)
        params:
            encoder_output -  memory from encoder. (seq_length1, batch_size, d_input)
            feat_dec - input for decoder. see dec_in_embedding layer for details
        Output shape see dec_out_embedding for details
        """

        decoder_input = self.dec_in_embedding(feat_dec)

        decoder_output = self.transformer.decoder(decoder_input, encoder_output,
                                                  tgt_mask=self.transformer.generate_square_subsequent_mask(
                                                      decoder_input.size(0)).to(decoder_input.device)
                                                  )

        return self.dec_out_embedding(decoder_output[-1:])  # prediction for last decoder input (dim seq=1)


class MyTransformer2Branch(MyTransformer):
    """
    Full transformer architecture with additional encoder output

    Parameters
    ----------
    d_model: No. of input fratures for encoder/decoder.
    nhead: Number of heads.
    d_feedforward:
        The dimension of the feedforward network model
    encoder_layers: Number of encoder layers to stack.
    decoder_layers: Number of decoder layers to stack.
    encoder_only_layers: Number of additional encoder layers to stack.
    dropout: Dropout probability.
    dec_in_embedding: module for feature extraction from decoder input
    dec_out_embedding: module for processing decoder output
    enc_in_embedding: module for feature extraction from encoder input
    enc_out_embedding: module for processing encoder output
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            d_feedforward: int,
            encoder_layers: int,
            decoder_layers: int,
            encoder_only_layers: int,
            dec_in_embedding: nn.Module,
            dec_out_embedding: nn.Module,
            enc_in_embedding: nn.Module,
            enc_out_embedding: nn.Module,
            dropout: float = 0.5,
            **kwargs
    ) -> None:
        super(MyTransformer2Branch, self).__init__(d_model, nhead, d_feedforward, encoder_layers, decoder_layers,
                                                   dec_in_embedding, dec_out_embedding, enc_in_embedding, dropout)

        self.transformer_encoder_only = TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead, d_feedforward, dropout), encoder_only_layers)

        self.enc_out_embedding = enc_out_embedding

    def forward(self, feat_enc: Tensor, feat_dec: Tensor, ) -> Tuple[Tensor, Tensor]:
        """
        params:
            feat_enc - input for encoder. see enc_in_embedding layer for details
            feat_dec - input for decoder. see dec_in_embedding layer for details
        * Input to transformer is (seq_length, batch_size, d_input)
        :returns
            encoder_extra_out. see enc_out_embedding for details
            decoder_output. see dec_out_embedding for details
        """
        encoder_only_output, memory = self.forward_encoder(feat_enc)
        decoder_output = self.forward_decoder(memory, feat_dec)
        return encoder_only_output, decoder_output

    def forward_encoder(self, feat_enc: Tensor) -> Tuple[Tensor, Tensor]:
        """
        params:
            feat_enc - input for encoder. see enc_in_embedding layer for details
        * Input to transformer is (seq_length, batch_size, d_input)
        returns:
            encoder_extra_out. see enc_out_embedding for details
            encoder_output. shape (sequence_length, batch_size, d_model)
        """
        encoder_output = super(MyTransformer2Branch, self).forward_encoder(feat_enc)

        encoder_only_out = self.transformer_encoder_only(encoder_output)
        encoder_extra_out = self.enc_out_embedding(encoder_only_out)
        return encoder_extra_out, encoder_output

    def forward_encoder_minimal(self, feat_enc: Tensor) -> Tensor:
        """
        params:
            feat_enc - input for encoder. see enc_in_embedding layer for details
        * Input to transformer is (seq_length, batch_size, d_input)
        returns:
            encoder_output. see enc_out_embedding for details
        """
        encoder_output = super(MyTransformer2Branch, self).forward_encoder(feat_enc)
        return encoder_output


transformer_models = {
    "transformer_encoder": MyTransformerEncoder,
    "transformer_full": MyTransformer,
    "transformer_2branch": MyTransformer2Branch,
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


def get_positional_encoding(cfg: DictConfig, pe_name: str, ) -> Tuple[nn.Module, int]:
    """
    :param pe_name: type of positional encoding
    :return:
        positional encoding module
        expected feature size in input tensor
    """
    # concatenate or add
    in_feat = cfg.arch.d_model // 3 if pe_name.endswith("cat") else cfg.arch.d_model
    pos_encoder = positional_encoders[pe_name](
        in_feat, dropout=cfg.arch.dropout, max_len=cfg.net_config.window_size * 2, )
    return pos_encoder, cfg.arch.d_model - in_feat if pe_name.endswith("cat") else cfg.arch.d_model


def build_transformer(arch: str, cfg: DictConfig, input_dim: int = 6, output_dim: int = 3) -> nn.Module:
    # build input/output layers
    global _output_activation
    _output_activation = output_activation_type[cfg.arch.get('output_activation', 'relu')]
    io_layers = {}

    def build_velocity_input(layer_args: Dict[str, Any]) -> nn.Module:
        args = get_args_as_dict(layer_args, ["name", "pe"])
        pe, layer_in = get_positional_encoding(cfg, layer_args["pe"])
        args["dropout"] = cfg.arch.dropout
        args["d_input"], args["d_output"] = input_dim, layer_in
        if "grid_dim" in args:
            args["grid_dim"] = cfg.grid.size
        if cfg.arch.get("sample", False):
            args["sample"] = cfg.arch.sample
        return input_embeddings[layer_args["name"]](pe=pe, **args)

    def build_position_input(layer_args: Dict[str, Any]) -> nn.Module:
        args = get_args_as_dict(layer_args, ["name", "pe"])
        pe, layer_in = get_positional_encoding(cfg, layer_args["pe"])
        args["dropout"] = cfg.arch.dropout
        args["d_input"], args["d_output"] = output_dim, layer_in
        if "grid_dim" in args:
            args["grid_dim"] = cfg.grid.size
        return input_embeddings[layer_args["name"]](pe=pe, **args)

    def build_position_output(layer_args: Dict[str, Any]) -> nn.Module:
        args = get_args_as_dict(layer_args)
        args["dropout"] = cfg.arch.dropout
        args["d_input"], args["d_output"] = cfg.arch.d_model, output_dim
        if "grid_dim" in args:
            args["grid_dim"] = cfg.grid.size
        return output_embeddings[layer_args["name"]](**args)

    if arch == "transformer_decoder":
        io_layers["dec_in_embedding"] = build_velocity_input(cfg.arch.decoder_input)
        io_layers["enc_in_embedding"] = build_position_input(cfg.arch.encoder_input)
    else:
        if cfg.arch.get("encoder_input", False):
            io_layers["enc_in_embedding"] = build_velocity_input(cfg.arch.encoder_input)
        if cfg.arch.get("decoder_input", False):
            io_layers["dec_in_embedding"] = build_position_input(cfg.arch.decoder_input)

    if cfg.arch.get("decoder_output", False):
        io_layers["dec_out_embedding"] = build_position_output(cfg.arch.decoder_output)
    if cfg.arch.get("encoder_output", False):
        io_layers["enc_out_embedding"] = build_position_output(cfg.arch.encoder_output)

    model_args = get_args_as_dict(cfg.arch, ["name", "encoder_input", "decoder_input", "encoder_output",
                                             "decoder_output"])
    model = transformer_models[arch](**model_args, **io_layers)
    return model
