from typing import Union

import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities import AttributeDict

from niloc.models.seq2seq_factory import build_seq2seq
from niloc.models.transformer_factory import build_transformer


def get_model(
        arch: str, cfg: Union[DictConfig, AttributeDict], input_dim: int = 6, output_dim: int = 3
) -> torch.nn.Module:
    """
    Create a model, given model name and configurations
    Args:
        - arch - model name
        - cfg - configuration
        - input_dim, output_dim - input, output feature dimensions
    """

    if "transformer" in cfg.arch.name:
        return build_transformer(arch, cfg, input_dim, output_dim)
    else:
        return build_seq2seq(arch, cfg, input_dim, output_dim)
