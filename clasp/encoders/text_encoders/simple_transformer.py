import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Tuple, Union, Callable, Optional

from ..modules import TransformerEncoder, LayerNorm

class SimpleTransformer(nn.Module):
    def __init__(self, 
        in_channels: int,
        out_channels: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = False,
        norm_first: bool = False,
        norm: nn.Module = None
        ):
        super().__init__()
        self.transformer_encoder = TransformerEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            norm=norm, 
        )
        self.ln = LayerNorm(out_channels)

    def forward(self, x: torch.Tensor):
        return self.ln(self.transformer_encoder(x))