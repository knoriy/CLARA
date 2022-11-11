import torch
import torch.nn as nn

from typing import Tuple, Union, Callable, Optional

from ..modules import TransformerEncoder, LayerNorm

class SimpleTransformer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, num_layers: int, nhead: int, batch_first:bool=False):
        super().__init__()
        self.transformer_encoder = TransformerEncoder(in_channels, out_channels, num_layers, nhead, batch_first)
        self.ln = LayerNorm(out_channels)

    def forward(self, x: torch.Tensor):
        return self.ln(self.transformer_encoder(x))