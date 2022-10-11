import torch
import torch.nn as nn

from typing import Tuple, Union, Callable, Optional


class TransformerEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels:int, num_layers: int, nhead: int, batch_first:bool=False):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=nhead, batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.project = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        return self.project(self.transformer_encoder(x))