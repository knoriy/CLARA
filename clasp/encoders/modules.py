import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch import Tensor

import math

from typing import Tuple, Union, Callable, Optional
from collections import OrderedDict

class MLPLayers(nn.Module):
    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, x):
        x = self.sequential(x)
        return x

class PositionalEncoding(nn.Module):
    '''
    Positional Encododing from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html#:~:text=inf%27)%2C%20diagonal%3D1)-,PositionalEncoding,-module%20injects%20some
    '''
    def __init__(self, in_channels: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_channels, 2) * (-math.log(10000.0) / in_channels))
        pe = torch.zeros(max_len, 1, in_channels)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return torch.tensor(self.dropout(x), requires_grad=True, device=x.device)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class TransformerEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels:int, 
                 num_layers: int, 
                 nhead: int, 
                 dim_feedforward:int=1024,
                 dropout:float=0.1, 
                 activation:Union[str, Callable[[Tensor], Tensor]] = F.relu, 
                 layer_norm_eps:float=1e-5,
                 batch_first:bool=False, 
                 norm_first: bool = False,
                 norm:nn.Module = None,
            ):
        super().__init__()
        if norm is None:
            norm = LayerNorm(in_channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = in_channels, 
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout, 
            activation = activation, 
            layer_norm_eps = layer_norm_eps, 
            batch_first = batch_first, 
            norm_first = norm_first, 
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer, 
            num_layers = num_layers, 
            norm = norm,
        )
        self.project = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        return self.project(self.transformer_encoder(x))