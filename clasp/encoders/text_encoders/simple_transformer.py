import torch
import torch.nn as nn

from ..modules import Transformer
from typing import Tuple, Union, Callable, Optional


class SimpleTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, act_layer: Callable = nn.GELU):
        super().__init__()
        self.transformer = Transformer(width, layers, heads, act_layer)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.transformer(x)