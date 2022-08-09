import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import math
import numpy as np

from typing import Tuple, Union, Callable, Optional
from collections import OrderedDict
from encoders.audio_encoders.pann_model import Cnn10
from text.symbols import symbols


class CLAP(nn.Module):
    '''
    Contrastive Language-Audio Pre-training 
    '''
    def __init__(self, hparm, text_encoder=None, audio_encoder=None) -> None:
        super().__init__()
        self.hparm = hparm

        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        
        if self.text_encoder == None:
            self.text_encoder = TextEncoder(
                width = self.hparm.text_encoder_width,
                layers = self.hparm.text_encoder_layers, 
                heads = self.hparm.text_encoder_heads, 
            )
        if self.audio_encoder == None:
            self.audio_encoder = Cnn10(1024)

        self.text_embedding = nn.Embedding(len(symbols) + 1, self.hparm.text_encoder_embedding)
        self.positional_embedding = PositionalEncoding(self.hparm.text_encoder_embedding)
        self.text_projection = nn.Parameter(torch.empty(self.hparm.text_encoder_width, self.hparm.text_encoder_embedding))
        self.ln_final = LayerNorm(self.hparm.text_encoder_width)

        self.tempeture = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # text branch parameters
        self.text_transform = MLPLayers(units=[1024,512,512], dropout=0.1)
        # audio branch parameters
        self.audio_transform = MLPLayers(units=[1024,512,512], dropout=0.1)

    def encode_text(self, text):
        x = self.text_embedding(text)

        x = self.positional_embedding(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_audio(self, audio):
        return self.audio_encoder(audio)


    def forward(self, text=None, audio=None):
        if audio is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_audio(audio)

        text_features = self.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        audio_features = self.encode_audio(audio)
        audio_features = F.normalize(audio_features, dim=-1)

        # Final MLP transform
        text_features = self.text_transform(text_features)
        audio_features = self.audio_transform(audio_features)

        return text_features, audio_features, self.tempeture.exp()

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
    
class TextEncoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, act_layer: Callable = nn.GELU):
        super().__init__()
        self.transformer = Transformer(width, layers, heads, act_layer)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.transformer(x)

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
        return self.dropout(x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, act_layer: Callable = nn.GELU):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, act_layer: Callable = nn.GELU):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, act_layer=act_layer)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x
