'''
Whisper: An ASR model by OpenAI
https://cdn.openai.com/papers/whisper.pdf
'''


import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from typing import Dict, Iterable, Optional

from ..modules import Transformer, PositionalEncoding, LayerNorm


class WhisperAudioEncoder(nn.Module):
    def __init__(self, n_mels: int, out_channels: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.positional_encoding = PositionalEncoding(out_channels)

        self.tranformer_blocks = Transformer(out_channels, n_layer, n_head)
        self.ln = LayerNorm(out_channels)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        
        x = x.permute(0, 2, 1)
        x = self.positional_encoding(x)

        x = self.tranformer_blocks(x)
        x = self.ln(x)
        return x