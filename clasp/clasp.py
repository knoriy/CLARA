import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np

from typing import Tuple, Union, Callable, Optional
from collections import OrderedDict
from encoders.text_encoders import TransformerEncoder 
from encoders.audio_encoders import WhisperAudioEncoder, Cnn10, SimpleCNN
from encoders.modules import PositionalEncoding, LayerNorm, MLPLayers

class CLASP(nn.Module):
    '''
    Contrastive Language-Speech Pre-training 
    '''
    def __init__(self, hparm, text_encoder:Optional=None, audio_encoder:Optional=None) -> None:
        super().__init__()
        self.hparm = hparm

        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        
        if self.text_encoder == None:
            self.text_encoder = TransformerEncoder(
                in_channels = self.hparm.text_encoder_width,
                out_channels = 1024,
                num_layers = self.hparm.text_encoder_layers,
                nhead = self.hparm.text_encoder_heads
                )

        if self.audio_encoder == None:
            self.audio_encoder = SimpleCNN(80, 1024)

        self.text_embedding = nn.Embedding(self.hparm.vocab_size, self.hparm.text_encoder_embedding)
        self.positional_embedding = PositionalEncoding(self.hparm.text_encoder_embedding)
        self.text_projection = nn.Parameter(torch.empty(self.hparm.text_encoder_width, self.hparm.text_encoder_embedding))
        self.ln_final = LayerNorm(self.hparm.text_encoder_width)

        self.tempeture = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # text branch parameters
        self.text_transform = MLPLayers(units=[1024,512,512], dropout=0.1)
        # audio branch parameters
        self.audio_transform = MLPLayers(units=[1024,512,512], dropout=0.1)

    def encode_text(self, text:torch.Tensor):
        x = self.text_embedding(text)
        x = self.positional_embedding(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_audio(self, audio:torch.Tensor):
        x = self.audio_encoder(audio)

        x1 = torch.mean(x, dim=2)
        x2, _ = torch.max(x, dim=2)
        x = x1 + x2
        return x


    def forward(self, text:torch.Tensor=None, audio:torch.Tensor=None):
        if audio is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_audio(audio)

        text_features = self.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        audio_features = self.encode_audio(audio)
        audio_features = F.normalize(audio_features, dim=-1)

        # Final MLP transform
        # text_features = self.text_transform(text_features)
        # audio_features = self.audio_transform(audio_features)

        return text_features, audio_features, self.tempeture.exp()