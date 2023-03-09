import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Tuple, Union, Callable, Optional
from encoders.text_encoders import SimpleTransformer 
from encoders.audio_encoders import WhisperAudioEncoder, SimpleCNN, SimpleCNNLarge, Cnn1D10, Cnn1D12, resnet18, ResNeXt
from encoders.modules import PositionalEncoding, LayerNorm, MLPLayers

class CLASP(nn.Module):
    '''
    Contrastive Language-Speech Pre-training 
    '''
    def __init__(self, hparm, text_encoder:Optional[nn.Module]=None, audio_encoder:Optional[nn.Module]=None) -> None:
        super().__init__()
        self.hparm = hparm

        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        
        if self.text_encoder == None:
            self.text_encoder = SimpleTransformer(
                in_channels = self.hparm.text_encoder_width,
                out_channels = self.hparm.text_encoder_embedding,
                num_layers = self.hparm.text_encoder_layers,
                nhead = self.hparm.text_encoder_heads, 
                batch_first=True,
                )

        if self.audio_encoder == None:
            # self.audio_encoder = SimpleCNN(80, 1024)
            # self.audio_encoder = SimpleCNNLarge(80, 1024)
            # self.audio_encoder = Cnn1D10(80, 1024)
            # self.audio_encoder = Cnn1D12(80, 1024)
            # self.audio_encoder = resnet18(1024)
            # self.audio_encoder = ResNeXt(5,12,1024, 2, 4)
            self.audio_encoder = WhisperAudioEncoder(80, 1024, 1, 1, batch_first=True)

        # ------------
        # Text Layers
        # ------------
        self.text_embedding = nn.Embedding(self.hparm.vocab_size, self.hparm.text_encoder_embedding)
        self.positional_embedding = PositionalEncoding(self.hparm.text_encoder_embedding)
        self.ln_final = LayerNorm(self.hparm.text_encoder_width)
        self.text_transform = MLPLayers(units=[1024,1024], dropout=0.1)

        # ------------
        # Audio Layers
        # ------------
        self.audio_fc1 = nn.Linear(1024, 1024)
        self.audio_transform = MLPLayers(units=[1024,1024], dropout=0.1)

        # ------------
        # Other
        # ------------
        self.audio_tempeture = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text_tempeture = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text(self, text:torch.Tensor):
        x = self.text_embedding(text)
        x = self.positional_embedding(x)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = torch.mean(x, 1)

        return x

    def encode_audio(self, audio:torch.Tensor):
        # audio = audio.unsqueeze(1).permute(0,1,3,2)
        x = self.audio_encoder(audio)

        x1 = torch.mean(x, dim=2)
        x2, _ = torch.max(x, dim=2)
        x = x1 + x2

        x = F.relu(self.audio_fc1(x))

        return x

    def forward(self, text:torch.Tensor, audio:torch.Tensor):
        text_features = self.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        audio_features = self.encode_audio(audio)
        audio_features = F.normalize(audio_features, dim=-1)

        # Final MLP transform
        mlp_text_features = self.text_transform(text_features)
        mlp_audio_features = self.audio_transform(audio_features)

        return text_features, audio_features, self.text_tempeture.exp(), self.audio_tempeture.exp(), mlp_text_features, mlp_audio_features