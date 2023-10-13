import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
pl_logger = logging.getLogger('pytorch_lightning')

class CLARALoss(nn.Module):
    '''
    CLARALoss is adopted from the mlfoundations' open_clip: https://github.com/mlfoundations/open_clip
    '''
    def __init__(self, cache_labels:bool = False) -> None:
        super().__init__()

        self.cache_labels = cache_labels

        # cache state
        self.prev_num_logits = 0
        self.labels = {}


    def forward(self, text_features, audio_features, text_temperature:float=1.0, audio_temperature:float=1.0):
        device = audio_features.device

        logits_per_audio = audio_temperature * audio_features @ text_features.T
        logits_per_text = text_temperature * text_features @ audio_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_audio.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        
        total_loss = (
            F.cross_entropy(logits_per_audio, labels) + 
            F.cross_entropy(logits_per_text, labels)) / 2
        
        return total_loss

class CLIPLoss(nn.Module):
    '''
    CLIP Loss
    '''
    def __init__(self, cache_labels:bool = False) -> None:
        super().__init__()

    def forward(self, text_features:torch.Tensor, audio_features:torch.Tensor, temperature:int=1):
        logits = (text_features @ audio_features.T) / temperature
        audio_similarity = audio_features @ audio_features.T
        texts_similarity = text_features @ text_features.T
        targets = F.softmax(
            (audio_similarity + texts_similarity) / 2 * temperature, dim=-1
        )

        texts_loss = F.cross_entropy(logits, targets)
        images_loss = F.cross_entropy(logits.T, targets.T)
        loss =  (images_loss + texts_loss) / 2 # shape: (batch_size)
        return loss