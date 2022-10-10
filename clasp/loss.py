import torch
import torch.nn as nn
import torch.nn.functional as F

class CLAPLoss(nn.Module):
    '''
    CLAPLoss is adopted from the mlfoundations' open_clip: https://github.com/mlfoundations/open_clip
    '''
    def __init__(self, cache_labels:bool = False) -> None:
        super().__init__()

        self.cache_labels = cache_labels

        # cache state
        self.prev_num_logits = 0
        self.labels = {}


    def forward(self, text_features, audio_features, temperature:int=1):
        device = audio_features.device
        
        logits_per_audio = (audio_features @ text_features.T) * temperature
        logits_per_text = (text_features @ audio_features.T) * temperature

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_audio.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            # if self.world_size > 1 and self.local_loss:
            #     labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        
        total_loss = (
            F.cross_entropy(logits_per_audio, labels) + 
            F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss