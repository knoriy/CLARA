import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

class Accuracy(nn.Module):
    '''
    CLAPLoss is adopted from the mlfoundations' open_clip: https://github.com/mlfoundations/open_clip
    '''
    def __init__(self, top_k:int = None, cache_labels:bool = False) -> None:
        super().__init__()

        self.cache_labels = cache_labels
        self.accuracy = torchmetrics.Accuracy('multiclass', top_k=top_k)

        # cache state
        self.prev_num_logits = 0
        self.labels = {}


    def forward(self, text_features, audio_features, text_temperature:float=1.0, audio_temperature:float=1.0, mlp_text_features=None, mlp_audio_features=None):
        device = audio_features.device

        t_logits_per_audio = text_temperature * mlp_audio_features @ text_features.T
        t_logits_per_text = text_temperature * text_features @ mlp_audio_features.T
        a_logits_per_audio = audio_temperature * audio_features @ mlp_text_features.T
        a_logits_per_text = audio_temperature * mlp_text_features @ audio_features.T

        # calculated ground-truth and cache if enabled
        num_logits = a_logits_per_audio.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            # if self.world_size > 1 and self.local_loss:
            #     labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        
        acc = self.accuracy(t_logits_per_audio, labels)
        return acc