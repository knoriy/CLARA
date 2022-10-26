import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    '''
    Simple Conv1d model with batchnorm and dropout
    '''
    def __init__(self, in_channel:int, out_channel:int) -> None:
        super().__init__()
        self.conv1d1 = nn.Conv1d(in_channel,out_channel, 3)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv1d2 = nn.Conv1d(out_channel,out_channel, 3)
        self.bn2 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1d1(x)))
        x = F.dropout(x)

        x = F.relu(self.bn2(self.conv1d2(x)))
        x = F.dropout(x)
        return x

class Cnn10(nn.Module):
    def __init__(self, n_mels:int, out_channel:int):
        super().__init__()
        in_sizes = [n_mels] + [128, 256, 512, 1024] + [out_channel]
        self.layers = nn.ModuleList(
            [ ConvBlock(in_size, out_size)
            for (in_size, out_size) in zip(in_sizes, in_sizes[1:])]
        )
    
    def forward(self, x):
        for linear in self.layers:
            x = linear(x)
        return x

class Cnn12(nn.Module):
    def __init__(self, n_mels:int, out_channel:int):
        super().__init__()
        self.block_1 = ConvBlock(n_mels, 128)
        self.layers = nn.Sequential(
            ConvBlock(128, 512),
            ConvBlock(512, 1024),
            ConvBlock(1024, 1024),
        )
    def forward(self, x):
        x = self.block_1(x)
        x = self.layers(x)
        return x

