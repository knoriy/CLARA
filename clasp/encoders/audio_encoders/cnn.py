import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, 1024, 1)
        self.conv2 = nn.Conv1d(1024, out_channel, 3)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.bn(self.conv2(x)))

class Cnn10(nn.Module):
    def __init__(self, n_mels:int, out_channel:int):
        super().__init__()
        in_sizes = [n_mels] + [128, 246, 512, 1024, 2048] + [out_channel]
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
        in_sizes = [n_mels] + [128, 246, 512, 1024, 2048] + [out_channel]
        self.layers = nn.ModuleList(
            [ ConvBlock(in_size, out_size)
            for (in_size, out_size) in zip(in_sizes, in_sizes[1:])]
        )
    
    def forward(self, x):
        for linear in self.layers:
            x = linear(x)
        return x

