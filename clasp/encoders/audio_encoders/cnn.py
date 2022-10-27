import torch.nn as nn
import torch.nn.functional as F

# CNN Networks
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
    def __init__(self, n_mels:int, out_channels:int):
        super().__init__()
        # in_sizes = [n_mels] + [1024, 1024, 1024, 1024] + [out_channel]
        in_sizes = [n_mels] + [1024] +  [out_channels]
        self.layers = nn.ModuleList(
            [ ConvBlock(in_size, out_size)
            for (in_size, out_size) in zip(in_sizes, in_sizes[1:])]
        )
    
    def forward(self, x):
        for linear in self.layers:
            x = linear(x)

        return x

class Cnn12(nn.Module):
    def __init__(self, n_mels:int, out_channels:int):
        super().__init__()
        in_sizes = [n_mels] + [128, 256, 512, 1024, 2048] + [out_channels]
        self.layers = nn.ModuleList(
            [ ConvBlock(in_size, out_size)
            for (in_size, out_size) in zip(in_sizes, in_sizes[1:])]
        )
    
    def forward(self, x):
        for linear in self.layers:
            x = linear(x)
        return x

# Residual Network 
class ResidualBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, 3, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channel)

        self.conv2 = nn.Conv1d(out_channel, out_channel, 3,padding='same')
        self.bn2 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x =  self.bn2(self.conv2(x))
        return F.relu(x + residual)

class ResNet(nn.Module):
    def __init__(self, n_mels:int, out_channels:int, kernal_size:int=3, depth:int=1):
        super().__init__()

        self.proj1 = nn.Conv1d(n_mels, out_channels, kernal_size)

        self.layers = nn.ModuleList(
            [ ResidualBlock(out_channels, out_channels) for _ in range(depth) ]
        )
    
    def forward(self, x):
        x = self.proj1(x)

        for linear in self.layers:
            x = linear(x)
        return x