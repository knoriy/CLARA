import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN Networks
class ConvBlock1D3x3(nn.Module):
    def __init__(self, in_channel:int, out_channel:int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, 3)
        self.conv2 = nn.Conv1d(out_channel, out_channel, 3)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.bn(self.conv2(x)))

class Cnn1D10(nn.Module):
    def __init__(self, n_mels:int, out_channels:int):
        super().__init__()
        in_sizes = [n_mels] + [2048, 2048, 2048, 2048] + [out_channels]
        self.layers = nn.ModuleList(
            [ ConvBlock1D3x3(in_size, out_size)
            for (in_size, out_size) in zip(in_sizes, in_sizes[1:])]
        )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Cnn1D12(nn.Module):
    def __init__(self, n_mels:int, out_channels:int):
        super().__init__()
        in_sizes = [n_mels] + [2048, 2048, 2048, 2048, 2048] + [out_channels]
        self.layers = nn.ModuleList(
            [ ConvBlock1D3x3(in_size, out_size)
            for (in_size, out_size) in zip(in_sizes, in_sizes[1:])]
        )
    
    def forward(self, x):
        for linear in self.layers:
            x = linear(x)
        return x

class ConvBlock2D3x3(nn.Module):
    def __init__(self, in_channel:int, out_channel:int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.bn(self.conv2(x)))

class Cnn2D10(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        in_sizes = [in_channels] + [2048, 2048, 2048, 2048] + [out_channels]
        self.layers = nn.ModuleList(
            [ ConvBlock2D3x3(in_size, out_size)
            for (in_size, out_size) in zip(in_sizes, in_sizes[1:])]
        )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = torch.mean(x, dim=3)
        
        return x