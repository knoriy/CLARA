import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    '''
    Simple Conv1d model with batchnorm and dropout
    '''
    def __init__(self, in_channel:int, out_channel:int) -> None:
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channel,out_channel, 3)
        self.cnn2 = nn.Conv1d(out_channel,out_channel, 3)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.cnn1(x))))
        x = F.dropout(F.relu(self.bn2(self.cnn2(x))))

        return x


class SimpleCNNLarge(nn.Module):
    '''
    Simple Conv1d model with batchnorm and dropout
    '''
    def __init__(self, in_channel:int, out_channel:int) -> None:
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channel,out_channel, 3)
        self.cnn2 = nn.Conv1d(out_channel,out_channel, 3)
        self.cnn3 = nn.Conv1d(out_channel,out_channel, 3)
        self.cnn4 = nn.Conv1d(out_channel,out_channel, 3)
        self.cnn5 = nn.Conv1d(out_channel,out_channel, 3)
        self.cnn6 = nn.Conv1d(out_channel,out_channel, 3)

        self.bn1 = nn.BatchNorm1d(out_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.bn3 = nn.BatchNorm1d(out_channel)
        self.bn4 = nn.BatchNorm1d(out_channel)
        self.bn5 = nn.BatchNorm1d(out_channel)
        self.bn6 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.cnn1(x))))
        x = F.dropout(F.relu(self.bn2(self.cnn2(x))))
        x = F.dropout(F.relu(self.bn3(self.cnn3(x))))
        x = F.dropout(F.relu(self.bn4(self.cnn4(x))))
        x = F.dropout(F.relu(self.bn5(self.cnn5(x))))
        x = F.dropout(F.relu(self.bn6(self.cnn6(x))))
        return x