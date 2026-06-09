import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class TwoLayerFC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TwoLayerFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # out = self.linear(F.softmax(x))
        x = F.relu(self.linear1(x))
        out = self.linear2(x)
        return out


class ThreeLayerMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ThreeLayerMLP, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # out = self.linear(F.softmax(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        out = self.linear3(x)
        return out


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"

