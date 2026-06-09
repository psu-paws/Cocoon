# Copied from https://github.com/google-research/DP-FTRL/commit/35b6134b
# Follow the same model architecture as Multi-Epoch MF

"""Defines the neural networks used in the experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numbers import Number


def to_tuple(v, n):
    """Converts input to tuple."""
    if isinstance(v, tuple):
        return v
    elif isinstance(v, Number):
        return (v,) * n
    else:
        return tuple(v)


def objax_kaiming_normal(tensor, kernel_size, in_channels, out_channels, gain=1):
    """Objax's way of initializing using kaiming normal."""
    shape = (*to_tuple(kernel_size, 2), in_channels, out_channels)
    fan_in = np.prod(shape[:-1])

    kaiming_normal_gain = np.sqrt(1 / fan_in)
    std = gain * kaiming_normal_gain
    with torch.no_grad():
        return tensor.normal_(0, std)


def objax_initialize_conv(convs):
    """Objax's default initialization for conv2d."""
    for conv in convs:
        objax_kaiming_normal(conv.weight, conv.kernel_size, conv.in_channels, conv.out_channels)
        nn.init.zeros_(conv.bias)


def objax_initialize_linear(fcs):
    """Objax's default initialization for linear layer."""
    for fc in fcs:
        nn.init.xavier_normal_(fc.weight)
        nn.init.zeros_(fc.bias)


# We'll use this architecture for CIFAR-10.
class VGG(nn.Module):
    def __init__(self, nclass, dense_size, activation=torch.tanh, colors=3):
        super(VGG, self).__init__()
        self.activation = activation

        self.conv1_1 = nn.Conv2d(colors, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.fc1 = nn.Linear(128 * 16, dense_size)
        self.fc2 = nn.Linear(dense_size, nclass)

        objax_initialize_conv([self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2, self.conv3_1, self.conv3_2])
        objax_initialize_linear([self.fc1, self.fc2])

        self._name = 'VGG' + str(dense_size)

    def forward(self, x):
        x = self.activation(self.conv1_1(x))
        x = self.activation(self.conv1_2(x))
        x = F.max_pool2d(x, 2, 2)

        x = self.activation(self.conv2_1(x))
        x = self.activation(self.conv2_2(x))
        x = F.max_pool2d(x, 2, 2)

        x = self.activation(self.conv3_1(x))
        x = self.activation(self.conv3_2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.reshape(-1, 128 * 4 * 4)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return self._name