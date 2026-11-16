"""Convolution blocks shared by the SCSNet inference modules."""
import torch
from torch import nn


def patch_downsample(in_channels):
    """Keep the checkpoint's Conv-LeakyReLU-BatchNorm ordering."""
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 2, stride=2, bias=False),
        nn.LeakyReLU(negative_slope=0.2, inplace=False),
        nn.BatchNorm2d(in_channels),
    )


def preactivated_conv2d(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
    )


def preactivated_conv3d(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm3d(in_channels),
        nn.ReLU(inplace=False),
        nn.Conv3d(in_channels, out_channels, 5, padding=2, bias=False),
    )


def cost_aggregation(layers, channels, G):
    return nn.Sequential(
        preactivated_conv3d(G, channels),
        *(preactivated_conv3d(channels, channels) for _ in range(layers)),
        preactivated_conv3d(channels, 1)
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, F_in):
        features = self.relu(self.bn1(self.conv1(F_in)))
        return self.relu(self.bn2(self.conv2(features)) + F_in)


class HardSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(inplace=False)

    def forward(self, X):
        return self.relu(X + 3) / 6


class HardSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = HardSigmoid()

    def forward(self, X):
        return X * self.sigmoid(X)


class CoordinateAttention(nn.Module):
    """Coordinate Attention used between SSR bottleneck residual blocks."""
    def __init__(self, channels, reduction=32):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, reduced, 1)
        self.bn1 = nn.BatchNorm2d(reduced)
        self.act = HardSwish()
        self.conv_h = nn.Conv2d(reduced, channels, 1)
        self.conv_w = nn.Conv2d(reduced, channels, 1)

    def forward(self, F_in):
        H, W = F_in.shape[-2:]
        z_h = self.pool_h(F_in)
        z_v = self.pool_w(F_in).permute(0, 1, 3, 2)
        joint = self.act(self.bn1(self.conv1(torch.cat((z_h, z_v), dim=2))))
        descriptor_h, descriptor_v = torch.split(joint, (H, W), dim=2)
        attention_h = self.conv_h(descriptor_h).sigmoid()
        attention_v = self.conv_w(descriptor_v.permute(0, 1, 3, 2)).sigmoid()
        return F_in * attention_h * attention_v
