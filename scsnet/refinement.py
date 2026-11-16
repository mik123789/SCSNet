"""Structure-Selective Refinement (SSR): RGB-guided full-resolution residual."""
from torch import nn

from .blocks import CoordinateAttention, ResidualBlock


def _conv_norm_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
    )


def _upsample(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
    )


class StructureSelectiveRefinement(nn.Module):
    """Two-level hourglass receiving X_in=Concat(I_L,D_init), in this order."""
    def __init__(self, channels=32):
        super().__init__()
        self.conv_start = _conv_norm_relu(4, channels)
        self.down1 = _conv_norm_relu(channels, 2 * channels, stride=2)
        self.down2 = _conv_norm_relu(2 * channels, 4 * channels, stride=2)
        self.bottleneck = nn.Sequential(
            ResidualBlock(4 * channels),
            CoordinateAttention(4 * channels),
            ResidualBlock(4 * channels),
        )
        self.up2 = _upsample(4 * channels, 2 * channels)
        self.up1 = _upsample(2 * channels, channels)
        self.final_res_block = ResidualBlock(channels)
        self.final_conv = nn.Conv2d(channels, 1, 3, padding=1, bias=True)

    def forward(self, X_in):
        F_1 = self.conv_start(X_in)
        F_2 = self.down1(F_1)
        F_4 = self.bottleneck(self.down2(F_2))
        F_2_out = self.up2(F_4) + F_2
        F_1_out = self.up1(F_2_out) + F_1
        return self.final_conv(self.final_res_block(F_1_out))
