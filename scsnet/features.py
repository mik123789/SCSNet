"""Adaptive Context Feature Extractor (ACFE) and Scale-Selective Encoder."""
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import patch_downsample, preactivated_conv2d


class ScaleSelectiveEncoder(nn.Module):
    """SSE: X_out = ReLU(P(sum_i omega_i F_di(X_red)) + shortcut).

    omega is an unconstrained scalar per dilation branch, matching the paper
    and trained checkpoints; it is not normalized through a softmax.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        reduced = max(8, out_channels // 2)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduced, 1, bias=False),
            nn.BatchNorm2d(reduced), nn.ReLU(inplace=False),
        )
        self.branches = nn.ModuleList(tuple(nn.Sequential(
            nn.Conv2d(reduced, reduced, 3, padding=d, dilation=d, bias=False),
            nn.BatchNorm2d(reduced), nn.ReLU(inplace=False),
        ) for d in (1, 2, 4)))
        self.omega = nn.Parameter(torch.ones(3))
        self.project = nn.Sequential(
            nn.Conv2d(reduced, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = None if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, X):
        X_red = self.reduce(X)
        weighted = sum(branch(X_red) * self.omega[i]
                       for i, branch in enumerate(self.branches))
        shortcut = X if self.shortcut is None else self.shortcut(X)
        return self.relu(self.project(weighted) + shortcut)


class FeatureUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # UpsamplingBilinear2d uses align_corners=True in the original ACFE.
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Sequential(
            preactivated_conv2d(int(in_channels * 1.5), out_channels),
            preactivated_conv2d(out_channels, out_channels),
        )

    def forward(self, skip, coarse):
        upsampled = self.up(coarse)
        # Retain the original decoder's crop convention for odd skip sizes.
        upsampled = F.pad(upsampled, (0, -(skip.shape[3] % 2),
                                      0, -(skip.shape[2] % 2)))
        return self.conv(torch.cat((skip, upsampled), dim=1))


def _encoder_block(in_channels, out_channels, sse_blocks):
    return nn.Sequential(
        patch_downsample(in_channels),
        ScaleSelectiveEncoder(in_channels, out_channels),
        *(ScaleSelectiveEncoder(out_channels, out_channels)
          for _ in range(sse_blocks - 1))
    )


class AdaptiveContextFeatureExtractor(nn.Module):
    """Shared ACFE returning (F_1/8, F_1/4, F_1/2)."""
    def __init__(self, c=48, sse_blocks=2):
        super().__init__()
        stem = nn.Sequential(nn.Conv2d(3, c, 3, padding=1),
                             preactivated_conv2d(c, c))
        self.block0 = nn.Sequential(stem, _encoder_block(c, 2 * c, sse_blocks))
        self.blocks = nn.ModuleList((
            _encoder_block(2 * c, 4 * c, sse_blocks),
            _encoder_block(4 * c, 8 * c, sse_blocks),
        ))
        self.upblocks = nn.ModuleList((
            FeatureUpsample(8 * c, 4 * c),
            FeatureUpsample(4 * c, 2 * c),
        ))

    def forward(self, I):
        F_half = self.block0(I)
        F_quarter = self.blocks[0](F_half)
        F_eighth = self.blocks[1](F_quarter)
        F_quarter = self.upblocks[0](F_quarter, F_eighth)
        F_half = self.upblocks[1](F_half, F_quarter)
        return F_eighth, F_quarter, F_half
