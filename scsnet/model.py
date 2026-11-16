"""Selective Context Stereo Network (SCSNet), inference architecture."""
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import cost_aggregation
from .config import SCSNetConfig
from .cost_volume import OutlierSuppressedCostVolume
from .features import AdaptiveContextFeatureExtractor
from .refinement import StructureSelectiveRefinement


def _validate_images(I_L, I_R):
    if not isinstance(I_L, torch.Tensor) or not isinstance(I_R, torch.Tensor):
        raise TypeError('Stereo inputs must be torch tensors')
    if I_L.ndim != 4 or I_L.shape != I_R.shape or I_L.shape[1] != 3:
        raise ValueError('Stereo inputs must have matching (B, 3, H, W) shapes')
    if not I_L.is_floating_point() or not I_R.is_floating_point():
        raise TypeError('Stereo inputs must be floating-point tensors')
    if I_L.dtype != I_R.dtype or I_L.device != I_R.device:
        raise ValueError('Stereo inputs must share dtype and device')
    if I_L.shape[0] < 1 or any(size < 8 or size % 8 for size in I_L.shape[-2:]):
        raise ValueError('Stereo H and W must be multiples of 8 and at least 8')


def _regress_disparity(C, start, end):
    probabilities = F.softmax(-C, dim=1)
    candidates = torch.arange(start, end, device=C.device, dtype=C.dtype).view(1, -1, 1, 1)
    candidates = candidates.repeat(C.shape[0], 1, C.shape[2], C.shape[3])
    return torch.sum(probabilities * candidates, dim=1, keepdim=True)


class SCSNet(nn.Module):
    """Return four full-resolution disparity estimates, ending in SSR output.

    Inputs use ImageNet-normalized RGB channels. Disparities are measured in
    input-image pixels. Use eval() and torch.no_grad() for inference.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = SCSNetConfig() if config is None else config
        if not isinstance(self.config, SCSNetConfig):
            raise TypeError('config must be SCSNetConfig')
        cfg = self.config
        self.acfe = AdaptiveContextFeatureExtractor(cfg.c, cfg.sse_blocks)
        self.oscv = OutlierSuppressedCostVolume(cfg.G)
        self.cost_aggregation = nn.ModuleList(tuple(
            cost_aggregation(cfg.aggregation_layers, cfg.aggregation_channels * growth, cfg.G)
            for growth in cfg.aggregation_growth
        ))
        self.ssr = StructureSelectiveRefinement(cfg.ssr_channels)

    def forward(self, I_L, I_R):
        _validate_images(I_L, I_R)
        H, W = I_L.shape[-2:]
        features_L, features_R = self.acfe(I_L), self.acfe(I_R)
        predictions = ()
        for scale, (F_L, F_R) in enumerate(zip(features_L, features_R)):
            D = self.config.stage_search_ranges[scale]
            D_init = None if scale == 0 else F.interpolate(
                predictions[-1], size=F_L.shape[-2:], mode='bilinear',
                align_corners=False) * F_L.shape[2] / H
            V_cost = self.oscv(F_L, F_R, D, D_init)
            C = self.cost_aggregation[scale](V_cost).squeeze(1)
            d_hat = _regress_disparity(C, 0 if scale == 0 else -D + 1, D)
            d_hat = d_hat * H / d_hat.shape[2]
            d_hat = F.interpolate(d_hat, size=(H, W), mode='bilinear', align_corners=False)
            predictions = predictions + (d_hat if scale == 0 else d_hat + predictions[-1],)
        D_init = predictions[-1]
        delta_D = self.ssr(torch.cat((I_L, D_init), dim=1))
        return predictions + (D_init + delta_D,)
