"""Outlier-Suppressed Cost Volume (OSCV) and legacy-compatible warping."""
import torch
from torch import nn
from torch.nn import functional as F


def _validate_features(F_L, F_R, G):
    if type(G) is not int or G <= 0:
        raise ValueError('G must be a positive integer')
    if not isinstance(F_L, torch.Tensor) or not isinstance(F_R, torch.Tensor):
        raise TypeError('Features must be torch tensors')
    if F_L.ndim != 4 or F_L.shape != F_R.shape:
        raise ValueError('Left and right features must have matching BCHW shapes')
    if not F_L.is_floating_point() or not F_R.is_floating_point():
        raise TypeError('Features must be floating-point tensors')
    if F_L.device != F_R.device or F_L.dtype != F_R.dtype:
        raise ValueError('Left and right features must share device and dtype')
    if F_L.shape[1] % G:
        raise ValueError('Feature channel count must be divisible by G')
    if any(size == 0 for size in F_L.shape):
        raise ValueError('Feature dimensions must be nonempty')


def robust_group_cost(F_L, F_R, G):
    """Compute mean_c log(1 + E_c), with E_c=(F_L-F_R)^2, within groups."""
    _validate_features(F_L, F_R, G)
    B, C, H, W = F_L.shape
    difference = (F_L - F_R).reshape(B, G, C // G, H, W)
    return torch.log1p(difference ** 2).mean(dim=2).contiguous()


def warp_right_feature(F_R, d_hat):
    """Warp using original normalization and align_corners=False.

    The W-1/H-1 normalization is retained for checkpoint compatibility even
    though grid_sample uses align_corners=False in the original implementation.
    """
    B, _, H, W = F_R.shape
    if d_hat.shape != (B, 1, H, W):
        raise ValueError('Warp disparity must have shape (B, 1, H, W)')
    if d_hat.device != F_R.device or d_hat.dtype != F_R.dtype:
        raise ValueError('Warp disparity and features must share device and dtype')
    x = torch.arange(W, device=F_R.device, dtype=F_R.dtype).view(1, 1, W)
    y = torch.arange(H, device=F_R.device, dtype=F_R.dtype).view(1, H, 1)
    grid_x = 2.0 * (x.expand(B, H, W) - d_hat[:, 0]) / max(W - 1, 1) - 1.0
    grid_y = 2.0 * y.expand(B, H, W) / max(H - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(F_R, grid, mode='bilinear', padding_mode='zeros',
                         align_corners=False)


class OutlierSuppressedCostVolume(nn.Module):
    """Construct V_cost in B,G,D,H,W order without changing input tensors."""
    def __init__(self, G=48):
        super().__init__()
        if type(G) is not int or G <= 0:
            raise ValueError('G must be a positive integer')
        self.G = G

    def forward(self, F_L, F_R, D, D_init=None):
        _validate_features(F_L, F_R, self.G)
        if type(D) is not int or D <= 0:
            raise ValueError('Disparity search range D must be a positive integer')
        if D_init is None:
            return self._initial_volume(F_L, F_R, D)
        return self._residual_volume(F_L, F_R, D, D_init)

    def _initial_volume(self, F_L, F_R, D):
        B, _, H, W = F_L.shape
        costs = tuple(
            F_L.new_zeros(B, self.G, H, W) if d >= W else
            F.pad(robust_group_cost(F_L[..., d:],
                                   F_R if d == 0 else F_R[..., :-d], self.G),
                  (d, 0))
            for d in range(D)
        )
        return torch.stack(costs, dim=2).contiguous()

    def _residual_volume(self, F_L, F_R, D, D_init):
        B, C, H, W = F_L.shape
        if D_init.shape != (B, 1, H, W):
            raise ValueError('Initial disparity must have shape (B, 1, H, W)')
        candidates = 2 * D - 1
        d_hat = D_init[:, None].repeat(1, candidates, 1, 1, 1).reshape(-1, 1, H, W)
        shifts = torch.arange(-D + 1, D, device=F_L.device,
                              dtype=F_L.dtype).repeat(B)[:, None, None, None]
        batch_L = F_L[:, None].repeat(1, candidates, 1, 1, 1).reshape(-1, C, H, W)
        batch_R = F_R[:, None].repeat(1, candidates, 1, 1, 1).reshape(-1, C, H, W)
        costs = robust_group_cost(batch_L, warp_right_feature(batch_R, d_hat - shifts), self.G)
        return costs.reshape(B, candidates, self.G, H, W).permute(0, 2, 1, 3, 4).contiguous()
