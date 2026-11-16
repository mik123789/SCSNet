"""Numerical contracts of the inference-only SCSNet implementation."""
from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn.functional as F

from scsnet.config import SCSNetConfig
from scsnet.cost_volume import (
    OutlierSuppressedCostVolume,
    robust_group_cost,
    warp_right_feature,
)
from scsnet.features import AdaptiveContextFeatureExtractor, ScaleSelectiveEncoder
from scsnet.model import SCSNet


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    return SCSNetConfig(c=4, G=4, stage_search_ranges=(3, 2, 2),
                        sse_blocks=1, aggregation_layers=0,
                        aggregation_channels=2, aggregation_growth=(1, 1, 1),
                        ssr_channels=4)


def test_config_defaults_and_immutability():
    config = SCSNetConfig()
    assert config.c == config.G == 48
    assert config.stage_search_ranges == (24, 3, 3)
    with pytest.raises(FrozenInstanceError):
        config.c = 8


@pytest.mark.parametrize('changes', [
    {'c': 0}, {'G': 0}, {'c': 5, 'G': 3},
    {'stage_search_ranges': (24, 3)},
    {'stage_search_ranges': (24, 0, 3)},
    {'sse_blocks': 0}, {'aggregation_layers': -1},
    {'aggregation_channels': 0}, {'aggregation_growth': (6, 3)},
    {'aggregation_growth': (6, 0, 3)}, {'ssr_channels': 0},
    {'c': True}, {'G': 2.0}, {'aggregation_layers': False},
    {'stage_search_ranges': [24, 3, 3]},
])
def test_invalid_config_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        SCSNetConfig(**changes)


def test_oscv_modulates_each_channel_before_averaging():
    F_L = torch.tensor([0., 1., 2., 3.]).view(1, 4, 1, 1)
    F_R = torch.zeros_like(F_L)
    V_cost = robust_group_cost(F_L, F_R, 2)
    expected = torch.stack((torch.log1p(torch.tensor(1.)) / 2,
                            (torch.log1p(torch.tensor(4.)) +
                             torch.log1p(torch.tensor(9.))) / 2)).view(1, 2, 1, 1)
    torch.testing.assert_close(V_cost, expected)
    assert not torch.allclose(V_cost, torch.log1p(F_L.square().view(1, 2, 2, 1, 1).mean(2)))


@pytest.mark.parametrize('shape_left,shape_right,G', [
    ((1, 3, 2, 2), (1, 3, 2, 2), 2),
    ((1, 4, 2, 2), (1, 4, 2, 3), 2),
    ((4, 2, 2), (4, 2, 2), 2),
    ((1, 4, 2, 2), (1, 4, 2, 2), 0),
])
def test_oscv_invalid_features(shape_left, shape_right, G):
    with pytest.raises((ValueError, TypeError)):
        robust_group_cost(torch.zeros(shape_left), torch.zeros(shape_right), G)


def test_initial_volume_has_zero_invalid_border_and_wide_search():
    F_L = torch.ones(1, 2, 1, 3)
    F_R = torch.zeros_like(F_L)
    V_cost = OutlierSuppressedCostVolume(1)(F_L, F_R, 5)
    expected = torch.tensor([[[[[1., 1., 1.]], [[0., 1., 1.]],
                                [[0., 0., 1.]], [[0., 0., 0.]],
                                [[0., 0., 0.]]]]]) * torch.log(torch.tensor(2.))
    torch.testing.assert_close(V_cost, expected)


def test_warp_retains_legacy_coordinate_convention():
    F_R = torch.arange(12.).view(1, 1, 3, 4)
    d_hat = torch.full((1, 1, 3, 4), .5)
    x = torch.arange(4.).view(1, 4).repeat(3, 1)
    y = torch.arange(3.).view(3, 1).repeat(1, 4)
    grid = torch.stack((2 * (x - .5) / 3 - 1, 2 * y / 2 - 1), dim=-1)[None]
    expected = F.grid_sample(F_R, grid, align_corners=False, padding_mode='zeros')
    torch.testing.assert_close(warp_right_feature(F_R, d_hat), expected)


def test_residual_volume_candidate_order_and_no_input_changes():
    F_L = torch.randn(2, 4, 3, 4)
    F_R = torch.randn_like(F_L)
    D_init = torch.ones(2, 1, 3, 4)
    originals = tuple(x.clone() for x in (F_L, F_R, D_init))
    V_cost = OutlierSuppressedCostVolume(2)(F_L, F_R, 2, D_init)
    expected = torch.stack(tuple(robust_group_cost(
        F_L, warp_right_feature(F_R, D_init - k), 2)
        for k in (-1, 0, 1)), dim=2)
    torch.testing.assert_close(V_cost, expected)
    for actual, original in zip((F_L, F_R, D_init), originals):
        assert torch.equal(actual, original)


@pytest.mark.parametrize('c_in,c_out', [(4, 4), (4, 8)])
def test_sse_uses_direct_unconstrained_omega_and_residual(c_in, c_out):
    encoder = ScaleSelectiveEncoder(c_in, c_out).eval()
    X = torch.randn(2, c_in, 4, 8)
    with torch.no_grad():
        encoder.omega.copy_(torch.tensor([-2., 0., 3.]))
        X_red = encoder.reduce(X)
        weighted = sum(branch(X_red) * encoder.omega[i]
                       for i, branch in enumerate(encoder.branches))
        shortcut = X if encoder.shortcut is None else encoder.shortcut(X)
        expected = torch.relu(encoder.project(weighted) + shortcut)
        actual = encoder(X)
    torch.testing.assert_close(actual, expected)


def test_acfe_returns_coarse_to_fine_features():
    acfe = AdaptiveContextFeatureExtractor(c=4, sse_blocks=1).eval()
    with torch.no_grad():
        features = acfe(torch.randn(1, 3, 16, 24))
    assert isinstance(features, tuple)
    assert tuple(x.shape for x in features) == (
        (1, 32, 2, 3), (1, 16, 4, 6), (1, 8, 8, 12))


def test_four_stage_cpu_inference_preserves_inputs():
    model = SCSNet(tiny_config()).eval()
    I_L = torch.randn(1, 3, 16, 24)
    I_R = torch.randn_like(I_L)
    originals = (I_L.clone(), I_R.clone())
    with torch.no_grad():
        predictions = model(I_L, I_R)
    assert isinstance(predictions, tuple) and len(predictions) == 4
    assert all(D.shape == (1, 1, 16, 24) for D in predictions)
    assert all(torch.isfinite(D).all() for D in predictions)
    assert torch.equal(I_L, originals[0]) and torch.equal(I_R, originals[1])
    assert 'acfe.block0.1.1.omega' in model.state_dict()
    assert not any(name.endswith('.weights') for name in model.state_dict())


@pytest.mark.parametrize('left,right', [
    (torch.zeros(1, 3, 15, 24), torch.zeros(1, 3, 15, 24)),
    (torch.zeros(1, 3, 16, 24), torch.zeros(1, 3, 16, 16)),
    (torch.zeros(1, 1, 16, 24), torch.zeros(1, 1, 16, 24)),
    (torch.zeros(3, 16, 24), torch.zeros(3, 16, 24)),
    (torch.zeros(1, 3, 16, 24, dtype=torch.int64),
     torch.zeros(1, 3, 16, 24, dtype=torch.int64)),
])
def test_model_rejects_invalid_input_pairs(left, right):
    with pytest.raises((ValueError, TypeError)):
        SCSNet(tiny_config()).eval()(left, right)


def test_public_boundaries_reject_non_tensor_and_mixed_dtype():
    F_L = torch.zeros(1, 4, 2, 2)
    with pytest.raises(TypeError, match='torch tensors'):
        robust_group_cost(None, F_L, 2)
    with pytest.raises(TypeError, match='floating'):
        robust_group_cost(F_L.long(), F_L.long(), 2)
    with pytest.raises(ValueError, match='dtype'):
        robust_group_cost(F_L, F_L.double(), 2)
    with pytest.raises(ValueError, match='nonempty'):
        robust_group_cost(F_L[..., :0], F_L[..., :0], 2)
    with pytest.raises(TypeError, match='SCSNetConfig'):
        SCSNet({})
    model = SCSNet(tiny_config()).eval()
    with pytest.raises(TypeError, match='torch tensors'):
        model(None, None)
    with pytest.raises(ValueError, match='dtype'):
        model(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8).double())


def test_cost_volume_rejects_invalid_search_and_disparity():
    F_L = torch.zeros(1, 4, 2, 2)
    with pytest.raises(ValueError, match='G'):
        OutlierSuppressedCostVolume(0)
    oscv = OutlierSuppressedCostVolume(2)
    with pytest.raises(ValueError, match='range'):
        oscv(F_L, F_L, 0)
    with pytest.raises(ValueError, match='Initial disparity'):
        oscv(F_L, F_L, 2, torch.zeros(1, 2, 2, 2))
    with pytest.raises(ValueError, match='Warp disparity'):
        warp_right_feature(F_L, torch.zeros(1, 2, 2, 2))
    with pytest.raises(ValueError, match='dtype'):
        warp_right_feature(F_L, torch.zeros(1, 1, 2, 2).double())
