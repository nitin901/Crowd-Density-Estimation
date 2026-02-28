import torch

from crowd_density import CrowdDensityLoss, OptimizedSpatialCrowdNet, TemporalSpatialCrowdNet


def test_temporal_model_output_shape() -> None:
    model = TemporalSpatialCrowdNet(use_pretrained_backbone=False)
    x = torch.randn(1, 3, 3, 128, 128)
    y = model(x)
    assert y.shape == (1, 3, 1, 128, 128)


def test_spatial_model_output_shape() -> None:
    model = OptimizedSpatialCrowdNet(base_channels=16)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    assert y.shape == (1, 1, 128, 128)


def test_combined_loss_non_negative() -> None:
    loss_fn = CrowdDensityLoss()
    pred = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64)
    loss = loss_fn(pred, target)
    assert float(loss) >= 0.0
