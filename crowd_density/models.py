from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell for spatio-temporal feature modeling."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = x.shape

        if state is None:
            h_prev = torch.zeros(batch, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(batch, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = state

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    """Minimal ConvLSTM for processing frame feature sequences."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        outputs: List[torch.Tensor] = []
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        for t in range(x.size(1)):
            h, c = self.cell(x[:, t], state)
            state = (h, c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class DensityHead(nn.Module):
    """Decoder head mapping features to density maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        x = self.layers(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class TemporalSpatialCrowdNet(nn.Module):
    """
    Real-time oriented temporal + spatial crowd density network.

    Input:  (B, T, 3, H, W)
    Output: (B, T, 1, H, W)
    """

    def __init__(
        self,
        feature_channels: int = 96,
        lstm_hidden_channels: int = 96,
        use_pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT if use_pretrained_backbone else None
        backbone = mobilenet_v3_small(weights=weights)
        self.spatial_encoder = backbone.features

        self.channel_adapter = nn.Conv2d(576, feature_channels, kernel_size=1)
        self.temporal_model = ConvLSTM(input_channels=feature_channels, hidden_channels=lstm_hidden_channels)
        self.density_head = DensityHead(channels=lstm_hidden_channels)

    def forward(self, video_clip: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = video_clip.shape
        if c != 3:
            raise ValueError(f"Expected RGB input with 3 channels, got {c}")

        frame_features = []
        for step in range(t):
            feat = self.spatial_encoder(video_clip[:, step])
            feat = self.channel_adapter(feat)
            frame_features.append(feat)

        features_seq = torch.stack(frame_features, dim=1)  # (B, T, C, H', W')
        temporal_features = self.temporal_model(features_seq)  # (B, T, C, H', W')

        density_maps = []
        for step in range(t):
            dm = self.density_head(temporal_features[:, step], output_size=(h, w))
            density_maps.append(dm)

        return torch.stack(density_maps, dim=1)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OptimizedSpatialCrowdNet(nn.Module):
    """
    Optimized spatial model for single-frame crowd density estimation.
    Lightweight and suitable for real-time inference.
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        self.enc1 = DepthwiseSeparableConv(3, c)
        self.enc2 = DepthwiseSeparableConv(c, c * 2, stride=2)
        self.enc3 = DepthwiseSeparableConv(c * 2, c * 4, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 4, c * 4, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(c * 4 + c * 2, c * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(c * 2 + c, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Sequential(nn.Conv2d(c, 1, kernel_size=1), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out_conv(d1)
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


@dataclass
class CrowdLossWeights:
    pixel_mse: float = 1.0
    count_l1: float = 0.01


class CrowdDensityLoss(nn.Module):
    """Combined density-map regression + total count consistency loss."""

    def __init__(self, weights: CrowdLossWeights = CrowdLossWeights()) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)

        pred_count = pred.flatten(start_dim=2).sum(dim=-1)
        target_count = target.flatten(start_dim=2).sum(dim=-1)
        count_l1 = F.l1_loss(pred_count, target_count)

        return self.weights.pixel_mse * mse + self.weights.count_l1 * count_l1
