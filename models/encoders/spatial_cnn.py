"""
==============================================================================
Spatial Encoder: 2D Residual Convolutional Neural Network
==============================================================================
Processes static geomorphology (DEM, Slope, TWI, Soils, Flow Accumulation) 
and instant surface condition maps.
Optimized for 32x32 km patch tensors.
Includes target layers for Grad-CAM explainability hooks.
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ResidualBlock2D(nn.Module):
    """Basic 2D Residual Block with BatchNorm and LeakyReLU."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


class SpatialCNNEncoder(nn.Module):
    """
    Spatial feature extractor for 32x32 patches.
    In channels: Static layers (DEM, Slope, TWI, FlowAcc, DistToRiver, Clay, Sand, BulkDensity, etc.)
    Output: Latent spatial embedding (B, out_dim) or spatial feature maps (B, C, H, W).
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_dim: int = 128,
        conv_channels: List[int] = [32, 64, 128],
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels[0]),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Stage 1 (32x32)
        self.stage1 = nn.Sequential(
            ResidualBlock2D(conv_channels[0], dropout=dropout),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1, bias=False), # -> 16x16
            nn.BatchNorm2d(conv_channels[1]),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Stage 2 (16x16)
        self.stage2 = nn.Sequential(
            ResidualBlock2D(conv_channels[1], dropout=dropout),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=2, padding=1, bias=False), # -> 8x8
            nn.BatchNorm2d(conv_channels[2]),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Final ResNet Stage for Grad-CAM hook
        self.target_conv = ResidualBlock2D(conv_channels[2], dropout=dropout)

        # Global Pooling and Linear Projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[2], out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor, return_feature_map: bool = False):
        """
        x: (B, in_channels, H=32, W=32)
        Returns:
            embedding: (B, out_dim)
            Optional feature_map: (B, C, H', W') for Grad-CAM
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        feature_map = self.target_conv(x)  # Target for Grad-CAM

        pooled = self.global_pool(feature_map).flatten(1)
        embedding = self.fc(pooled)

        if return_feature_map:
            return embedding, feature_map
        return embedding
