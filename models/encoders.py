"""
==============================================================================
PHASE 3: Modality-Specific Encoders (SpatialCNN & TemporalGRU)
==============================================================================
Scientific Directives Enforced:
1. SpatialCNN:
   - Processes 32x32 spatial patches from the 1 km aligned grid.
   - Accepts multi-channel static rasters (DEM, slope, TWI, soils, water)
     and dynamic image inputs (Sentinel-1 SAR, NDVI).
   - Enforces strict shape assertion: (H, W) == (32, 32).
   - Provides exposed target convolutional layer for Grad-CAM explainability.
   - NO Swin Transformers or GNNs (strictly establishes CNN baseline).
2. TemporalGRU:
   - Causal Gated Recurrent Unit (unidirectional, bidirectional=False).
   - Processes antecedent climate sequences (precipitation, temp, PET, SM).
   - Automatically handles 3D (B, T, C) and 5D (B, T, C, 32, 32) tensors.
   - Enforces causal temporal integrity (strictly < t, zero future leakage).
==============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict, Any

# Ensure submodules inside models/encoders/ remain discoverable
__path__ = [os.path.join(os.path.dirname(__file__), "encoders")]


class ResidualBlock2D(nn.Module):
    """
    Standard 2D Residual Block with BatchNorm, LeakyReLU, and Dropout.
    Used within SpatialCNN stages to preserve gradient flow and prevent overfitting.
    """

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.drop = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.act2(out + residual)
        return out


class SpatialCNN(nn.Module):
    """
    Compact 2D Convolutional Neural Network for 32x32 spatial patch representation.
    
    Accepts:
      - Static rasters (DEM, slope, TWI, flow accumulation, river distance, soils)
      - Dynamic image inputs (Sentinel-1 SAR VH/VV, NDVI)
    
    Shape Guarantee:
      Input spatial dimensions MUST be exactly (32, 32).
    """

    def __init__(
        self,
        static_channels: int = 8,
        dynamic_channels: int = 2,
        out_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.static_channels = static_channels
        self.dynamic_channels = dynamic_channels
        self.in_channels = static_channels + dynamic_channels
        self.out_dim = out_dim

        # 1. Input Stem (32x32 -> 32x32, 32 channels)
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout * 0.5)
        )

        # 2. Stage 1 (32x32 -> 16x16, 64 channels)
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout),
            ResidualBlock2D(64, dropout=dropout)
        )

        # 3. Stage 2 (16x16 -> 8x8, 128 channels)
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout),
            ResidualBlock2D(128, dropout=dropout)
        )

        # 4. Target Layer for Grad-CAM explainability (8x8 -> 8x8, 128 channels)
        self.target_conv = ResidualBlock2D(128, dropout=dropout)

        # 5. Global Adaptive Average Pooling & Dense Projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming Normal initialization for convs, Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_images: Optional[torch.Tensor] = None,
        return_feature_map: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for 32x32 spatial patch.
        
        Args:
            static_features: (B, C_s, 32, 32)
            dynamic_images:  Optional (B, C_d, 32, 32) or (B, T, C_d, 32, 32)
            return_feature_map: If True, returns (embedding, target_conv_feature_map) for Grad-CAM
        
        Returns:
            embedding: (B, out_dim)
            feature_map: (B, 128, 8, 8) if return_feature_map=True
        """
        # Shape assertion 1: Verify 32x32 spatial patch dimensions
        assert static_features.shape[-2:] == (32, 32), (
            f"SpatialCNN strictly requires 32x32 patches. Got spatial dimensions {static_features.shape[-2:]}."
        )

        if dynamic_images is not None:
            # Handle 5D dynamic tensor (B, T, C_d, 32, 32) -> take latest pre-event frame t-1
            if dynamic_images.dim() == 5:
                dynamic_images = dynamic_images[:, -1, :, :, :]

            # Shape assertion 2: Verify dynamic image dimensions match
            assert dynamic_images.shape[-2:] == (32, 32), (
                f"Dynamic images must be 32x32. Got {dynamic_images.shape[-2:]}."
            )
            assert dynamic_images.shape[0] == static_features.shape[0], (
                f"Batch size mismatch: static {static_features.shape[0]} vs dynamic {dynamic_images.shape[0]}."
            )

            # Concatenate along channel dimension
            x = torch.cat([static_features, dynamic_images], dim=1)
        else:
            x = static_features

        # Assert total channels match configuration
        assert x.shape[1] == self.in_channels, (
            f"Input channel mismatch: expected {self.in_channels} (static={self.static_channels} + dynamic={self.dynamic_channels}), got {x.shape[1]}."
        )

        # Forward through CNN stages
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        feature_map = self.target_conv(h)  # (B, 128, 8, 8)

        # Pool and project
        pooled = self.global_pool(feature_map)  # (B, 128, 1, 1)
        flattened = torch.flatten(pooled, 1)    # (B, 128)
        embedding = self.projection(flattened)  # (B, out_dim)

        if return_feature_map:
            return embedding, feature_map
        return embedding


class TemporalGRU(nn.Module):
    """
    Causal Gated Recurrent Unit (GRU) for Antecedent Hydrometeorological Sequences.
    
    Processes climate sequences (rainfall, temperature, humidity, PET, soil moisture)
    moving forward in time [t - lookback, t).
    
    Strictly Causal:
      bidirectional=False guarantees no future temporal leakage.
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # Causal unidirectional GRU
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False  # ZERO future leakage
        )

        # Temporal attention mechanism to score antecedent time step saliency
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Projection to standard latent dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier Uniform initialization for GRU and linear projections."""
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
        for m in [self.attention, self.projection]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for temporal climate sequence.
        
        Args:
            x: Can be:
               - 3D: (B, T, in_features)
               - 5D: (B, T, in_features, 32, 32) from spatial patch dataset
            return_attention: If True, returns (embedding, attn_weights)
        
        Returns:
            embedding: (B, out_dim)
            attn_weights: (B, T, 1) if return_attention=True
        """
        # Automatically handle 5D spatiotemporal patch input by spatial mean pooling
        if x.dim() == 5:
            # Shape assertion on spatial dimensions
            assert x.shape[-2:] == (32, 32), (
                f"TemporalGRU 5D input must have 32x32 spatial dimensions, got {x.shape[-2:]}."
            )
            # Pool spatial dimensions (H, W) -> (B, T, in_features)
            x = x.mean(dim=[-2, -1])

        # Assert tensor is now 3D (B, T, in_features)
        assert x.dim() == 3, f"Expected 3D tensor (B, T, C), got {x.dim()}D tensor with shape {x.shape}."

        B, T, C = x.shape
        assert T > 0, "Temporal sequence length T must be greater than 0."
        assert C == self.in_features, (
            f"Input feature mismatch: expected {self.in_features}, got {C}."
        )

        # Pass through causal GRU
        gru_out, h_n = self.gru(x)  # gru_out: (B, T, hidden_dim)

        # Attentive pooling across time steps
        attn_logits = self.attention(gru_out)  # (B, T, 1)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, T, 1)
        context = torch.sum(attn_weights * gru_out, dim=1)  # (B, hidden_dim)

        # Project to target latent dimension
        embedding = self.projection(context)  # (B, out_dim)

        if return_attention:
            return embedding, attn_weights
        return embedding


# Backward compatibility aliases
SpatialCNNEncoder = SpatialCNN
TemporalGRUEncoder = TemporalGRU
