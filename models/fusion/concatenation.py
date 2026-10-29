"""
==============================================================================
Fusion Module 1: Plain Concatenation Fusion (Ablation Baseline)
==============================================================================
Direct feature concatenation of spatial and temporal embeddings.
Used as the standard baseline to benchmark against gated cross-modal attention.
==============================================================================
"""

import torch
import torch.nn as nn


class ConcatenationFusion(nn.Module):
    """
    Concatenates spatial and temporal latent vectors and applies linear projection.
    """

    def __init__(self, spatial_dim: int = 128, temporal_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, z_spatial: torch.Tensor, z_temporal: torch.Tensor) -> torch.Tensor:
        """
        z_spatial: (B, spatial_dim)
        z_temporal: (B, temporal_dim)
        Returns:
            z_fused: (B, out_dim)
        """
        concat = torch.cat([z_spatial, z_temporal], dim=1)
        return self.fc(concat)
