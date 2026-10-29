"""
==============================================================================
Fusion Module 2: Gated Cross-Modal Attention Fusion
==============================================================================
Dynamically balances static geomorphology (spatial terrain susceptibility) 
and antecedent hydrometeorological dynamics (temporal storm sequences).
Gate: g = sigmoid(W_g [z_s, z_t] + b_g)
Fused: z_fused = g * W_s(z_s) + (1 - g) * W_t(z_t)
==============================================================================
"""

import torch
import torch.nn as nn
from typing import Tuple


class GatedCrossModalAttentionFusion(nn.Module):
    """
    Gated cross-modal fusion with dynamic modal balance.
    """

    def __init__(self, spatial_dim: int = 128, temporal_dim: int = 128, fusion_dim: int = 128):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.fusion_dim = fusion_dim

        # Modality projections
        self.proj_spatial = nn.Sequential(
            nn.Linear(spatial_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.proj_temporal = nn.Sequential(
            nn.Linear(temporal_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Cross-modal Gating Mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, fusion_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )

        # Post-fusion refinement
        self.out_net = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(0.2)
        )

    def forward(self, z_spatial: torch.Tensor, z_temporal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_spatial: (B, spatial_dim)
        z_temporal: (B, temporal_dim)
        Returns:
            z_fused: (B, fusion_dim)
            gate: (B, fusion_dim) Gating weights (higher = spatial dominant, lower = temporal dominant)
        """
        h_s = self.proj_spatial(z_spatial)
        h_t = self.proj_temporal(z_temporal)

        concat = torch.cat([z_spatial, z_temporal], dim=1)
        gate = self.gate_net(concat)

        # Convex combination of modal streams
        fused = gate * h_s + (1.0 - gate) * h_t
        out = self.out_net(fused)
        return out, gate
