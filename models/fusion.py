"""
==============================================================================
PHASE 4: Adaptive Feature-Level Fusion (PlainConcatenation & GatedFusion)
==============================================================================
Scientific Directives Enforced:
1. Ablation Step A3 (Plain Concatenation vs Gated Fusion):
   - PlainConcatenation: Direct feature concatenation baseline [z_s; z_t].
   - GatedFusion: State-dependent gating mechanism learning dynamic weights g
     to adaptively balance spatial geomorphology and antecedent meteorology.
2. Squeeze-and-Excitation / Cross-Modal Gating:
   - Gate: g = sigmoid(W_g [z_s, z_t] + b_g) in [0, 1]^D
   - Fused: z_fused = g * proj_s(z_s) + (1 - g) * proj_t(z_t)
   - Exposes learned gate values for XAI inspection and modal balance tracking.
==============================================================================
"""

import os
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

# Ensure submodules inside models/fusion/ remain discoverable
__path__ = [os.path.join(os.path.dirname(__file__), "fusion")]


class PlainConcatenation(nn.Module):
    """
    Plain Feature Concatenation for Spatiotemporal Fusion (Ablation Baseline).
    
    Flattens spatial and temporal latent vectors and concatenates them into
    a unified spatiotemporal representation vector.
    """

    def __init__(
        self,
        spatial_dim: int = 128,
        temporal_dim: int = 128,
        out_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.concat_dim = spatial_dim + temporal_dim
        self.out_dim = out_dim

        # Projection block with normalization and regularization
        self.projection = nn.Sequential(
            nn.Linear(self.concat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for projection weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        z_spatial: torch.Tensor,
        z_temporal: torch.Tensor,
        return_raw_concat: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Concatenates spatial and temporal feature vectors.
        
        Args:
            z_spatial: (B, spatial_dim) or (B, spatial_dim, 1, 1)
            z_temporal: (B, temporal_dim) or (B, temporal_dim, 1)
            return_raw_concat: If True, returns (z_fused, raw_concat)
        
        Returns:
            z_fused: (B, out_dim)
        """
        if z_spatial.dim() > 2:
            z_spatial = torch.flatten(z_spatial, start_dim=1)

        if z_temporal.dim() > 2:
            z_temporal = torch.flatten(z_temporal, start_dim=1)

        assert z_spatial.shape[0] == z_temporal.shape[0], (
            f"Batch size mismatch in PlainConcatenation: spatial ({z_spatial.shape[0]}) vs temporal ({z_temporal.shape[0]})."
        )
        assert z_spatial.shape[1] == self.spatial_dim, (
            f"Spatial dimension mismatch: expected {self.spatial_dim}, got {z_spatial.shape[1]}."
        )
        assert z_temporal.shape[1] == self.temporal_dim, (
            f"Temporal dimension mismatch: expected {self.temporal_dim}, got {z_temporal.shape[1]}."
        )

        raw_concat = torch.cat([z_spatial, z_temporal], dim=1)
        z_fused = self.projection(raw_concat)

        if return_raw_concat:
            return z_fused, raw_concat
        return z_fused


class GatedFusion(nn.Module):
    """
    Adaptive Gated Cross-Modal Feature Fusion.
    
    Learns input-dependent gating weights g in [0, 1] to adaptively weight the
    spatial and temporal representations:
        h_s = LayerNorm(Linear(z_spatial))
        h_t = LayerNorm(Linear(z_temporal))
        g = Sigmoid(Linear(LeakyReLU(Linear([z_spatial, z_temporal]))))
        z_fused = g * h_s + (1 - g) * h_t
    
    Higher g (> 0.5) implies geomorphic/terrain dominance (e.g. flat basins, high TWI).
    Lower g (< 0.5) implies meteorological/dynamic storm dominance (e.g. extreme rainfall).
    """

    def __init__(
        self,
        spatial_dim: int = 128,
        temporal_dim: int = 128,
        out_dim: int = 128,
        gate_hidden_dim: Optional[int] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.out_dim = out_dim
        gate_hidden = gate_hidden_dim or out_dim

        # 1. Individual Modality Projection & Normalization
        self.proj_spatial = nn.Sequential(
            nn.Linear(spatial_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.proj_temporal = nn.Sequential(
            nn.Linear(temporal_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 2. State-Dependent Gating Network (Squeeze-and-Excitation Style)
        self.gate_network = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, gate_hidden),
            nn.BatchNorm1d(gate_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, out_dim),
            nn.Sigmoid()  # Learnable gate g in [0, 1]
        )

        # 3. Post-Fusion Regularization and Refinement
        self.refinement = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier Uniform initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        z_spatial: torch.Tensor,
        z_temporal: torch.Tensor,
        return_gate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Executes adaptive gated feature fusion.
        
        Args:
            z_spatial: (B, spatial_dim)
            z_temporal: (B, temporal_dim)
            return_gate: If True, returns (z_fused, gate_weights)
        
        Returns:
            z_fused: (B, out_dim) fused representation
            gate_weights: (B, out_dim) learned gating values
        """
        if z_spatial.dim() > 2:
            z_spatial = torch.flatten(z_spatial, start_dim=1)

        if z_temporal.dim() > 2:
            z_temporal = torch.flatten(z_temporal, start_dim=1)

        assert z_spatial.shape[0] == z_temporal.shape[0], (
            f"Batch size mismatch in GatedFusion: spatial ({z_spatial.shape[0]}) vs temporal ({z_temporal.shape[0]})."
        )
        assert z_spatial.shape[1] == self.spatial_dim, (
            f"Spatial dimension mismatch: expected {self.spatial_dim}, got {z_spatial.shape[1]}."
        )
        assert z_temporal.shape[1] == self.temporal_dim, (
            f"Temporal dimension mismatch: expected {self.temporal_dim}, got {z_temporal.shape[1]}."
        )

        # Project modalities into aligned representation space
        h_s = self.proj_spatial(z_spatial)  # (B, out_dim)
        h_t = self.proj_temporal(z_temporal)  # (B, out_dim)

        # Compute dynamic gating weights conditioned on joint state
        joint_state = torch.cat([z_spatial, z_temporal], dim=1)  # (B, spatial_dim + temporal_dim)
        gate = self.gate_network(joint_state)  # (B, out_dim) in [0, 1]

        # Convex linear combination of modalities
        fused = gate * h_s + (1.0 - gate) * h_t  # (B, out_dim)

        # Refinement and output
        z_fused = self.refinement(fused)

        if return_gate:
            return z_fused, gate
        return z_fused


# Backward compatibility aliases
ConcatenationFusion = PlainConcatenation
GatedCrossModalAttentionFusion = GatedFusion
