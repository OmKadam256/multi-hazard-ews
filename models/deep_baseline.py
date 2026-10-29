"""
==============================================================================
PHASE 3: Deep Baseline Model Architecture (CNN + GRU + Plain Concatenation)
==============================================================================
Scientific Directives Enforced:
1. CNN_GRU_Baseline Model:
   - Modality-specific encoders: SpatialCNN (32x32 terrain rasters & SAR) 
     and TemporalGRU (causal antecedent climate sequences).
   - Fusion layer: PlainConcatenation (Ablation step A3).
   - Flood Head: Binary classification with Sigmoid activation.
   - Drought Head: Continuous future SPEI regression.
2. Leakage & Shape Guarantees:
   - Enforces strict 32x32 patch spatial dimensions.
   - Enforces causal sequence length T > 0.
   - Ensures strict batch alignment across modalities.
3. Hardware Ready:
   - DataParallel / DDP compatible.
   - Automatic Mixed Precision (torch.cuda.amp) ready.
==============================================================================
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union, Any

from models.encoders import SpatialCNN, TemporalGRU
from models.fusion import PlainConcatenation


class CNN_GRU_Baseline(nn.Module):
    """
    Deep Spatiotemporal Baseline Model (CNN + GRU + Plain Concatenation).
    
    Combines:
      - SpatialCNN for 32x32 geomorphology & surface condition patches
      - TemporalGRU for antecedent hydrometeorology sequences
      - PlainConcatenation for feature fusion
      - Flood Head (Sigmoid inundation probability)
      - Drought Head (Linear SPEI index regression)
    """

    def __init__(
        self,
        static_channels: int = 8,
        dynamic_channels: int = 2,
        climate_features: int = 3,
        spatial_dim: int = 128,
        temporal_dim: int = 128,
        fused_dim: int = 128,
        temporal_hidden_dim: int = 64,
        temporal_num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.static_channels = static_channels
        self.dynamic_channels = dynamic_channels
        self.climate_features = climate_features
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.fused_dim = fused_dim

        # 1. Spatial CNN Encoder (32x32 patches)
        self.spatial_encoder = SpatialCNN(
            static_channels=static_channels,
            dynamic_channels=dynamic_channels,
            out_dim=spatial_dim,
            dropout=dropout
        )

        # 2. Temporal GRU Encoder (causal sequence < t)
        self.temporal_encoder = TemporalGRU(
            in_features=climate_features,
            hidden_dim=temporal_hidden_dim,
            out_dim=temporal_dim,
            num_layers=temporal_num_layers,
            dropout=dropout
        )

        # 3. Plain Concatenation Fusion Module
        self.fusion = PlainConcatenation(
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
            out_dim=fused_dim,
            dropout=dropout
        )

        # 4. Baseline Flood Output Head (Sigmoid Inundation Probability)
        self.flood_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 1)  # Raw logit
        )

        # 5. Baseline Drought Output Head (Linear SPEI Regression)
        self.drought_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 1)  # Continuous SPEI prediction
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming and Xavier initialization for stable deep baseline convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
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
        climate_sequence: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass executing end-to-end spatiotemporal inference.
        
        Args:
            static_features:  (B, C_s, 32, 32)
            dynamic_images:   (B, C_d, 32, 32) or (B, T, C_d, 32, 32)
            climate_sequence: (B, T, C_c) or (B, T, C_c, 32, 32)
            return_dict:      If True, returns comprehensive dictionary
        
        Returns:
            Dictionary with keys:
              - 'flood_prob': Inundation probability [0, 1] (B, 1)
              - 'flood_logit': Raw logit for Focal/BCE loss (B, 1)
              - 'spei_pred': Continuous future SPEI prediction (B, 1)
              - 'fused_embedding': Unified spatiotemporal vector (B, fused_dim)
              - 'spatial_embedding': Spatial vector (B, spatial_dim)
              - 'temporal_embedding': Temporal vector (B, temporal_dim)
              - 'temporal_attention': Time-step attention weights (B, T, 1)
              - 'spatial_feature_map': CNN stage 2 feature map for Grad-CAM (B, 128, 8, 8)
        """
        # ======================================================================
        # 1. SHAPE & INTEGRITY ASSERTIONS
        # ======================================================================
        assert static_features.dim() == 4, (
            f"static_features must be 4D (B, C, H, W). Got {static_features.dim()}D tensor."
        )
        assert static_features.shape[-2:] == (32, 32), (
            f"Spatial patch dimensions must be strictly 32x32. Got {static_features.shape[-2:]}."
        )

        B = static_features.shape[0]

        if dynamic_images is not None:
            assert dynamic_images.shape[0] == B, (
                f"Batch size mismatch between static ({B}) and dynamic ({dynamic_images.shape[0]})."
            )
            assert dynamic_images.shape[-2:] == (32, 32), (
                f"dynamic_images spatial patch must be 32x32. Got {dynamic_images.shape[-2:]}."
            )

        assert climate_sequence is not None, "climate_sequence is required for CNN_GRU_Baseline."
        assert climate_sequence.shape[0] == B, (
            f"Batch size mismatch between static ({B}) and climate ({climate_sequence.shape[0]})."
        )
        assert climate_sequence.dim() in [3, 5], (
            f"climate_sequence must be 3D (B, T, C) or 5D (B, T, C, 32, 32). Got {climate_sequence.dim()}D."
        )
        assert climate_sequence.shape[1] > 0, "Temporal climate sequence length T must be > 0."

        # ======================================================================
        # 2. MODALITY ENCODING
        # ======================================================================
        # Spatial encoding with Grad-CAM feature map preservation
        z_spatial, spatial_fmap = self.spatial_encoder(
            static_features=static_features,
            dynamic_images=dynamic_images,
            return_feature_map=True
        )

        # Temporal causal GRU encoding with attention saliency
        z_temporal, temporal_attn = self.temporal_encoder(
            x=climate_sequence,
            return_attention=True
        )

        # ======================================================================
        # 3. FEATURE-LEVEL FUSION (Plain Concatenation)
        # ======================================================================
        z_fused, raw_concat = self.fusion(
            z_spatial=z_spatial,
            z_temporal=z_temporal,
            return_raw_concat=True
        )

        # ======================================================================
        # 4. OUTPUT HEADS
        # ======================================================================
        # Flood head
        flood_logit = self.flood_head(z_fused)          # (B, 1)
        flood_prob = torch.sigmoid(flood_logit)          # (B, 1)

        # Drought head (SPEI continuous regression)
        spei_pred = self.drought_head(z_fused)           # (B, 1)

        if not return_dict:
            return flood_prob, spei_pred

        return {
            "flood_prob": flood_prob,
            "flood_logit": flood_logit,
            "spei_pred": spei_pred,
            "fused_embedding": z_fused,
            "raw_concat": raw_concat,
            "spatial_embedding": z_spatial,
            "temporal_embedding": z_temporal,
            "spatial_feature_map": spatial_fmap,
            "temporal_attention": temporal_attn
        }

    def get_parameter_count(self) -> Dict[str, int]:
        """Returns trainable and total parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "spatial_encoder": sum(p.numel() for p in self.spatial_encoder.parameters()),
            "temporal_encoder": sum(p.numel() for p in self.temporal_encoder.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "heads": sum(p.numel() for p in self.flood_head.parameters()) + sum(p.numel() for p in self.drought_head.parameters())
        }

    @torch.no_grad()
    def predict(
        self,
        static_features: torch.Tensor,
        dynamic_images: Optional[torch.Tensor] = None,
        climate_sequence: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Inference helper returning numpy arrays for API and downstream consumers.
        """
        was_training = self.training
        self.eval()
        outputs = self.forward(
            static_features=static_features,
            dynamic_images=dynamic_images,
            climate_sequence=climate_sequence,
            return_dict=True
        )
        if was_training:
            self.train()

        return {
            "flood_probability": outputs["flood_prob"].cpu().numpy().squeeze(-1),
            "spei_prediction": outputs["spei_pred"].cpu().numpy().squeeze(-1),
            "temporal_attention": outputs["temporal_attention"].cpu().numpy()
        }
