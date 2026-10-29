"""
==============================================================================
PHASE 4: Joint Multi-Hazard Spatiotemporal Architecture (Flood + Drought)
==============================================================================
Scientific Directives Enforced:
1. Ablation Step A3 (Plain Concatenation vs Gated Fusion):
   - Supports both GatedFusion and PlainConcatenation via fusion_type parameter.
2. Ablation Step A4 (Independent vs Shared Heads):
   - Shared Trunk: Fused representation passes through a shared dense MLP trunk,
     learning shared multi-hazard hydrological features before branching.
   - Independent Heads: use_shared_trunk=False bypasses the shared representation
     trunk, sending the fused vector directly to separate head networks.
3. Multi-Hazard Dual Heads:
   - Flood Head: Continuous probability of inundation in future window [0, 1].
   - Drought Head: Continuous future SPEI forecast (and 5-class categorical severity).
4. Hardware Profile: Dual T4 GPU DataParallel / DDP ready with mixed precision (AMP).
==============================================================================
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, Union

from models.encoders import SpatialCNN, TemporalGRU
from models.fusion import GatedFusion, PlainConcatenation


class FloodHead(nn.Module):
    """
    Flood Inundation Classification Head.
    Outputs raw logit for Focal Loss / BCE and Sigmoid probability in [0, 1].
    """

    def __init__(self, in_features: int = 128, hidden_dims: list = [64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        curr = in_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(curr, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)
            ])
            curr = h
        layers.append(nn.Linear(curr, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logit: (B, 1) raw logit for loss calculation
            prob:  (B, 1) calibrated sigmoid probability
        """
        logit = self.net(x)
        prob = torch.sigmoid(logit)
        return logit, prob


class DroughtHead(nn.Module):
    """
    Drought Forecast Head.
    Outputs continuous future SPEI index (Linear regression)
    and optional 5-class drought severity categorization.
    """

    def __init__(
        self,
        in_features: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
        # Continuous SPEI regression (Standard Normal Z-score from -3.0 to +3.0)
        self.regressor = nn.Linear(hidden_dim, 1)
        # Categorical classification (Normal, Mild, Moderate, Severe, Extreme)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            spei_cont: (B, 1) continuous predicted SPEI
            class_logits: (B, 5) categorical severity logits
        """
        h = self.shared(x)
        spei_cont = self.regressor(h)
        class_logits = self.classifier(h)
        return spei_cont, class_logits


class MultiHazardModel(nn.Module):
    """
    Joint Multi-Hazard Deep Learning Model for Simultaneous Flood & Drought Early Warning.
    
    Ablation Parameters:
      - fusion_type: 'gated' (GatedFusion) vs 'concat' (PlainConcatenation) [Ablation A3]
      - use_shared_trunk: True (Shared MLP trunk) vs False (Independent heads) [Ablation A4]
    """

    def __init__(
        self,
        static_channels: int = 8,
        dynamic_channels: int = 2,
        temporal_in_features: int = 3,
        spatial_dim: int = 128,
        temporal_dim: int = 128,
        fusion_dim: int = 128,
        trunk_dim: int = 128,
        fusion_type: str = "gated",
        use_shared_trunk: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.static_channels = static_channels
        self.dynamic_channels = dynamic_channels
        self.temporal_in_features = temporal_in_features
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.fusion_dim = fusion_dim
        self.trunk_dim = trunk_dim
        self.fusion_type = fusion_type.lower()
        self.use_shared_trunk = use_shared_trunk

        # 1. Spatial CNN Encoder (32x32 terrain & surface condition rasters)
        self.spatial_encoder = SpatialCNN(
            static_channels=static_channels,
            dynamic_channels=dynamic_channels,
            out_dim=spatial_dim,
            dropout=dropout
        )

        # 2. Causal Temporal GRU Encoder (antecedent meteorology < t)
        self.temporal_encoder = TemporalGRU(
            in_features=temporal_in_features,
            hidden_dim=64,
            out_dim=temporal_dim,
            num_layers=2,
            dropout=dropout
        )

        # 3. Fusion Layer (Ablation Step A3: Plain Concatenation vs Gated Fusion)
        if self.fusion_type in ["gated", "gated_cross_modal"]:
            self.fusion = GatedFusion(
                spatial_dim=spatial_dim,
                temporal_dim=temporal_dim,
                out_dim=fusion_dim,
                dropout=dropout
            )
        else:
            self.fusion = PlainConcatenation(
                spatial_dim=spatial_dim,
                temporal_dim=temporal_dim,
                out_dim=fusion_dim,
                dropout=dropout
            )

        # 4. Shared Trunk vs Independent Heads (Ablation Step A4)
        if self.use_shared_trunk:
            self.shared_trunk = nn.Sequential(
                nn.Linear(fusion_dim, trunk_dim),
                nn.BatchNorm1d(trunk_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout),
                nn.Linear(trunk_dim, trunk_dim),
                nn.BatchNorm1d(trunk_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)
            )
            head_in_features = trunk_dim
        else:
            self.shared_trunk = None
            head_in_features = fusion_dim

        # 5. Dual Hazard Output Heads
        self.flood_head = FloodHead(in_features=head_in_features, dropout=dropout)
        self.drought_head = DroughtHead(in_features=head_in_features, num_classes=5, dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        """Kaiming and Xavier initialization across model layers."""
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
        static_features: Optional[torch.Tensor] = None,
        dynamic_images: Optional[torch.Tensor] = None,
        climate_sequence: Optional[torch.Tensor] = None,
        x_spatial: Optional[torch.Tensor] = None,
        x_temporal: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Joint multi-hazard forward pass.
        Supports both clean keyword arguments and legacy x_spatial / x_temporal.
        """
        # Backward compatibility argument resolution
        if static_features is None and x_spatial is not None:
            static_features = x_spatial
        if climate_sequence is None and x_temporal is not None:
            climate_sequence = x_temporal

        assert static_features is not None, "static_features must be provided."
        assert climate_sequence is not None, "climate_sequence must be provided."

        # 1. Modality Encodings
        z_spatial, spatial_fmap = self.spatial_encoder(
            static_features=static_features,
            dynamic_images=dynamic_images,
            return_feature_map=True
        )
        z_temporal, temp_attn = self.temporal_encoder(
            x=climate_sequence,
            return_attention=True
        )

        # 2. Modality Fusion
        gate_weights = None
        if isinstance(self.fusion, GatedFusion):
            z_fused, gate_weights = self.fusion(z_spatial, z_temporal, return_gate=True)
        else:
            z_fused = self.fusion(z_spatial, z_temporal)

        # 3. Representation Trunk (Shared vs Independent)
        if self.use_shared_trunk and self.shared_trunk is not None:
            rep = self.shared_trunk(z_fused)
        else:
            rep = z_fused

        # 4. Hazard Heads
        flood_logit, flood_prob = self.flood_head(rep)
        spei_cont, drought_logits = self.drought_head(rep)

        if not return_dict:
            return flood_prob, spei_cont

        outputs = {
            "flood_logit": flood_logit,
            "flood_prob": flood_prob,
            "spei_continuous": spei_cont,
            "drought_logits": drought_logits,
            "drought_probs": torch.softmax(drought_logits, dim=-1),
            "z_spatial": z_spatial,
            "z_temporal": z_temporal,
            "z_fused": z_fused,
            "representation": rep,
            "spatial_feature_map": spatial_fmap,
            "temporal_attention": temp_attn
        }
        if gate_weights is not None:
            outputs["gate_weights"] = gate_weights

        return outputs

    def get_parameter_count(self) -> Dict[str, int]:
        """Returns parameter breakdown for ablation tracking."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        trunk_params = sum(p.numel() for p in self.shared_trunk.parameters()) if self.shared_trunk else 0
        return {
            "trainable": trainable,
            "total": total,
            "spatial_encoder": sum(p.numel() for p in self.spatial_encoder.parameters()),
            "temporal_encoder": sum(p.numel() for p in self.temporal_encoder.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "shared_trunk": trunk_params,
            "flood_head": sum(p.numel() for p in self.flood_head.parameters()),
            "drought_head": sum(p.numel() for p in self.drought_head.parameters())
        }


# Backward compatibility aliases
MultiHazardDeepModel = MultiHazardModel


def create_model_for_hardware(
    config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, torch.device]:
    """
    Instantiates MultiHazardModel with automatic hardware detection and Dual-T4 wrapping.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config.get("models", {}).get("deep_learning", {})
    fusion_type = model_cfg.get("fusion", {}).get("type", "gated")
    
    model = MultiHazardModel(
        static_channels=8,
        dynamic_channels=2,
        temporal_in_features=3,
        spatial_dim=128,
        temporal_dim=128,
        fusion_dim=128,
        trunk_dim=128,
        fusion_type=fusion_type,
        use_shared_trunk=True
    )

    model = model.to(device)

    # Multi-GPU DataParallel wrapping for dual T4 accelerators
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        import logging
        logging.info(f"Detected {torch.cuda.device_count()} GPUs. Wrapping MultiHazardModel in nn.DataParallel.")
        model = nn.DataParallel(model)

    return model, device
