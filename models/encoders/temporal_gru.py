"""
==============================================================================
Temporal Encoder: Causal Gated Recurrent Unit (GRU) & ConvGRU
==============================================================================
Processes dynamic antecedent time series (CHIRPS rain, SAR backscatter, soil moisture,
ERA5 temp, PET) strictly moving forward in time [t - T, t).
Strict Causal Guarantee: No bidirectional looking into future time t.
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalGRUEncoder(nn.Module):
    """
    Causal GRU temporal encoder.
    Processes sequences of shape (B, T, num_features).
    Outputs temporal summary representation (B, out_dim).
    """

    def __init__(
        self,
        in_features: int = 6,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Unidirectional GRU to enforce causal temporal integrity
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        # Self-attention over temporal steps
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, in_features)
        Returns:
            embedding: (B, out_dim)
            attn_weights: (B, T, 1) temporal saliency scores
        """
        # gru_out: (B, T, hidden_dim)
        gru_out, _ = self.gru(x)

        # Attentive pooling across time steps
        scores = self.temporal_attention(gru_out)  # (B, T, 1)
        attn_weights = F.softmax(scores, dim=1)  # (B, T, 1)
        context = torch.sum(attn_weights * gru_out, dim=1)  # (B, hidden_dim)

        embedding = self.projection(context)
        return embedding, attn_weights


class ConvGRUCell(nn.Module):
    """2D Convolutional GRU Cell for Spatiotemporal patch sequences."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv_gates = nn.Conv2d(in_channels + hidden_channels, 2 * hidden_channels, kernel_size, padding=padding)
        self.conv_candidate = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_channels, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        
        combined = torch.cat([x, h], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        r, z = torch.chunk(gates, 2, dim=1)
        
        candidate_input = torch.cat([x, r * h], dim=1)
        c = torch.tanh(self.conv_candidate(candidate_input))
        
        h_next = (1.0 - z) * h + z * c
        return h_next
