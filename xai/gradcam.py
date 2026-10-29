"""
==============================================================================
Explainable AI (XAI): Spatial Grad-CAM for Deep Multitask Model
==============================================================================
Computes Gradient-weighted Class Activation Mapping (Grad-CAM) to explain 
spatial inundation hotspots and soil saturation susceptibility.
Target Layer: Final residual block of SpatialCNNEncoder.
==============================================================================
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GradCAM")


class SpatiotemporalGradCAM:
    """
    Grad-CAM engine hooked into the spatial convolutional feature extractor.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Extract underlying module if DataParallel
        underlying = self.model.module if hasattr(self.model, "module") else self.model
        target_layer = underlying.spatial_encoder.target_conv

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        target_hazard: str = "flood"
    ) -> Tuple[np.ndarray, float]:
        """
        x_spatial: (1, C, H=32, W=32)
        x_temporal: (1, T=30, C=6)
        Returns:
            cam_heatmap: (H, W) normalized [0, 1] saliency map.
            predicted_score: Float prediction value.
        """
        self.model.zero_grad()
        outputs = self.model(x_spatial, x_temporal)

        if target_hazard == "flood":
            score = outputs["flood_logit"].view(-1)
            prob = float(outputs["flood_prob"].view(-1).item())
        else:
            # Drought continuous SPEI
            score = -outputs["spei_continuous"].view(-1)  # Negative gradient for drought deficit
            prob = float(outputs["spei_continuous"].view(-1).item())

        score.backward(retain_graph=True)

        gradients = self.gradients.detach()  # (1, C, H', W')
        activations = self.activations.detach()  # (1, C, H', W')

        # Global average pooling of gradients along spatial dimensions to get channel weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # Only features that have positive impact on the hazard

        # Upsample CAM back to original patch resolution (32x32)
        h, w = x_spatial.shape[2], x_spatial.shape[3]
        cam_upsampled = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam_arr = cam_upsampled.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam_arr), np.max(cam_arr)
        if cam_max > cam_min:
            cam_norm = (cam_arr - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_arr)

        return cam_norm.astype(np.float32), prob
