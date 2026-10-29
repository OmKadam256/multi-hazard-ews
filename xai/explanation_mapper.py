"""
==============================================================================
XAI Explanation Geo-Mapper & API Serializer
==============================================================================
Transforms raw SHAP rankings and Grad-CAM 32x32 arrays into GeoJSON 
polygons and visual contour payloads for web map rendering.
==============================================================================
"""

import numpy as np
from typing import Dict, List, Any


class ExplanationGeoMapper:
    """
    Serializes explainability outputs for frontend visualization.
    """

    @staticmethod
    def serialize_gradcam_to_grid(
        heatmap: np.ndarray,
        origin_utm: List[float],
        cell_size_m: float = 1000.0,
        threshold: float = 0.4
    ) -> Dict[str, Any]:
        """
        heatmap: (H, W) array of [0, 1] saliency.
        origin_utm: [min_x, min_y]
        Returns:
            GeoJSON FeatureCollection of high-saliency hotspots.
        """
        features = []
        H, W = heatmap.shape
        min_x, min_y = origin_utm

        for i in range(H):
            for j in range(W):
                val = float(heatmap[i, j])
                if val >= threshold:
                    # Cell bounding box
                    x0 = min_x + j * cell_size_m
                    x1 = x0 + cell_size_m
                    # In raster coords, i goes from top to bottom
                    y1 = min_y + (H - i) * cell_size_m
                    y0 = y1 - cell_size_m

                    polygon = [
                        [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]
                    ]
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [polygon]
                        },
                        "properties": {
                            "saliency": round(val, 3),
                            "significance": "High Hazard Driver" if val > 0.75 else "Moderate Driver"
                        }
                    })

        return {
            "type": "FeatureCollection",
            "features": features
        }
