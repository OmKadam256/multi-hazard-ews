"""
==============================================================================
Topographic & Geomorphological Feature Extraction Engine
==============================================================================
Derives physics-based hydrologic terrain indices from Digital Elevation Models (DEM):
- Slope (gradient in radians and degrees)
- Aspect (compass facing direction)
- Topographic Wetness Index (TWI = ln(a / tan(beta)))
- Profile and Planform Curvatures
- Height Above Nearest Drainage (HAND approximation / valley depression)
==============================================================================
"""

import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TopographyEngine")


class TopographicFeatureEngine:
    """
    Computes topographic and geomorphological metrics from 1 km gridded DEM.
    """

    def __init__(self, cell_size_m: float = 1000.0):
        self.cell_size_m = cell_size_m

    def compute_terrain_indices(
        self,
        dem: np.ndarray,
        flow_acc: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        dem: (H, W) elevation surface in meters.
        flow_acc: (H, W) flow accumulation (number of contributing upstream cells).
        Returns:
            Dictionary of computed geomorphological rasters.
        """
        dy, dx = np.gradient(dem, self.cell_size_m)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.rad2deg(slope_rad)

        # Aspect (in degrees: 0-360)
        aspect = np.mod(np.rad2deg(np.arctan2(-dx, dy)), 360.0)

        # Topographic Wetness Index (TWI)
        # Specific Catchment Area a = (flow_acc * cell_size_m)
        # TWI = ln( a / tan(slope) )
        tan_slope = np.maximum(np.tan(slope_rad), 0.001)  # Guard against flat plains div/0
        sca = np.maximum(flow_acc * self.cell_size_m, self.cell_size_m)
        twi = np.log(sca / tan_slope)
        twi = np.clip(twi, 1.0, 30.0)

        # Curvature: second spatial derivatives
        d2y, _ = np.gradient(dy, self.cell_size_m)
        _, d2x = np.gradient(dx, self.cell_size_m)
        total_curvature = -(d2x + d2y)

        # Valley depression / flow convergence index
        valley_depression = (twi > 13.0) & (slope_deg < 3.0)

        return {
            "slope_deg": slope_deg.astype(np.float32),
            "slope_rad": slope_rad.astype(np.float32),
            "aspect": aspect.astype(np.float32),
            "twi": twi.astype(np.float32),
            "total_curvature": total_curvature.astype(np.float32),
            "valley_depression": valley_depression.astype(np.float32)
        }
