"""
==============================================================================
Hydrological Upstream Aggregation & Catchment Dynamics
==============================================================================
Aggregates upstream rainfall and soil saturation using HydroSHEDS 
flow accumulation and Pfafstetter sub-basin boundaries.
Ensures runoff generated upstream in headwaters is routed to downstream risk cells.
==============================================================================
"""

import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HydrologyAggregator")


class HydrologyCatchmentAggregator:
    """
    Computes upstream hydrologic pressures and catchment-level aggregates.
    """

    @staticmethod
    def compute_upstream_weighted_rainfall(
        rain_sum: np.ndarray,
        flow_acc: np.ndarray,
        sub_basins: np.ndarray
    ) -> np.ndarray:
        """
        rain_sum: (H, W) cumulative rainfall grid.
        flow_acc: (H, W) flow accumulation grid.
        sub_basins: (H, W) categorical sub-basin IDs.
        Returns:
            upstream_rain: (H, W) hydrologically routed catchment pressure.
        """
        upstream_rain = rain_sum.copy()
        unique_basins = np.unique(sub_basins)

        # Basin-level flow-weighted antecedent moisture
        for b_id in unique_basins:
            mask = (sub_basins == b_id)
            if not np.any(mask):
                continue
            
            b_rain = rain_sum[mask]
            b_acc = flow_acc[mask]
            
            # Weighted average precipitation across the entire contributing sub-basin
            weights = np.log1p(b_acc)
            weights /= (np.sum(weights) + 1e-6)
            basin_mean_rain = float(np.sum(b_rain * weights))

            # Downstream cells in high accumulation receive combined local + basin headwater pulse
            norm_acc = b_acc / (np.max(b_acc) + 1e-6)
            routed = b_rain * (1.0 - 0.4 * norm_acc) + basin_mean_rain * (0.4 * norm_acc)
            upstream_rain[mask] = routed

        return upstream_rain.astype(np.float32)

    @staticmethod
    def extract_basin_aggregates(
        feature_maps: Dict[str, np.ndarray],
        sub_basins: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Computes basin-wide summary statistics (mean, max, std) for each HydroSHEDS sub-basin.
        """
        basins = np.unique(sub_basins)
        basin_stats = {}

        for b_id in basins:
            mask = (sub_basins == b_id)
            stats = {}
            for name, arr in feature_maps.items():
                if arr.shape == sub_basins.shape:
                    vals = arr[mask]
                    stats[f"{name}_mean"] = float(np.mean(vals))
                    stats[f"{name}_max"] = float(np.max(vals))
                    stats[f"{name}_std"] = float(np.std(vals))
            basin_stats[int(b_id)] = stats

        return basin_stats
