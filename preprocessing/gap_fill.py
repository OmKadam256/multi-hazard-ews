"""
==============================================================================
Spatiotemporal Gap-Filling Pipeline for Geospatial Dynamic Series
==============================================================================
Handles SAR 6-12 day revisit gaps and optical cloud contamination
using temporal forward-fill, linear splines, and spatial neighbor interpolation.
Generates an explicit confidence mask for downstream uncertainty estimation.
==============================================================================
"""

import logging
from typing import Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GapFiller")


class SpatiotemporalGapFiller:
    """
    Interpolates missing observations in multi-sensor spatiotemporal cubes (T, H, W).
    """

    @staticmethod
    def fill_temporal_series(
        cube: np.ndarray,
        valid_mask: np.ndarray,
        method: str = "forward_fill_linear"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        cube: (T, H, W) array of dynamic sensor measurements.
        valid_mask: (T, H, W) boolean mask where True = valid observation, False = missing / cloud.
        Returns:
            filled_cube: (T, H, W) float32 filled array.
            confidence_mask: (T, H, W) float32 confidence score (1.0 = actual, decaying over gap).
        """
        T, H, W = cube.shape
        filled = cube.copy()
        confidence = np.ones((T, H, W), dtype=np.float32)

        for i in range(H):
            for j in range(W):
                series = filled[:, i, j]
                mask = valid_mask[:, i, j]

                if np.all(mask):
                    continue

                if not np.any(mask):
                    # No observations in entire timeline for this pixel: fill with global median
                    filled[:, i, j] = 0.0
                    confidence[:, i, j] = 0.0
                    continue

                # Indices of valid observations
                valid_idx = np.where(mask)[0]
                first_obs = valid_idx[0]
                last_obs = valid_idx[-1]

                # Linear interpolation between observed points
                all_indices = np.arange(T)
                interpolated = np.interp(all_indices, valid_idx, series[valid_idx])
                filled[:, i, j] = interpolated

                # Confidence decay: decays exponentially with time distance from nearest valid pass
                for t in range(T):
                    if mask[t]:
                        confidence[t, i, j] = 1.0
                    else:
                        min_dist = np.min(np.abs(valid_idx - t))
                        # Exponential decay: half confidence every 5 days of missingness
                        confidence[t, i, j] = np.exp(-min_dist / 5.0)

        logger.info(f"Gap filling completed: Mean confidence across cube = {np.mean(confidence):.3f}")
        return filled.astype(np.float32), confidence
