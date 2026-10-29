"""
==============================================================================
Standardized Precipitation Evapotranspiration Index (SPEI & SPI) Engine
==============================================================================
Calculates multi-scalar drought indices (SPEI-30, SPEI-90) from precipitation (P) 
and Potential Evapotranspiration (PET).
Water balance deficit: D = P - PET.
Standardizes anomalies to standard normal Z-scores (mean=0, std=1).
==============================================================================
"""

import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SPEIEngine")


class DroughtIndexCalculator:
    """
    Computes Standardized Precipitation Index (SPI) and 
    Standardized Precipitation Evapotranspiration Index (SPEI).
    """

    @staticmethod
    def compute_spei_series(
        precipitation: np.ndarray,
        pet: np.ndarray,
        scale_days: int = 30
    ) -> np.ndarray:
        """
        precipitation: 1D array or (T, ...) array of daily precipitation (mm).
        pet: 1D array or (T, ...) array of daily PET (mm).
        scale_days: Accumulation scale (e.g. 30 days for 1-month, 90 days for 3-month).
        Returns:
            spei: Array of standardized values with extreme drought <= -2.0, normal >= -0.5.
        """
        # Water balance deficit D = P - PET
        D = precipitation - pet

        # Cumulative rolling window over the temporal axis (axis 0)
        T = D.shape[0]
        accumulated_D = np.zeros_like(D)

        for t in range(T):
            start_t = max(0, t - scale_days + 1)
            accumulated_D[t] = np.sum(D[start_t:t+1], axis=0)

        # Standardize accumulated water balance
        # If 1D:
        if D.ndim == 1:
            mean = np.nanmean(accumulated_D)
            std = np.nanstd(accumulated_D) + 1e-6
            spei = (accumulated_D - mean) / std
        else:
            # Multi-dimensional (T, H, W)
            mean = np.nanmean(accumulated_D, axis=0, keepdims=True)
            std = np.nanstd(accumulated_D, axis=0, keepdims=True) + 1e-6
            spei = (accumulated_D - mean) / std

        return np.clip(spei, -3.5, 3.5).astype(np.float32)

    @staticmethod
    def classify_spei(spei: np.ndarray) -> np.ndarray:
        """
        Classifies SPEI into WMO standard drought categories:
        0: Normal / Wet (SPEI >= -0.5)
        1: Mild Drought (-1.0 < SPEI <= -0.5)
        2: Moderate Drought (-1.5 < SPEI <= -1.0)
        3: Severe Drought (-2.0 < SPEI <= -1.5)
        4: Extreme Drought (SPEI <= -2.0)
        """
        categories = np.zeros_like(spei, dtype=np.int64)
        categories[spei <= -0.5] = 1
        categories[spei <= -1.0] = 2
        categories[spei <= -1.5] = 3
        categories[spei <= -2.0] = 4
        return categories
