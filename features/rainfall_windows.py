"""
==============================================================================
Antecedent Rainfall Windows & Flash Flood Feature Engineering
==============================================================================
Extracts multi-window precipitation indicators strictly prior to forecast issue time:
- Cumulative windows: 3-day, 7-day, 15-day, 30-day
- Maximum 24h intensity in the antecedent window
- Antecedent Precipitation Index (API) with soil memory decay k = 0.90
- Rainfall acceleration / storm burst rate
==============================================================================
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RainfallWindows")


class AntecedentRainfallEngine:
    """
    Engine for multi-temporal rainfall accumulation and storm dynamics.
    """

    def __init__(self, windows: Optional[List[int]] = None, api_decay: float = 0.90):
        self.windows = windows or [3, 7, 15, 30]
        self.api_decay = api_decay

    def compute_temporal_windows(self, daily_rain_cube: np.ndarray) -> Dict[str, np.ndarray]:
        """
        daily_rain_cube: (T, H, W) daily rainfall series strictly prior to t.
        Returns:
            Dictionary of {feature_name: (H, W) or (T, H, W) array}
        """
        T, H, W = daily_rain_cube.shape
        features = {}

        # 1. Cumulative rolling sums for each window
        for w in self.windows:
            w_eff = min(w, T)
            # Sum over the last w_eff days
            rain_w = np.sum(daily_rain_cube[-w_eff:], axis=0)
            features[f"rain_sum_{w}d"] = rain_w.astype(np.float32)

            # Max 24h intensity inside the window
            features[f"rain_max_{w}d"] = np.max(daily_rain_cube[-w_eff:], axis=0).astype(np.float32)

        # 2. Antecedent Precipitation Index (API)
        api = np.zeros((H, W), dtype=np.float32)
        for t in range(T):
            api = self.api_decay * api + daily_rain_cube[t]
        features["api_decay_index"] = api.astype(np.float32)

        # 3. Storm burst / Rainfall acceleration: (last 24h) - (mean of prior 3 days)
        last_24h = daily_rain_cube[-1]
        prior_mean = np.mean(daily_rain_cube[-4:-1], axis=0) if T >= 4 else last_24h
        features["rainfall_burst_acceleration"] = (last_24h - prior_mean).astype(np.float32)

        return features

    @staticmethod
    def extract_tabular_rainfall_features(daily_rain_series: np.ndarray, windows: List[int] = [3, 7, 15, 30]) -> Dict[str, float]:
        """
        Extracts rainfall window scalars for tabular pixel models.
        daily_rain_series: 1D array of daily rainfall up to t.
        """
        feats = {}
        for w in windows:
            slice_w = daily_rain_series[-w:] if len(daily_rain_series) >= w else daily_rain_series
            feats[f"rain_sum_{w}d"] = float(np.sum(slice_w))
            feats[f"rain_max_{w}d"] = float(np.max(slice_w)) if len(slice_w) > 0 else 0.0

        api = 0.0
        for p in daily_rain_series:
            api = 0.90 * api + float(p)
        feats["api_decay_index"] = api
        feats["rain_burst_24h"] = float(daily_rain_series[-1] - np.mean(daily_rain_series[-4:-1])) if len(daily_rain_series) >= 4 else 0.0
        return feats
