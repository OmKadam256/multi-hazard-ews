"""
==============================================================================
PHASE 2: Classical Machine Learning Baselines & Persistence References
==============================================================================
1. DroughtPersistenceBaseline:
   - Persistence model (predicting current SPEI as future SPEI)
   - Climatology model (predicting historical mean SPEI)
2. TabularBaselines:
   - Feature flattener extracting center-pixel and neighbourhood statistics
     from 32x32 patches and dynamic time sequences.
   - Logistic Regression (class-balanced)
   - Random Forest (bagging ensemble)
   - XGBoost (gradient boosting with scale_pos_weight)
==============================================================================
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BaselinesEngine")


class DroughtPersistenceBaseline:
    """
    Hydrometeorological benchmark references:
    1. Persistence: SPEI_{t+lead} = SPEI_{t}
    2. Climatology: SPEI_{t+lead} = Mean(Historical_SPEI)
    """

    def __init__(self):
        self.climatology_mean: float = 0.0

    def fit_climatology(self, historical_spei: np.ndarray):
        """Fits historical climatology reference on training observations only."""
        valid_spei = historical_spei[~np.isnan(historical_spei)]
        self.climatology_mean = float(np.mean(valid_spei)) if len(valid_spei) > 0 else 0.0
        logger.info(f"Fitted Climatology Baseline Mean: {self.climatology_mean:.4f}")

    def predict_persistence(self, current_spei: np.ndarray) -> np.ndarray:
        """Forecasts that current drought conditions persist unchanged into future."""
        return np.asarray(current_spei, dtype=np.float32)

    def predict_climatology(self, n_samples: int) -> np.ndarray:
        """Forecasts historical climatological mean."""
        return np.full(n_samples, self.climatology_mean, dtype=np.float32)


class TabularBaselines:
    """
    Manages tabular feature flattening and classical machine learning models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    """

    def __init__(
        self,
        rf_estimators: int = 150,
        xgb_estimators: int = 200,
        xgb_scale_pos_weight: float = 10.0
    ):
        # Flood Classifiers
        self.flood_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        self.flood_rf = RandomForestClassifier(n_estimators=rf_estimators, max_depth=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        self.flood_xgb = xgb.XGBClassifier(
            n_estimators=xgb_estimators,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=xgb_scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            tree_method="hist"
        )

        # Drought Regressors
        self.drought_ridge = Ridge(alpha=1.0, random_state=42)
        self.drought_rf = RandomForestRegressor(n_estimators=rf_estimators, max_depth=12, random_state=42, n_jobs=-1)
        self.drought_xgb = xgb.XGBRegressor(
            n_estimators=xgb_estimators,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="rmse",
            random_state=42,
            tree_method="hist"
        )

        self.feature_names: List[str] = []
        self.is_fitted = False

    @staticmethod
    def flatten_patch_and_sequence(
        static_patch: np.ndarray,       # (C_s, 32, 32)
        dynamic_sequence: np.ndarray,   # (T_d, C_d, 32, 32)
        climate_sequence: np.ndarray    # (T_c, C_c, 32, 32)
    ) -> np.ndarray:
        """
        Engineers tabular features from the 32x32 patch and temporal sequences.
        Extracts:
        - Center cell values (1x1)
        - Immediate 3x3 neighbourhood context (mean, min, max, gradient)
        - Patch-level geomorphological summary (mean, std, 90th percentile)
        - Antecedent precipitation windows (3d, 7d, 15d, 30d sum & max)
        - Antecedent soil moisture and SAR backscatter trend
        """
        C_s, H, W = static_patch.shape
        cy, cx = H // 2, W // 2 # Center pixel (16, 16)

        features = []

        # 1. Center cell static features (DEM, Slope, TWI, FlowAcc, RiverDist, Soils)
        for c in range(C_s):
            features.append(static_patch[c, cy, cx])

        # 2. 3x3 Immediate neighbourhood context
        y0, y1 = max(0, cy - 1), min(H, cy + 2)
        x0, x1 = max(0, cx - 1), min(W, cx + 2)
        for c in range(min(4, C_s)): # DEM, Slope, TWI, FlowAcc
            nb = static_patch[c, y0:y1, x0:x1]
            features.append(float(np.mean(nb)))
            features.append(float(np.max(nb)))
            features.append(float(np.min(nb)))

        # 3. Patch-wide geomorphology summary
        for c in range(min(3, C_s)):
            features.append(float(np.mean(static_patch[c])))
            features.append(float(np.percentile(static_patch[c], 90)))

        # 4. Dynamic & Climate sequence feature extraction
        # Daily rain is channel 0 of climate_sequence (T_c, C_c, 32, 32)
        rain_series_center = climate_sequence[:, 0, cy, cx]
        rain_series_patch = np.mean(climate_sequence[:, 0], axis=(1, 2))

        # Windows: 3d, 7d, 15d, 30d
        for w in [3, 7, 15, 30]:
            slice_w = rain_series_center[-w:] if len(rain_series_center) >= w else rain_series_center
            features.append(float(np.sum(slice_w)))
            features.append(float(np.max(slice_w)) if len(slice_w) > 0 else 0.0)

        # Patch-wide upstream rain impulse (7d, 15d)
        features.append(float(np.sum(rain_series_patch[-7:])))
        features.append(float(np.sum(rain_series_patch[-15:])))

        # Dynamic SAR VV (channel 0) and Soil Moisture (channel 1) in dynamic_sequence
        if dynamic_sequence.ndim == 4 and dynamic_sequence.shape[1] >= 2:
            sar_curr = dynamic_sequence[-1, 0, cy, cx]
            sm_curr = dynamic_sequence[-1, 1, cy, cx]
            sm_trend = dynamic_sequence[-1, 1, cy, cx] - dynamic_sequence[-7, 1, cy, cx] if dynamic_sequence.shape[0] >= 7 else 0.0
            features.extend([float(sar_curr), float(sm_curr), float(sm_trend)])
        else:
            features.extend([0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    @staticmethod
    def get_feature_names(C_s: int = 8) -> List[str]:
        names = [f"center_static_c{c}" for c in range(C_s)]
        for c in range(min(4, C_s)):
            names.extend([f"nb3x3_mean_c{c}", f"nb3x3_max_c{c}", f"nb3x3_min_c{c}"])
        for c in range(min(3, C_s)):
            names.extend([f"patch_mean_c{c}", f"patch_p90_c{c}"])
        for w in [3, 7, 15, 30]:
            names.extend([f"rain_sum_{w}d", f"rain_max_{w}d"])
        names.extend(["patch_rain_sum_7d", "patch_rain_sum_15d", "sar_vv_curr", "soil_moist_curr", "soil_moist_7d_trend"])
        return names

    def fit_all(
        self,
        X_train: np.ndarray,
        y_flood_train: np.ndarray,
        y_spei_train: np.ndarray
    ):
        """Fits Logistic Regression, Random Forest, and XGBoost on tabular features."""
        logger.info(f"Fitting Tabular Baselines on {X_train.shape[0]} training samples (Features={X_train.shape[1]})...")

        # Class balance guard
        if len(np.unique(y_flood_train)) < 2:
            logger.warning("Single class in flood training labels. Adding synthetic boundary support point.")
            X_flood_fit = np.vstack([X_train, X_train[-1:]])
            y_flood_fit = np.append(y_flood_train, 1.0 - y_flood_train[0])
        else:
            X_flood_fit, y_flood_fit = X_train, y_flood_train

        # 1. Flood Models
        self.flood_lr.fit(X_flood_fit, y_flood_fit)
        self.flood_rf.fit(X_flood_fit, y_flood_fit)
        self.flood_xgb.fit(X_flood_fit, y_flood_fit, verbose=False)

        # 2. Drought Models
        self.drought_ridge.fit(X_train, y_spei_train)
        self.drought_rf.fit(X_train, y_spei_train)
        self.drought_xgb.fit(X_train, y_spei_train, verbose=False)

        self.is_fitted = True
        logger.info("All Tabular Baselines fitted successfully.")

    def predict_flood(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Returns calibrated flood probability for each baseline."""
        if not self.is_fitted:
            raise RuntimeError("Tabular baselines have not been fitted.")
        return {
            "Logistic_Regression": self.flood_lr.predict_proba(X)[:, 1],
            "Random_Forest": self.flood_rf.predict_proba(X)[:, 1],
            "XGBoost": self.flood_xgb.predict_proba(X)[:, 1]
        }

    def predict_drought(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Returns continuous SPEI predictions for each baseline."""
        if not self.is_fitted:
            raise RuntimeError("Tabular baselines have not been fitted.")
        return {
            "Ridge_Regression": self.drought_ridge.predict(X),
            "Random_Forest": self.drought_rf.predict(X),
            "XGBoost": self.drought_xgb.predict(X)
        }
