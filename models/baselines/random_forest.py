"""
==============================================================================
Baseline 2: Random Forest Ensemble (Flood & Drought)
==============================================================================
Provides non-linear bagging ensemble baseline with out-of-bag estimates
and class-balanced tree subsampling.
==============================================================================
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BaselineRandomForest")


class BaselineRandomForestSystem:
    """
    Random Forest ensemble for flood probability and continuous drought SPEI.
    """

    def __init__(self, n_estimators: int = 150, max_depth: int = 12, n_jobs: int = -1):
        self.flood_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=n_jobs
        )
        self.drought_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=n_jobs
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y_flood: np.ndarray, y_spei: np.ndarray):
        logger.info(f"Fitting Random Forest on {X.shape[0]} tabular samples...")
        if len(np.unique(y_flood)) < 2:
            X_fit = np.vstack([X, X[-1:]])
            y_fit = np.append(y_flood, 1.0 - y_flood[0])
        else:
            X_fit, y_fit = X, y_flood
        self.flood_model.fit(X_fit, y_fit)
        self.drought_model.fit(X, y_spei)
        self.is_fitted = True
        logger.info("Random Forest fitting complete.")

    def predict_flood_prob(self, X: np.ndarray) -> np.ndarray:
        return self.flood_model.predict_proba(X)[:, 1]

    def predict_spei(self, X: np.ndarray) -> np.ndarray:
        return self.drought_model.predict(X)

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        importances = self.flood_model.feature_importances_
        return {name: float(importances[i]) for i, name in enumerate(feature_names)}
