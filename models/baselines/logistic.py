"""
==============================================================================
Baseline 1: Logistic Regression (Flood) & Ridge Regression (Drought SPEI)
==============================================================================
Provides the fundamental linear baseline for ablation comparison.
Supports class-balanced weighting for severe flood class imbalance.
==============================================================================
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BaselineLinear")


class BaselineLinearSystem:
    """
    Standard linear baseline model for flood classification and drought regression.
    """

    def __init__(self, max_iter: int = 1000):
        self.flood_model = LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            solver="lbfgs",
            random_state=42
        )
        self.drought_model = Ridge(alpha=1.0, random_state=42)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y_flood: np.ndarray, y_spei: np.ndarray):
        """
        X: (N, num_features) Tabular feature matrix.
        y_flood: (N,) Binary flood labels (0 or 1).
        y_spei: (N,) Continuous SPEI labels.
        """
        logger.info(f"Fitting Baseline Linear models on {X.shape[0]} tabular samples...")
        if len(np.unique(y_flood)) < 2:
            logger.warning("Single class in y_flood: adding synthetic boundary point to allow solver fitting.")
            X_fit = np.vstack([X, X[-1:]])
            y_fit = np.append(y_flood, 1.0 - y_flood[0])
        else:
            X_fit, y_fit = X, y_flood
        self.flood_model.fit(X_fit, y_fit)
        self.drought_model.fit(X, y_spei)
        self.is_fitted = True
        logger.info("Baseline Linear fitting complete.")

    def predict_flood_prob(self, X: np.ndarray) -> np.ndarray:
        """Returns P(Flood=1) calibrated to [0, 1]."""
        return self.flood_model.predict_proba(X)[:, 1]

    def predict_spei(self, X: np.ndarray) -> np.ndarray:
        """Returns continuous SPEI predictions."""
        return self.drought_model.predict(X)

    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        """Returns normalized absolute linear weights."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        coefs = np.abs(self.flood_model.coef_[0])
        total = np.sum(coefs) + 1e-6
        return {name: float(coefs[i] / total) for i, name in enumerate(feature_names)}
