"""
==============================================================================
Baseline 3: XGBoost Gradient Boosted Trees (Flood & Drought SPEI)
==============================================================================
High-performance gradient boosting baseline.
Utilizes scale_pos_weight to address extreme flood class imbalance.
Integrates natively with SHAP TreeExplainer.
==============================================================================
"""

import numpy as np
import xgboost as xgb
from typing import Dict, Any, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BaselineXGBoost")


class BaselineXGBoostSystem:
    """
    XGBoost gradient boosting models for multi-hazard tabular forecasting.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 12.0
    ):
        self.flood_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            tree_method="hist"
        )
        self.drought_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="rmse",
            random_state=42,
            tree_method="hist"
        )
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_flood_train: np.ndarray,
        y_spei_train: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    ):
        logger.info(f"Fitting XGBoost on {X_train.shape[0]} tabular samples...")
        if len(np.unique(y_flood_train)) < 2:
            X_fit = np.vstack([X_train, X_train[-1:]])
            y_fit = np.append(y_flood_train, 1.0 - y_flood_train[0])
        else:
            X_fit, y_fit = X_train, y_flood_train

        flood_eval = None
        drought_eval = None
        if eval_set is not None:
            X_val, y_flood_val, y_spei_val = eval_set
            flood_eval = [(X_val, y_flood_val)]
            drought_eval = [(X_val, y_spei_val)]

        self.flood_model.fit(X_fit, y_fit, eval_set=flood_eval, verbose=False)
        self.drought_model.fit(X_train, y_spei_train, eval_set=drought_eval, verbose=False)
        self.is_fitted = True
        logger.info("XGBoost training completed.")

    def predict_flood_prob(self, X: np.ndarray) -> np.ndarray:
        return self.flood_model.predict_proba(X)[:, 1]

    def predict_spei(self, X: np.ndarray) -> np.ndarray:
        return self.drought_model.predict(X)

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        scores = self.flood_model.feature_importances_
        return {name: float(scores[i]) for i, name in enumerate(feature_names)}
