"""
==============================================================================
PHASE 5: Probability Calibration Engine (Isotonic Regression & Platt Scaling)
==============================================================================
Scientific Directives Enforced:
1. Strict Validation Fitting:
   - Calibration curves MUST be fitted strictly on held-out validation data.
   - Prevents overconfidence leakage into operational thresholding.
2. Dual Calibration Algorithms:
   - Isotonic Regression (non-parametric, preserves ranking while correcting scale).
   - Platt Scaling (parametric logistic calibration).
3. Calibration Quality Metrics:
   - Brier Score (BS) & Brier Skill Score (BSS).
   - Expected Calibration Error (ECE) & Maximum Calibration Error (MCE).
   - Reliability diagram coordinates (empirical accuracy vs confidence).
==============================================================================
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Calibrator")


class ProbabilityCalibrator:
    """
    Calibrates raw predictive probabilities into empirical, frequentist likelihoods.
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method.lower()
        if self.method in ["isotonic", "iso"]:
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        elif self.method in ["platt", "logistic", "sigmoid"]:
            self.model = LogisticRegression(C=1.0, solver="lbfgs")
        else:
            raise ValueError(f"Unknown calibration method '{method}'. Supported: 'isotonic', 'platt'.")
        self.is_fitted = False

    def fit(self, y_val_raw_prob: np.ndarray, y_val_true: np.ndarray):
        """
        Fits calibration curve strictly on held-out validation observations.
        
        Args:
            y_val_raw_prob: Uncalibrated probabilities from validation split [0, 1]
            y_val_true: True binary outcomes {0, 1}
        """
        y_prob = np.asarray(y_val_raw_prob, dtype=np.float64).flatten()
        y_true = np.asarray(y_val_true, dtype=np.float64).flatten()

        assert len(y_prob) == len(y_true), "Input shapes must match."
        assert len(y_prob) > 0, "Validation set cannot be empty."

        logger.info(f"Fitting {self.method.upper()} probability calibrator on {len(y_prob)} validation instances...")

        if self.method in ["isotonic", "iso"]:
            self.model.fit(y_prob, y_true)
        else:
            # Platt scaling operates on log-odds / logits or probabilities
            X_feat = y_prob.reshape(-1, 1)
            self.model.fit(X_feat, y_true.astype(int))

        self.is_fitted = True
        logger.info(f"Calibrator successfully fitted on validation split.")

    def calibrate(self, raw_probs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Transforms raw probabilities into calibrated probabilities.
        """
        if not self.is_fitted:
            logger.warning("Calibrator has not been fitted. Returning raw probabilities.")
            return raw_probs

        is_scalar = np.isscalar(raw_probs)
        probs_arr = np.atleast_1d(np.asarray(raw_probs, dtype=np.float64))

        if self.method in ["isotonic", "iso"]:
            calibrated = self.model.predict(probs_arr)
        else:
            X_feat = probs_arr.reshape(-1, 1)
            # Probability of positive class
            calibrated = self.model.predict_proba(X_feat)[:, 1]

        calibrated = np.clip(calibrated, 0.0, 1.0).astype(np.float32)

        if is_scalar:
            return float(calibrated[0])
        return calibrated

    @staticmethod
    def evaluate_calibration(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Computes Brier Score, Expected Calibration Error (ECE), MCE, and Reliability Curve.
        
        Args:
            y_true: True binary labels {0, 1}
            y_prob: Calibrated probabilities [0, 1]
            n_bins: Number of confidence bins for ECE discretization
        
        Returns:
            Dict containing 'brier_score', 'brier_skill_score', 'ece', 'mce', and 'reliability_curve'
        """
        y_t = np.asarray(y_true, dtype=np.float64).flatten()
        y_p = np.asarray(y_prob, dtype=np.float64).flatten()

        assert len(y_t) == len(y_p), "Array lengths must match."
        N = len(y_t)
        if N == 0:
            return {"brier_score": 0.0, "ece": 0.0, "mce": 0.0, "reliability_curve": []}

        # 1. Brier Score & Brier Skill Score
        bs = float(brier_score_loss(y_t, y_p))
        base_rate = float(np.mean(y_t))
        bs_climatology = float(np.mean((y_t - base_rate) ** 2)) + 1e-6
        bss = float(1.0 - (bs / bs_climatology))

        # 2. Binning for ECE and Reliability Diagram
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_assignments = np.digitize(y_p, bin_edges) - 1
        bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

        ece = 0.0
        mce = 0.0
        reliability_curve = []

        for b in range(n_bins):
            mask = (bin_assignments == b)
            bin_size = int(np.sum(mask))

            if bin_size > 0:
                bin_acc = float(np.mean(y_t[mask]))
                bin_conf = float(np.mean(y_p[mask]))
                calibration_gap = abs(bin_acc - bin_conf)
                ece += (bin_size / N) * calibration_gap
                mce = max(mce, calibration_gap)

                reliability_curve.append({
                    "bin_index": b,
                    "bin_center": round(float((bin_edges[b] + bin_edges[b+1]) / 2.0), 3),
                    "empirical_accuracy": round(bin_acc, 4),
                    "mean_confidence": round(bin_conf, 4),
                    "sample_count": bin_size,
                    "calibration_gap": round(calibration_gap, 4)
                })

        return {
            "brier_score": round(bs, 4),
            "brier_skill_score": round(bss, 4),
            "ece": round(float(ece), 4),
            "mce": round(float(mce), 4),
            "base_rate": round(base_rate, 4),
            "reliability_curve": reliability_curve
        }
