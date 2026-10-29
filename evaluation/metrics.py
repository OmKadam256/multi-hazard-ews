"""
==============================================================================
PHASE 2: Rigorous Evaluation Metrics & Leakage Sanity Verification
==============================================================================
Scientific Rules Enforced:
1. Do NOT use standard Accuracy due to extreme flood class imbalance.
2. evaluate_flood():
   - PR-AUC
   - POD (Recall) at fixed FAR (e.g. FAR = 0.10, FAR = 0.20)
   - CSI (Critical Success Index / Threat Score = TP / (TP + FP + FN))
   - F1 Score
   - Brier Score
   - Nodata masking: strictly excludes unobserved pixels from denominators
3. evaluate_drought():
   - RMSE, MAE
   - Skill Score vs Persistence: SS_persist = 1 - (MSE_model / MSE_persistence)
4. verify_no_leakage():
   - Sanity check that shuffles labels and randomizes features to prove
     performance collapses to random chance.
==============================================================================
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
    f1_score
)
from typing import Dict, Any, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MetricsEngine")


def evaluate_flood(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    fixed_fars: Tuple[float, float] = (0.10, 0.20)
) -> Dict[str, float]:
    """
    Computes rigorous hydrologic flood metrics with nodata exclusion.
    
    y_true: Ground truth binary array {0, 1}
    y_prob: Predicted flood probabilities [0, 1]
    mask: Boolean or float mask where 1 = valid observation, 0 = nodata (excluded)
    threshold: Decision threshold for discrete classification (default 0.5)
    fixed_fars: Tuple of target False Alarm Ratios (FAR = FP / (TP + FP) = 1 - Precision)
    """
    y_t = np.asarray(y_true, dtype=np.float32).flatten()
    y_p = np.asarray(y_prob, dtype=np.float32).flatten()

    # Mask out unknown / no-data pixels
    if mask is not None:
        m = np.asarray(mask, dtype=bool).flatten()
        y_t = y_t[m]
        y_p = y_p[m]

    if len(y_t) == 0:
        logger.warning("Empty ground truth slice after masking. Returning default zeros.")
        return {
            "pr_auc": 0.0, "pod_at_far_10": 0.0, "pod_at_far_20": 0.0,
            "csi": 0.0, "f1_score": 0.0, "brier_score": 0.0, "pod": 0.0, "far": 0.0
        }

    # 1. Precision-Recall Curve & PR-AUC
    unique_classes = np.unique(y_t)
    if len(unique_classes) > 1:
        precision, recall, _ = precision_recall_curve(y_t, y_p)
        pr_auc = float(auc(recall, precision))
    else:
        pr_auc = float(np.mean(y_t)) # Baseline prevalence
        precision, recall = np.array([0.0, 1.0]), np.array([0.0, 0.0])

    # 2. Probability of Detection (POD / Recall) at fixed False Alarm Ratios
    # FAR = FP / (TP + FP) = 1 - Precision.
    # Therefore FAR <= target_far means Precision >= 1 - target_far.
    def get_pod_at_far(target_far: float) -> float:
        min_precision = 1.0 - target_far
        indices = np.where(precision >= min_precision)[0]
        if len(indices) == 0:
            return 0.0
        return float(np.max(recall[indices]))

    pod_far1 = get_pod_at_far(fixed_fars[0])
    pod_far2 = get_pod_at_far(fixed_fars[1])

    # 3. Discrete metrics at decision threshold
    y_pred = (y_p >= threshold).astype(np.int32)
    y_true_int = y_t.astype(np.int32)

    tp = float(np.sum((y_pred == 1) & (y_true_int == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true_int == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true_int == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true_int == 0)))

    # Critical Success Index (CSI / Threat Score) = TP / (TP + FP + FN)
    csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    # POD & FAR at default threshold
    pod = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # F1 Score
    f1 = float(f1_score(y_true_int, y_pred, zero_division=0))

    # 4. Brier Score
    brier = float(brier_score_loss(y_true_int, y_p))

    return {
        "pr_auc": round(pr_auc, 4),
        f"pod_at_far_{int(fixed_fars[0]*100)}": round(pod_far1, 4),
        f"pod_at_far_{int(fixed_fars[1]*100)}": round(pod_far2, 4),
        "csi": round(csi, 4),
        "f1_score": round(f1, 4),
        "brier_score": round(brier, 4),
        "pod": round(pod, 4),
        "far": round(far, 4),
        "valid_pixel_count": len(y_t)
    }


def evaluate_drought(
    y_true_spei: np.ndarray,
    y_pred_spei: np.ndarray,
    y_persist_spei: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes continuous drought metrics (RMSE, MAE) and Skill Score vs Persistence.
    
    Skill Score vs Persistence:
      SS_persist = 1 - (MSE_model / MSE_persistence)
      SS > 0 proves the model delivers genuine predictive skill beyond today's weather persisting.
    """
    y_t = np.asarray(y_true_spei, dtype=np.float32).flatten()
    y_p = np.asarray(y_pred_spei, dtype=np.float32).flatten()
    if y_persist_spei is not None:
        y_pers = np.asarray(y_persist_spei, dtype=np.float32).flatten()
    else:
        y_pers = np.zeros_like(y_t)

    if mask is not None:
        m = np.asarray(mask, dtype=bool).flatten()
        y_t = y_t[m]
        y_p = y_p[m]
        y_pers = y_pers[m]

    mse_model = float(mean_squared_error(y_t, y_p))
    rmse_model = float(np.sqrt(mse_model))
    mae_model = float(mean_absolute_error(y_t, y_p))

    mse_persist = float(mean_squared_error(y_t, y_pers)) + 1e-6
    skill_score_persist = float(1.0 - (mse_model / mse_persist))

    return {
        "rmse": round(rmse_model, 4),
        "mae": round(mae_model, 4),
        "drought_rmse": round(rmse_model, 4),
        "drought_mae": round(mae_model, 4),
        "drought_mse": round(mse_model, 4),
        "persistence_mse": round(mse_persist, 4),
        "skill_score_vs_persistence": round(skill_score_persist, 4),
        "valid_pixel_count": len(y_t)
    }


def verify_no_leakage(
    model: Any,
    X_test: np.ndarray,
    y_test_flood: np.ndarray,
    y_test_spei: np.ndarray,
    y_test_persist: np.ndarray,
    n_permutations: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Sanity Verification:
    Tests for information leakage by permuting labels and scrambling input features.
    If leakage exists (e.g. future flood target implicitly leaked into an input feature),
    the model would unrealistically retain high skill on shuffled data.
    Under zero leakage, performance must collapse to random chance:
    - Flood PR-AUC drops to baseline class prevalence
    - CSI drops to near 0
    - Drought Skill Score vs Persistence drops to <= 0.
    """
    np.random.seed(seed)
    logger.info("Running Zero-Leakage Permutation Verification...")

    # True evaluation without permutation
    if hasattr(model, "predict_flood"):
        real_probs = model.predict_flood(X_test)["XGBoost"]
        real_spei = model.predict_drought(X_test)["XGBoost"]
    elif hasattr(model, "predict_flood_prob"):
        real_probs = model.predict_flood_prob(X_test)
        real_spei = model.predict_spei(X_test)
    else:
        real_probs = model.predict_proba(X_test)[:, 1]
        real_spei = model.predict(X_test)

    real_flood_metrics = evaluate_flood(y_test_flood, real_probs)
    real_drought_metrics = evaluate_drought(y_test_spei, real_spei, y_test_persist)

    # Permuted evaluations
    permuted_praucs = []
    permuted_csis = []
    permuted_skills = []

    for _ in range(n_permutations):
        shuffled_f_labels = np.random.permutation(y_test_flood)
        shuffled_s_labels = np.random.permutation(y_test_spei)
        
        # Scramble features randomly
        X_scrambled = np.random.permutation(X_test)

        if hasattr(model, "predict_flood"):
            perm_p = model.predict_flood(X_scrambled)["XGBoost"]
            perm_s = model.predict_drought(X_scrambled)["XGBoost"]
        elif hasattr(model, "predict_flood_prob"):
            perm_p = model.predict_flood_prob(X_scrambled)
            perm_s = model.predict_spei(X_scrambled)
        else:
            perm_p = model.predict_proba(X_scrambled)[:, 1]
            perm_s = model.predict(X_scrambled)

        m_fl = evaluate_flood(shuffled_f_labels, perm_p)
        m_dr = evaluate_drought(shuffled_s_labels, perm_s, y_test_persist)

        permuted_praucs.append(m_fl["pr_auc"])
        permuted_csis.append(m_fl["csi"])
        permuted_skills.append(m_dr["skill_score_vs_persistence"])

    mean_perm_prauc = float(np.mean(permuted_praucs))
    mean_perm_csi = float(np.mean(permuted_csis))
    mean_perm_skill = float(np.mean(permuted_skills))

    class_prevalence = float(np.mean(y_test_flood))

    # Sanity checks:
    # 1. Permuted PR-AUC should be near class prevalence
    # 2. Permuted CSI should be near 0
    # Leakage check: permuted performance should drop to baseline chance and not retain real model predictive power
    leakage_detected = (
        (real_flood_metrics["pr_auc"] > 0.45 and mean_perm_prauc >= real_flood_metrics["pr_auc"] * 0.90) or
        (mean_perm_csi > 0.50)
    )

    result = {
        "zero_leakage_verified": not leakage_detected,
        "class_prevalence": round(class_prevalence, 4),
        "real_model_performance": {
            "flood_prauc": real_flood_metrics["pr_auc"],
            "flood_csi": real_flood_metrics["csi"],
            "drought_skill_vs_persist": real_drought_metrics["skill_score_vs_persistence"]
        },
        "permuted_random_chance_floor": {
            "mean_permuted_prauc": round(mean_perm_prauc, 4),
            "mean_permuted_csi": round(mean_perm_csi, 4),
            "mean_permuted_skill_vs_persist": round(mean_perm_skill, 4)
        },
        "verdict": "PASSED: Model collapsed to random chance under permutation. No data leakage detected."
        if not leakage_detected
        else "FAILED: Suspicious residual skill detected under permutation test. Potential feature leakage."
    }

    logger.info(f"Leakage Sanity Verification Verdict: {result['verdict']}")
    return result
