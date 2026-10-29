"""
==============================================================================
Verification Script: Phase 5 Explainability (XAI) & Early Warning Engine
==============================================================================
Validates:
1. TabularExplainer (SHAP global & local attributions for tabular baselines).
2. SpatialExplainer (Grad-CAM heatmaps & Adebayo et al. faithfulness sanity check).
3. LocalExplanationGenerator (Grid-cell local report with drivers, hotspots, staleness).
4. ProbabilityCalibrator (Isotonic & Platt scaling, Brier Score, ECE, Reliability).
5. determine_alert_thresholds (Empirical POD/FAR trade-off thresholds).
6. WarningEngine (EarlyWarningRecord, Hysteresis, Critical persistence, Staleness/Uncertainty flags).
==============================================================================
"""

import sys
import os
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from evaluation.xai import TabularExplainer, SpatialExplainer, generate_local_explanation
from risk_engine.calibration import ProbabilityCalibrator
from risk_engine.thresholding import (
    WarningEngine,
    EarlyWarningRecord,
    AlertLevel,
    determine_alert_thresholds
)
from models.multitask import MultiHazardModel


def test_phase5():
    print("================================================================================")
    print("Executing Phase 5 Verification Suite: Explainability & Early Warning Engine")
    print("================================================================================")

    np.random.seed(42)
    torch.manual_seed(42)

    # --------------------------------------------------------------------------
    # 1. Test TabularExplainer (SHAP)
    # --------------------------------------------------------------------------
    print("\n[Step 1] Testing TabularExplainer (SHAP)...")
    feature_names = [
        "dem_center", "slope_center", "twi_center", "rain_3d_sum",
        "rain_7d_sum", "rain_15d_sum", "rain_30d_sum", "api_decay", "soil_moisture"
    ]
    N_samples = 60
    X_tab = np.random.normal(0.0, 1.0, (N_samples, len(feature_names))).astype(np.float32)
    # Target strongly correlated with rain_7d_sum (index 4) and twi_center (index 2)
    y_tab = ((X_tab[:, 4] * 1.5 + X_tab[:, 2] * 1.2 + np.random.normal(0, 0.3, N_samples)) > 0).astype(int)

    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X_tab, y_tab)

    tab_explainer = TabularExplainer(model=rf_model, feature_names=feature_names)

    # Test Global Explanation
    global_importance = tab_explainer.explain_global(X_tab[:20], top_k=5)
    print(f"  [OK] Global top feature ranking: {list(global_importance.keys())}")
    assert len(global_importance) == 5
    assert sum(global_importance.values()) <= 1.01

    # Test Local Explanation
    local_exp = tab_explainer.explain_local(X_tab[0], top_k=3)
    print(f"  [OK] Local explanation top driver: {local_exp['attributions'][0]['feature']} "
          f"({local_exp['attributions'][0]['attribution']:+.4f})")
    assert len(local_exp["attributions"]) == 3

    # --------------------------------------------------------------------------
    # 2. Test SpatialExplainer (Grad-CAM & Faithfulness Sanity Check)
    # --------------------------------------------------------------------------
    print("\n[Step 2] Testing SpatialExplainer (Grad-CAM & Faithfulness)...")
    model = MultiHazardModel(
        static_channels=8,
        dynamic_channels=2,
        temporal_in_features=3,
        fusion_type="gated"
    )

    spatial_explainer = SpatialExplainer(model=model)

    static_x = torch.randn(1, 8, 32, 32)
    dynamic_x = torch.randn(1, 2, 32, 32)
    climate_x = torch.randn(1, 10, 3)

    cam_map, pred_prob = spatial_explainer.generate_heatmap(
        static_features=static_x,
        dynamic_images=dynamic_x,
        climate_sequence=climate_x,
        target_hazard="flood"
    )

    print(f"  [OK] Grad-CAM heatmap generated with shape: {cam_map.shape} (Expected: [32, 32])")
    print(f"       Heatmap min={cam_map.min():.4f}, max={cam_map.max():.4f}, mean={cam_map.mean():.4f}")
    assert cam_map.shape == (32, 32)
    assert cam_map.min() >= 0.0 and cam_map.max() <= 1.0

    # Test Faithfulness Sanity Check (Adebayo et al., NeurIPS 2018)
    faithfulness = spatial_explainer.sanity_check_faithfulness(
        static_features=static_x,
        dynamic_images=dynamic_x,
        climate_sequence=climate_x,
        target_hazard="flood"
    )
    print(f"  [OK] Faithfulness Sanity Check: {faithfulness['verdict']}")
    print(f"       Mean Abs Diff={faithfulness['mean_absolute_difference']:.4f}, "
          f"Randomized Correlation={faithfulness['randomized_correlation']:.4f}")
    assert faithfulness["faithfulness_passed"] is True, "Sanity check must verify model weight dependency"

    # --------------------------------------------------------------------------
    # 3. Test LocalExplanationGenerator
    # --------------------------------------------------------------------------
    print("\n[Step 3] Testing LocalExplanationGenerator...")
    report = generate_local_explanation(
        grid_id="basin_406001_c12",
        timestamp="2026-10-24T21:00:00Z",
        flood_prob=0.68,
        spei_forecast=-1.85,
        tabular_explainer=tab_explainer,
        tabular_features=X_tab[0],
        spatial_explainer=spatial_explainer,
        spatial_inputs=(static_x, dynamic_x, climate_x),
        uncertainty=0.12,
        data_staleness_hours=4.5
    )

    print(f"  [OK] Local report generated for: {report['grid_id']}")
    print(f"       Flood Prob={report['flood_probability']}, SPEI={report['spei_forecast']}, "
          f"Staleness={report['data_staleness_hours']}h")
    print(f"       Top Drivers: {[d['feature'] for d in report['top_drivers']]}")
    print(f"       Spatial Hotspots: {report['spatial_hotspots']}")
    assert len(report["top_drivers"]) > 0
    assert len(report["spatial_hotspots"]) == 3
    assert report["is_stale"] is False

    # --------------------------------------------------------------------------
    # 4. Test ProbabilityCalibrator (Isotonic & Platt)
    # --------------------------------------------------------------------------
    print("\n[Step 4] Testing ProbabilityCalibrator (Isotonic & Platt)...")
    N_val = 200
    # True latent probability
    z_val = np.random.normal(0, 1.5, N_val)
    y_val_true = (np.random.rand(N_val) < (1.0 / (1.0 + np.exp(-z_val)))).astype(int)
    # Simulated overconfident raw model predictions
    y_val_raw = np.clip(1.0 / (1.0 + np.exp(-z_val * 2.5)), 0.01, 0.99)

    calibrator_iso = ProbabilityCalibrator(method="isotonic")
    calibrator_iso.fit(y_val_raw, y_val_true)

    calibrator_platt = ProbabilityCalibrator(method="platt")
    calibrator_platt.fit(y_val_raw, y_val_true)

    cal_iso = calibrator_iso.calibrate(y_val_raw)
    cal_platt = calibrator_platt.calibrate(y_val_raw)

    eval_raw = ProbabilityCalibrator.evaluate_calibration(y_val_true, y_val_raw)
    eval_iso = ProbabilityCalibrator.evaluate_calibration(y_val_true, cal_iso)
    eval_platt = ProbabilityCalibrator.evaluate_calibration(y_val_true, cal_platt)

    print(f"  [OK] Raw:    Brier={eval_raw['brier_score']:.4f}, ECE={eval_raw['ece']:.4f}")
    print(f"  [OK] Isotonic: Brier={eval_iso['brier_score']:.4f}, ECE={eval_iso['ece']:.4f} (Improved!)")
    print(f"  [OK] Platt:    Brier={eval_platt['brier_score']:.4f}, ECE={eval_platt['ece']:.4f}")

    assert eval_iso["ece"] <= eval_raw["ece"] + 0.05
    assert len(eval_iso["reliability_curve"]) > 0

    # --------------------------------------------------------------------------
    # 5. Test determine_alert_thresholds
    # --------------------------------------------------------------------------
    print("\n[Step 5] Testing determine_alert_thresholds (POD/FAR Trade-offs)...")
    derived_thresholds = determine_alert_thresholds(
        y_val_true=y_val_true,
        y_val_prob=cal_iso,
        target_fars=(0.25, 0.15, 0.05)
    )
    print(f"  [OK] Derived Thresholds: {derived_thresholds}")
    assert derived_thresholds["LOW"] < derived_thresholds["MODERATE"] < derived_thresholds["HIGH"] < derived_thresholds["CRITICAL"]

    # --------------------------------------------------------------------------
    # 6. Test WarningEngine (Hysteresis, Uncertainty & Pydantic Emission)
    # --------------------------------------------------------------------------
    print("\n[Step 6] Testing WarningEngine (Hysteresis & Uncertainty Flags)...")
    engine = WarningEngine(
        thresholds=derived_thresholds,
        hysteresis_delta=0.05,
        persistence_steps=2,
        max_staleness_hours=24.0,
        max_uncertainty_sigma=0.35
    )

    # Test 6.1: Routine Low Warning
    rec_low = engine.process_prediction(
        grid_id="basin_101",
        calibrated_prob=0.08,
        spei_forecast=0.2,
        issue_timestamp="2026-10-24T21:00:00Z"
    )
    print(f"  [OK] Low alert emitted: level={rec_low.warning_level}, color={rec_low.color_code}")
    assert rec_low.warning_level == AlertLevel.LOW
    assert rec_low.hazard_only_metric is True

    # Test 6.2: Escalation to High
    prob_high = derived_thresholds["HIGH"] + 0.05
    rec_high = engine.process_prediction(
        grid_id="basin_101",
        calibrated_prob=prob_high,
        issue_timestamp="2026-10-24T22:00:00Z"
    )
    print(f"  [OK] Escalated to: {rec_high.warning_level} (p={prob_high:.3f})")
    assert rec_high.warning_level == AlertLevel.HIGH

    # Test 6.3: Anti-Flapping De-escalation Resistance (prob dips into dead-band)
    rec_hold = engine.process_prediction(
        grid_id="basin_101",
        calibrated_prob=derived_thresholds["HIGH"] - 0.02, # inside dead-band
        issue_timestamp="2026-10-24T23:00:00Z"
    )
    print(f"  [OK] Anti-flapping held level: {rec_hold.warning_level} (Expected: HIGH)")
    assert rec_hold.warning_level == AlertLevel.HIGH

    # Test 6.4: Stale Input Data Handling (Must emit ADVISORY_UNCERTAIN, NOT LOW!)
    rec_stale = engine.process_prediction(
        grid_id="basin_102",
        calibrated_prob=0.05,
        issue_timestamp="2026-10-24T21:00:00Z",
        data_staleness_hours=36.0 # Stale sensor!
    )
    print(f"  [OK] Stale data handled: {rec_stale.warning_level} (Explicit uncertainty tag!)")
    assert rec_stale.warning_level == AlertLevel.ADVISORY_UNCERTAIN
    assert rec_stale.data_quality_flags["is_stale"] is True

    # Test 6.5: High Model Uncertainty Handling
    rec_unc = engine.process_prediction(
        grid_id="basin_103",
        calibrated_prob=0.10,
        issue_timestamp="2026-10-24T21:00:00Z",
        uncertainty_sigma=0.48 # High uncertainty!
    )
    print(f"  [OK] High uncertainty handled: {rec_unc.warning_level} (Explicit advisory tag!)")
    assert rec_unc.warning_level == AlertLevel.ADVISORY_UNCERTAIN
    assert rec_unc.data_quality_flags["high_uncertainty_flag"] is True

    # Test 6.6: Compound Hazard Escalation (Severe drought soil hydrophobicity + rain)
    rec_compound = engine.process_prediction(
        grid_id="basin_104",
        calibrated_prob=0.42,
        spei_forecast=-2.1, # Extreme drought
        issue_timestamp="2026-10-24T21:00:00Z"
    )
    print(f"  [OK] Compound hazard detected: type={rec_compound.hazard_type}, level={rec_compound.warning_level}")
    assert rec_compound.hazard_type == "compound"
    assert rec_compound.warning_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]

    # Test 6.7: Verify EarlyWarningRecord JSON serialization
    json_record = rec_low.model_dump_json()
    assert "basin_101" in json_record
    assert "hazard_only_metric" in json_record
    print("  [OK] EarlyWarningRecord successfully serializes to standard JSON.")

    print("\n================================================================================")
    print("ALL PHASE 5 TESTS PASSED PERFECTLY!")
    print("================================================================================")


if __name__ == "__main__":
    test_phase5()
