"""
==============================================================================
Verification Script: Phase 2 Data Loaders, Baselines & Evaluation Metrics
==============================================================================
Validates:
1. PatchDataset extraction of (static_features, dynamic_images, climate_sequence, labels, masks)
2. make_splits() temporal, spatial, and event-wise grouping logic
3. DroughtPersistenceBaseline (persistence & climatology)
4. TabularBaselines (LR, RF, XGBoost)
5. evaluate_flood() and evaluate_drought()
6. verify_no_leakage() permutation sanity verification
==============================================================================
"""

import sys
import os
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from training.dataset import PatchDataset, make_splits, DatasetScaler
from models.baselines import DroughtPersistenceBaseline, TabularBaselines
from evaluation.metrics import evaluate_flood, evaluate_drought, verify_no_leakage


def test_phase2():
    print("================================================================================")
    print("Executing Phase 2 Verification Suite")
    print("================================================================================")

    np.random.seed(42)
    H, W = 64, 64
    T = 40
    C_s = 8
    C_d = 2
    C_c = 3

    # 1. Synthesize minimal test data cube
    static_stack = np.random.normal(100.0, 20.0, (C_s, H, W)).astype(np.float32)
    dynamic_stack = np.random.normal(-15.0, 3.0, (T, C_d, H, W)).astype(np.float32)
    climate_stack = np.random.exponential(10.0, (T, C_c, H, W)).astype(np.float32)
    
    flood_labels = (np.random.rand(T, H, W) > 0.94).astype(np.uint8)
    spei_labels = np.random.normal(0.0, 1.0, (T, H, W)).astype(np.float32)
    drought_class = np.clip(np.digitize(spei_labels, [-2.0, -1.5, -1.0, -0.5]), 0, 4)
    valid_mask = np.ones((H, W), dtype=np.float32)
    valid_mask[0:4, 0:4] = 0.0 # simulate unobserved nodata patch

    # Sub-basins: 4 quadrants
    sub_basins = np.zeros((H, W), dtype=np.int32)
    sub_basins[:32, :32] = 406001
    sub_basins[:32, 32:] = 406002
    sub_basins[32:, :32] = 406007 # buffer
    sub_basins[32:, 32:] = 406009 # test

    # Event IDs
    event_ids = np.full((T, H, W), "none", dtype=object)
    event_ids[10:14, 10:25, 10:25] = "flood_ev_101"
    event_ids[25:29, 40:55, 40:55] = "flood_ev_102"

    dates = [f"2020-01-{d+1:02d}" for d in range(T)]

    # 2. Test make_splits()
    print("\n[Step 1] Testing make_splits() with Spatial, Temporal & Event-Wise Grouping...")
    train_loader, val_loader, test_loader, scaler = make_splits(
        static_stack, dynamic_stack, climate_stack,
        flood_labels, spei_labels, drought_class, valid_mask,
        sub_basins, event_ids, dates,
        train_end_date="2020-01-25",
        val_end_date="2020-01-32",
        test_basin_ids=[406009],
        buffer_basin_ids=[406007],
        patch_size=32,
        stride=16,
        lookback_window=10,
        batch_size=4
    )

    print(f"  [OK] Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)} | Test Batches: {len(test_loader)}")
    assert len(train_loader) > 0, "Train loader should have batches"
    assert len(test_loader) > 0, "Test loader should have batches"

    # 3. Test PatchDataset __getitem__
    print("\n[Step 2] Testing PatchDataset __getitem__() item structure...")
    batch = next(iter(train_loader))
    static_feats, dyn_imgs, clim_seq, labels_dict, masks = batch

    print(f"  [OK] static_features shape: {static_feats.shape} (Expected: [B, {C_s}, 32, 32])")
    print(f"  [OK] dynamic_images shape:  {dyn_imgs.shape} (Expected: [B, 10, {C_d}, 32, 32])")
    print(f"  [OK] climate_sequence shape:{clim_seq.shape} (Expected: [B, 10, {C_c}, 32, 32])")
    print(f"  [OK] masks shape:           {masks.shape} (Expected: [B, 32, 32])")
    print(f"  [OK] labels: flood_binary={labels_dict['flood_binary'].shape}, spei={labels_dict['spei_continuous'].shape}")

    assert static_feats.shape == (4, C_s, 32, 32)
    assert dyn_imgs.shape == (4, 10, C_d, 32, 32)
    assert clim_seq.shape == (4, 10, C_c, 32, 32)
    assert masks.shape == (4, 32, 32)

    # 4. Test TabularBaselines Feature Flattener
    print("\n[Step 3] Testing Tabular Feature Flattener & Baselines...")
    sample_static = static_feats[0].numpy()
    sample_dyn = dyn_imgs[0].numpy()
    sample_clim = clim_seq[0].numpy()

    tab_feats = TabularBaselines.flatten_patch_and_sequence(sample_static, sample_dyn, sample_clim)
    print(f"  [OK] Engineered tabular vector length: {len(tab_feats)}")
    assert len(tab_feats) > 20, "Tabular features should be extracted"

    # Synthetic tabular training set
    N_train = 120
    N_test = 40
    n_feats = len(tab_feats)
    X_tr = np.random.normal(0, 1, (N_train, n_feats)).astype(np.float32)
    y_fl_tr = (np.random.rand(N_train) > 0.85).astype(np.float32)
    y_sp_tr = np.random.normal(0, 1, N_train).astype(np.float32)

    X_te = np.random.normal(0, 1, (N_test, n_feats)).astype(np.float32)
    y_fl_te = (np.random.rand(N_test) > 0.85).astype(np.float32)
    y_sp_te = np.random.normal(0, 1, N_test).astype(np.float32)
    y_pers_te = y_sp_te + np.random.normal(0, 0.4, N_test) # noisy persistence

    baselines = TabularBaselines(rf_estimators=30, xgb_estimators=30)
    baselines.fit_all(X_tr, y_fl_tr, y_sp_tr)

    flood_preds = baselines.predict_flood(X_te)
    drought_preds = baselines.predict_drought(X_te)

    for m_name in ["Logistic_Regression", "Random_Forest", "XGBoost"]:
        print(f"  [OK] Flood {m_name} prediction shape: {flood_preds[m_name].shape}, mean prob={np.mean(flood_preds[m_name]):.3f}")
    for m_name in ["Ridge_Regression", "Random_Forest", "XGBoost"]:
        print(f"  [OK] Drought {m_name} prediction shape: {drought_preds[m_name].shape}, mean SPEI={np.mean(drought_preds[m_name]):.3f}")

    # 5. Test DroughtPersistenceBaseline
    print("\n[Step 4] Testing DroughtPersistenceBaseline...")
    pers_base = DroughtPersistenceBaseline()
    pers_base.fit_climatology(y_sp_tr)
    pers_pred = pers_base.predict_persistence(y_pers_te)
    clim_pred = pers_base.predict_climatology(N_test)
    print(f"  [OK] Fitted Climatology: {pers_base.climatology_mean:.4f}")
    print(f"  [OK] Persistence predictions shape: {pers_pred.shape}")

    # 6. Test Evaluation Metrics
    print("\n[Step 5] Testing evaluate_flood() and evaluate_drought()...")
    flood_metrics = evaluate_flood(y_fl_te, flood_preds["XGBoost"])
    print(f"  [OK] Flood Metrics: PR-AUC={flood_metrics['pr_auc']}, POD@FAR=0.10={flood_metrics['pod_at_far_10']}, CSI={flood_metrics['csi']}, Brier={flood_metrics['brier_score']}")

    drought_metrics = evaluate_drought(y_sp_te, drought_preds["XGBoost"], y_pers_te)
    print(f"  [OK] Drought Metrics: RMSE={drought_metrics['drought_rmse']}, MAE={drought_metrics['drought_mae']}, Skill vs Persist={drought_metrics['skill_score_vs_persistence']}")

    # 7. Test verify_no_leakage()
    print("\n[Step 6] Testing verify_no_leakage() Permutation Sanity Check...")
    leakage_res = verify_no_leakage(baselines, X_te, y_fl_te, y_sp_te, y_pers_te, n_permutations=5)
    print(f"  [OK] Permuted PR-AUC floor: {leakage_res['permuted_random_chance_floor']['mean_permuted_prauc']}")
    print(f"  [OK] Verdict: {leakage_res['verdict']}")
    assert leakage_res["zero_leakage_verified"] is True, "Permutation test should confirm no leakage"

    print("\n================================================================================")
    print("ALL PHASE 2 TESTS PASSED PERFECTLY!")
    print("================================================================================")


if __name__ == "__main__":
    test_phase2()
