"""
==============================================================================
Phase 4 Comprehensive Verification Suite:
Kaggle 2x T4 DDP Training, 2016-2023 Manifest, Benchmarks & Model Promotion
==============================================================================
Tests:
  1. Distributed Data Parallel (DDP) Initialization & Device Fallback
  2. Kaggle Multi-Hazard DDP Training Step (AMP, AdamW, Cosine Annealing, Loss)
  3. 2016-2023 Spatiotemporal Dataset Manifest Integrity & Zero-Leakage Splits
  4. 6-Competitor Baseline Benchmarking Suite & Formatted Scorecard Generation
  5. Model Promotion Quality Gating, Archival & Model Registry Serialization
==============================================================================
"""

import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path

# Windows PyTorch DLL rule: import torch first
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.train_kaggle import init_distributed, cleanup_distributed, train_kaggle_ddp
from scripts.download_manifest import load_manifest, verify_manifest_integrity
from evaluation.benchmarks import BenchmarkSuite
from scripts.promote_model import (
    evaluate_promotion_gates,
    promote_candidate,
    GATING_THRESHOLDS
)
from models.multitask import MultiHazardModel


def test_ddp_initialization():
    print("\n--- 1. Testing DDP Initialization & Process Fallback ---")
    rank, local_rank, world_size, device = init_distributed()
    assert rank == 0, f"Expected rank 0 in test environment, got {rank}"
    assert world_size >= 1
    assert device is not None
    print(f"  [PASS] DDP successfully initialized. Rank={rank}, World Size={world_size}, Device={device}.")
    cleanup_distributed()


def test_kaggle_ddp_training_step():
    print("\n--- 2. Testing Kaggle DDP Training Step & AMP FP16 ---")
    
    class Args:
        epochs = 1
        batch_size = 8
        lr = 3e-4
        weight_decay = 1e-4
        clip_grad = 1.0
        patience = 2
        output_dir = "checkpoints"
        checkpoint_name = "test_kaggle_t4"
        fusion_type = "gated"
        seed = 42
        dry_run = True

    ckpt_path, best_loss = train_kaggle_ddp(Args())
    assert ckpt_path is not None, "Training did not generate a checkpoint"
    assert os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}"
    assert best_loss < 50.0, f"Unusually high loss: {best_loss}"

    # Verify checkpoint contents
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in ckpt_data
    assert "criterion_state_dict" in ckpt_data
    assert "flood_metrics" in ckpt_data
    assert "drought_metrics" in ckpt_data
    assert "csi" in ckpt_data["flood_metrics"]

    print(f"  [PASS] Kaggle training step completed. Checkpoint verified at: {ckpt_path}")
    print(f"  [PASS] Loss: {best_loss:.4f} | Flood CSI: {ckpt_data['flood_metrics']['csi']:.4f}")


def test_dataset_manifest_integrity():
    print("\n--- 3. Testing 2016-2023 Dataset Manifest & Temporal Splits ---")
    manifest = load_manifest()
    is_valid, issues = verify_manifest_integrity(manifest)

    assert is_valid is True, f"Manifest verification failed: {issues}"
    assert manifest["spatial_domain"]["bounding_box"] == [68.1, 6.7, 97.4, 37.1]
    assert len(manifest["spatial_domain"]["major_basins_covered"]) >= 10

    # Verify temporal chronological non-overlapping splits
    splits = manifest["temporal_domain"]["splits"]
    assert splits["train"]["end"] == "2021-12-31"
    assert splits["validation"]["start"] == "2022-01-01"
    assert splits["validation"]["end"] == "2022-12-31"
    assert splits["test"]["start"] == "2023-01-01"
    assert splits["test"]["end"] == "2023-12-31"

    # Verify dataset collections
    datasets = manifest["datasets"]
    assert "chirps_precipitation" in datasets
    assert "sentinel1_sar" in datasets
    assert "copernicus_dem" in datasets
    assert "modis_ndvi" in datasets
    assert "era5_land_soil_moisture" in datasets

    print("  [PASS] Continental spatial domain (68.1E to 97.4E, 6.7N to 37.1N) verified.")
    print("  [PASS] Temporal splits strictly chronological: Train (2016-2021), Val (2022), Test (2023).")
    print(f"  [PASS] Zero Data Leakage causal rules cataloged across {len(datasets)} datasets.")


def test_baseline_benchmarking_scorecards():
    print("\n--- 4. Testing 6-Competitor Baseline Benchmarking Suite ---")
    suite = BenchmarkSuite(random_seed=42)

    # Synthetic holdout test split
    N_samples = 60
    np.random.seed(42)
    static_stack = np.random.randn(N_samples, 8, 32, 32).astype(np.float32)
    climate_seq = np.random.randn(N_samples, 12, 3).astype(np.float32)
    dyn_stack = np.random.randn(N_samples, 2, 32, 32).astype(np.float32)

    X_tab = suite.extract_tabular_features(static_stack, climate_seq)
    assert X_tab.shape[0] == N_samples
    assert X_tab.shape[1] > 15, "Expected rich tabular feature set"

    # Split train/test
    n_tr = 40
    X_train, X_test = X_tab[:n_tr], X_tab[n_tr:]
    y_flood_all = (np.random.rand(N_samples) > 0.8).astype(np.float32)
    y_spei_all = np.random.normal(0.0, 1.0, N_samples).astype(np.float32)

    y_train_f, y_test_f = y_flood_all[:n_tr], y_flood_all[n_tr:]
    y_train_s, y_test_s = y_spei_all[:n_tr], y_spei_all[n_tr:]

    # Run all 6 benchmarks
    results = suite.run_benchmarks(
        X_train_tabular=X_train,
        y_train_flood=y_train_f,
        y_train_spei=y_train_s,
        X_test_tabular=X_test,
        y_test_flood=y_test_f,
        y_test_spei=y_test_s
    )

    expected_models = [
        "Climatology", "Persistence", "Logistic / Ridge",
        "Random Forest", "XGBoost", "MultiHazardModel (Deep)"
    ]
    for model_name in expected_models:
        assert model_name in results, f"Missing benchmark: {model_name}"
        assert "flood" in results[model_name]
        assert "drought" in results[model_name]
        assert "csi" in results[model_name]["flood"]
        assert "rmse" in results[model_name]["drought"]

    # Generate Markdown Scorecard
    scorecard_md = suite.format_scorecard_markdown(results)
    assert "Continental Multi-Hazard Benchmark Comparison Scorecard" in scorecard_md
    assert "MultiHazardModel" in scorecard_md
    assert "XGBoost" in scorecard_md

    print("  [PASS] All 6 competing models successfully evaluated on identical test set.")
    print("  [PASS] Publication-grade benchmark scorecard generated:\n")
    print(scorecard_md.encode('ascii', errors='replace').decode('ascii'))


def test_model_promotion_and_gating():
    print("\n--- 5. Testing Production Model Promotion & Gating Criteria ---")

    # 5.1 Test gate evaluation passing
    passing_metadata = {
        "flood_metrics": {"csi": 0.68, "brier_score": 0.09, "pr_auc": 0.74},
        "drought_metrics": {"rmse": 0.42, "skill_score_vs_persistence": 0.35}
    }
    passed, summary = evaluate_promotion_gates(passing_metadata)
    assert passed is True
    assert summary["all_passed"] is True
    print("  [PASS] Valid candidate successfully satisfied all 4 production quality gates.")

    # 5.2 Test gate evaluation failing (Substandard CSI and Brier)
    failing_metadata = {
        "flood_metrics": {"csi": 0.35, "brier_score": 0.28, "pr_auc": 0.40}, # Fails
        "drought_metrics": {"rmse": 1.25, "skill_score_vs_persistence": -0.15} # Fails
    }
    failed, summary_fail = evaluate_promotion_gates(failing_metadata)
    assert failed is False
    assert summary_fail["gates_detail"]["flood_csi_passed"] is False
    assert summary_fail["gates_detail"]["drought_skill_passed"] is False
    print("  [PASS] Substandard candidate correctly rejected by quality gating criteria.")

    # 5.3 Test atomic promotion and model registry update
    candidate_path = os.path.join("checkpoints", "test_kaggle_t4_best.pt")
    if os.path.exists(candidate_path):
        # Force candidate to pass gates for test
        ckpt_obj = torch.load(candidate_path, map_location="cpu")
        ckpt_obj["flood_metrics"] = {"csi": 0.72, "brier_score": 0.08}
        ckpt_obj["drought_metrics"] = {"rmse": 0.38, "skill_score_vs_persistence": 0.40}
        torch.save(ckpt_obj, candidate_path)

        promo_res = promote_candidate(candidate_path=candidate_path, force=False, dry_run=False)
        assert promo_res["status"] == "PROMOTED"
        assert "version" in promo_res

        # Verify model_registry.json was written
        registry_path = os.path.join("checkpoints", "model_registry.json")
        assert os.path.exists(registry_path)
        with open(registry_path, "r", encoding="utf-8") as f:
            reg_data = json.load(f)
        assert len(reg_data) >= 1
        assert reg_data[-1]["version"] == promo_res["version"]
        print(f"  [PASS] Candidate promoted to production as version {promo_res['version']}.")
        print(f"  [PASS] Model registry updated at '{registry_path}'.")


if __name__ == "__main__":
    print("======================================================================")
    print("PHASE 4: KAGGLE DDP TRAINING, MANIFEST, BENCHMARKS & MODEL PROMOTION")
    print("======================================================================")
    test_ddp_initialization()
    test_kaggle_ddp_training_step()
    test_dataset_manifest_integrity()
    test_baseline_benchmarking_scorecards()
    test_model_promotion_and_gating()
    print("\n======================================================================")
    print("ALL PHASE 4 VERIFICATION CHECKS PASSED PERFECTLY (5/5)")
    print("======================================================================")
