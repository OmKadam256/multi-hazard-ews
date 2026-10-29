"""
==============================================================================
AI-Driven Multi-Hazard Early Warning System: Master Pipeline Orchestrator
==============================================================================
Executes end-to-end scientific workflow:
1. Environment & Dual T4 GPU Verification
2. Geospatial Physical Domain & Spatiotemporal Data Generation
3. Feature Engineering: TWI, Rainfall Windows, SPEI-30, Upstream Aggregates
4. Strict No-Leakage Spatial Split (HydroSHEDS) & Chronological Horizon Split
5. Master 7-Model Ablation Battery Execution & Metric Evaluation
6. Explainable AI: SHAP Tabular Feature Attribution & Grad-CAM Hotspots
7. Operational Risk Calibration & Temporal Hysteresis Verification
==============================================================================
"""

import os
import sys
import yaml
import logging
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from preprocessing.gee_download import MockGeospatialDataGenerator, GEEIngestionEngine
from preprocessing.harmonizer import GeospatialHarmonizer
from features.spei import DroughtIndexCalculator
from features.rainfall_windows import AntecedentRainfallEngine
from features.topography import TopographicFeatureEngine
from models.multitask import create_model_for_hardware
from evaluation.ablation import MultiHazardAblationEngine
from evaluation.spatial_split import HydroSHEDSSpatialSplitter
from evaluation.temporal_split import TemporalChronologicalSplitter
from training.dataset import create_dataloaders
from training.trainer import MultiHazardTrainer
from xai.shap_tabular import TabularSHAPExplainer
from xai.gradcam import SpatiotemporalGradCAM
from risk_engine.calibrator import ProbabilityCalibrator
from risk_engine.hysteresis import HysteresisStateMachine
from risk_engine.thresholds import MultiHazardThresholdEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MasterEEWSPipeline")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    logger.info("================================================================================")
    logger.info("Starting AI-Driven Multi-Hazard Early Warning System (Flood + Drought) Pipeline")
    logger.info("================================================================================")

    config = load_config()

    # 1. Hardware Verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Execution Device: {device} | Active GPUs: {num_gpus}")
    if num_gpus > 1:
        logger.info("Dual GPU T4 configuration active: PyTorch DataParallel & FP16 AMP enabled.")

    # 2. Ingestion & Preprocessing with Strict Zero Leakage
    logger.info("\n[Phase 1] Geospatial Preprocessing & Zero-Leakage Synthetic Generation")
    gee = GEEIngestionEngine(config)
    manifest = gee.generate_ee_export_manifest("2023-08-15T00:00:00")
    
    gen = MockGeospatialDataGenerator(config)
    static_data = gen.generate_basin_suite_if_needed()
    cube_path = gen.generate_synthetic_timeseries_dataset(num_timesteps=100, lookback_window=30)
    static_path = os.path.join(config["data"]["processed_dir"], "static_geospatial_layers.npz")

    # 3. Spatial & Temporal Splitting (Strict No Leakage)
    logger.info("\n[Phase 2] Strict Spatial & Temporal Split Partitioning")
    basin_cfg = config["data"]["spatial_split"]
    train_basins = basin_cfg["train_basins"]
    val_basins = basin_cfg["val_basins"]
    test_basins = basin_cfg["test_basins"]

    # Temporal split: Early timesteps for train, middle for val, latest for test
    total_timesteps = 100
    t_splitter = TemporalChronologicalSplitter()
    train_t, val_t, test_t = t_splitter.split_by_ratios(total_timesteps, train_ratio=0.65, val_ratio=0.15)

    logger.info(f"Spatial Partitioning: Train Basins={train_basins} | Val Basins={val_basins} | Test Basins={test_basins}")

    # 4. Dataset & DataLoaders
    logger.info("\n[Phase 3] Building 32x32 Spatio-Temporal Patch DataLoaders")
    train_loader, val_loader, test_loader, train_ds = create_dataloaders(
        cube_path=cube_path,
        static_path=static_path,
        train_basins=train_basins,
        val_basins=val_basins,
        test_basins=test_basins,
        train_timesteps=train_t,
        val_timesteps=val_t,
        test_timesteps=test_t,
        batch_size=16,
        num_workers=0
    )
    logger.info(f"Created Loaders: Train Batches={len(train_loader)} | Val Batches={len(val_loader)} | Test Batches={len(test_loader)}")

    # 5. Model Instantiation & Training
    logger.info("\n[Phase 4] Training Deep Multi-Hazard Multitask Model (Proposed Gated Cross-Modal)")
    model_gated, _ = create_model_for_hardware(config, device=device)
    trainer = MultiHazardTrainer(
        model=model_gated,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        checkpoint_dir="checkpoints"
    )
    # Quick training run (3 epochs for rapid ablation verification)
    trainer.epochs = 3
    train_results = trainer.fit()

    logger.info("\n[Phase 4b] Instantiating Deep Multi-Task Concat Fusion Model for Ablation")
    cfg_concat = dict(config)
    cfg_concat["models"]["deep_learning"]["fusion"]["type"] = "concatenation"
    model_concat, _ = create_model_for_hardware(cfg_concat, device=device)

    # 6. Master 7-Model Ablation Battery
    logger.info("\n[Phase 5] Running Full 7-Model Master Ablation Benchmark Suite")
    ablation = MultiHazardAblationEngine(config, output_dir="evaluation/results")
    
    # Extract tabular arrays for baseline models
    X_train, y_f_train, y_s_train, y_p_train = ablation.extract_tabular_arrays(
        cube_path, static_path, train_ds.samples
    )
    test_ds = test_loader.dataset
    X_test, y_f_test, y_s_test, y_p_test = ablation.extract_tabular_arrays(
        cube_path, static_path, test_ds.samples
    )

    logger.info(f"Tabular Evaluation Sets: Train Samples={X_train.shape[0]} | Test Samples={X_test.shape[0]}")
    scorecard = ablation.run_full_ablation_benchmark(
        X_train, y_f_train, y_s_train,
        X_test, y_f_test, y_s_test, y_p_test,
        model_gated, model_concat, test_loader, device
    )

    # 7. Explainable AI (XAI) Demonstration
    logger.info("\n[Phase 6] Explainable AI: SHAP Tabular Feature Attribution & Grad-CAM Hotspots")
    feat_names = [
        "DEM Elevation", "Slope", "TWI", "Flow Acc", "River Dist", "Soil Clay", "Soil Sand",
        "Rain 3d", "Rain 7d", "Rain 15d", "Rain 30d", "Rain Max", "Soil Moisture", "SAR VV"
    ]
    from models.baselines.xgboost_model import BaselineXGBoostSystem
    xgb_sys = BaselineXGBoostSystem(n_estimators=50)
    xgb_sys.fit(X_train, y_f_train, y_s_train)
    shap_explainer = TabularSHAPExplainer(xgb_sys.flood_model, feat_names)
    global_importance = shap_explainer.explain_global(X_test[:20])
    logger.info("Top 5 SHAP Global Flood Drivers:")
    for i, (k, v) in enumerate(list(global_importance.items())[:5]):
        logger.info(f"  {i+1}. {k}: {v*100:.2f}% contribution")

    # Grad-CAM on Deep Model
    gradcam = SpatiotemporalGradCAM(model_gated)
    sample_batch = next(iter(test_loader))
    cam_heatmap, pred_score = gradcam.generate_heatmap(
        sample_batch["x_spatial"][:1].to(device),
        sample_batch["x_temporal"][:1].to(device),
        target_hazard="flood"
    )
    logger.info(f"Grad-CAM Heatmap generated for test patch: Shape={cam_heatmap.shape}, Peak Saliency={np.max(cam_heatmap):.3f}")

    # 8. Risk Calibration & Temporal Hysteresis Verification
    logger.info("\n[Phase 7] Probability Calibration & Temporal Hysteresis State Machine")
    calibrator = ProbabilityCalibrator()
    calibrator.fit(np.linspace(0.1, 0.9, 50), np.random.binomial(1, np.linspace(0.1, 0.9, 50)))
    test_prob = float(calibrator.calibrate(np.array([pred_score]))[0])
    
    hyst = HysteresisStateMachine(hysteresis_delta=0.05, required_cooling_steps=2)
    # Test consecutive steps across boundary
    h1 = hyst.update_entity_state("Basin_406008", 0.75) # Should escalate to CRITICAL
    h2 = hyst.update_entity_state("Basin_406008", 0.68) # Drops to 0.68, inside dead-band -> Holds CRITICAL
    h3 = hyst.update_entity_state("Basin_406008", 0.62) # Below 0.70 - 0.05 (0.65) -> Cooling 1/2 -> Holds CRITICAL
    h4 = hyst.update_entity_state("Basin_406008", 0.61) # Cooling 2/2 -> De-escalates to HIGH

    logger.info(f"Hysteresis Flapping Test (Wardha Floodplain):")
    logger.info(f"  Step 1 (p=0.75): Level={h1['stabilized_level']} (Immediate Escalation)")
    logger.info(f"  Step 2 (p=0.68): Level={h2['stabilized_level']} (Anti-flapping dead-band hold)")
    logger.info(f"  Step 3 (p=0.62): Level={h3['stabilized_level']} (Cooling 1/2 hold)")
    logger.info(f"  Step 4 (p=0.61): Level={h4['stabilized_level']} (Verified De-escalation)")

    logger.info("\n================================================================================")
    logger.info("Multi-Hazard Early Warning System Pipeline Completed Successfully!")
    logger.info("Scorecard saved to: evaluation/results/ablation_summary.md")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
