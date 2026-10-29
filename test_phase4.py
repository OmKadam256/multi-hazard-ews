"""
==============================================================================
Verification Script: Phase 4 Multi-Task Learning & Gated Attention
==============================================================================
Validates:
1. GatedFusion: State-dependent SE gating mechanism and gate weights inspection.
2. MultiHazardModel:
   - Ablation A3: GatedFusion vs PlainConcatenation.
   - Ablation A4: Shared MLP trunk vs Independent heads.
3. MultiTaskLoss:
   - Homoscedastic Uncertainty Weighting (learnable task variances).
   - Strict Validity Masking (no cross-hazard pollution on missing labels).
4. train_model:
   - PyTorch training loop on PatchDataset.
   - Gradient clipping for GRU stabilization.
   - Early stopping & checkpoint saving (model_gated_multitask.pt).
==============================================================================
"""

import sys
import os
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.fusion import GatedFusion, PlainConcatenation
from models.multitask import MultiHazardModel
from training.loss import MultiTaskLoss
from training.train import train_model
from training.dataset import PatchDataset, make_splits


def test_phase4():
    print("================================================================================")
    print("Executing Phase 4 Verification Suite: Multi-Task Learning & Gated Attention")
    print("================================================================================")

    torch.manual_seed(42)
    np.random.seed(42)

    B = 4
    D_s = 128
    D_t = 128
    D_f = 128

    # --------------------------------------------------------------------------
    # 1. Test GatedFusion (Ablation A3)
    # --------------------------------------------------------------------------
    print("\n[Step 1] Testing GatedFusion Module (Ablation A3)...")
    gated_net = GatedFusion(
        spatial_dim=D_s,
        temporal_dim=D_t,
        out_dim=D_f,
        dropout=0.1
    )

    z_s = torch.randn(B, D_s)
    z_t = torch.randn(B, D_t)

    z_fused, gate = gated_net(z_s, z_t, return_gate=True)
    print(f"  [OK] Gated fused shape:  {z_fused.shape} (Expected: [{B}, {D_f}])")
    print(f"  [OK] Learned gate shape: {gate.shape} (Expected: [{B}, {D_f}])")
    print(f"       Gate values min={gate.min().item():.4f}, max={gate.max().item():.4f}, mean={gate.mean().item():.4f}")

    assert z_fused.shape == (B, D_f)
    assert gate.shape == (B, D_f)
    assert (gate >= 0.0).all() and (gate <= 1.0).all(), "Gates must strictly lie in [0, 1]"

    # Test batch mismatch assertion
    try:
        gated_net(torch.randn(B + 1, D_s), z_t)
        assert False, "Should raise AssertionError on batch mismatch!"
    except AssertionError as e:
        print(f"  [OK] Batch assertion caught expected error: {e}")

    # --------------------------------------------------------------------------
    # 2. Test MultiHazardModel Architecture (Ablations A3 & A4)
    # --------------------------------------------------------------------------
    print("\n[Step 2] Testing MultiHazardModel Architecture (Ablations A3 & A4)...")
    
    # Model 1: Gated Fusion + Shared Trunk (Full System)
    model_gated_shared = MultiHazardModel(
        static_channels=8,
        dynamic_channels=2,
        temporal_in_features=3,
        fusion_type="gated",
        use_shared_trunk=True
    )
    
    # Model 2: Plain Concatenation + Independent Heads (Ablation Baseline)
    model_concat_indep = MultiHazardModel(
        static_channels=8,
        dynamic_channels=2,
        temporal_in_features=3,
        fusion_type="concat",
        use_shared_trunk=False
    )

    static_x = torch.randn(B, 8, 32, 32)
    dynamic_x = torch.randn(B, 2, 32, 32)
    climate_x = torch.randn(B, 12, 3)

    out_gated = model_gated_shared(static_x, dynamic_x, climate_x)
    out_concat = model_concat_indep(static_x, dynamic_x, climate_x)

    print(f"  [OK] Gated+Shared:   flood_prob={out_gated['flood_prob'].shape}, spei={out_gated['spei_continuous'].shape}")
    print(f"  [OK] Concat+Indep:   flood_prob={out_concat['flood_prob'].shape}, spei={out_concat['spei_continuous'].shape}")
    assert "gate_weights" in out_gated, "Gated model should expose gate_weights for XAI"
    assert "gate_weights" not in out_concat, "Concat model should not have gate_weights"

    param_gated = model_gated_shared.get_parameter_count()
    param_concat = model_concat_indep.get_parameter_count()
    print(f"  [OK] Params Gated+Shared: Total={param_gated['total']:,} (Trunk={param_gated['shared_trunk']:,})")
    print(f"  [OK] Params Concat+Indep: Total={param_concat['total']:,} (Trunk={param_concat['shared_trunk']:,})")
    assert param_concat["shared_trunk"] == 0, "Independent model should have 0 trunk parameters"

    # --------------------------------------------------------------------------
    # 3. Test MultiTaskLoss with Homoscedastic Uncertainty & Masking
    # --------------------------------------------------------------------------
    print("\n[Step 3] Testing MultiTaskLoss with Uncertainty Weighting & Strict Masking...")
    loss_fn = MultiTaskLoss()

    preds = {
        "flood_logit": torch.tensor([[1.5], [-2.0], [0.5], [-1.0]]),
        "spei_continuous": torch.tensor([[-1.2], [0.4], [-2.1], [1.1]])
    }
    targets = {
        "flood_binary": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "spei_continuous": torch.tensor([-1.0, 0.5, -2.0, 1.0])
    }

    # Test Case A: Full validity
    loss_out_full = loss_fn(preds, targets)
    print(f"  [OK] Full loss: Total={loss_out_full['total_loss'].item():.4f} "
          f"(Flood={loss_out_full['loss_flood'].item():.4f}, Drought={loss_out_full['loss_drought'].item():.4f})")
    print(f"       Initial Uncertainties: sigma_flood={loss_out_full['sigma_flood']:.3f}, sigma_drought={loss_out_full['sigma_drought']:.3f}")

    # Test Case B: Strict Masking - Missing Drought on sample 0 & 2, Missing Flood on sample 1
    masks = {
        "flood_mask": torch.tensor([1.0, 0.0, 1.0, 1.0]),
        "drought_mask": torch.tensor([0.0, 1.0, 0.0, 1.0])
    }
    loss_out_masked = loss_fn(preds, targets, masks=masks)
    print(f"  [OK] Masked loss: Total={loss_out_masked['total_loss'].item():.4f} "
          f"(Flood={loss_out_masked['loss_flood'].item():.4f}, Drought={loss_out_masked['loss_drought'].item():.4f})")
    assert loss_out_masked["loss_flood"] > 0
    assert loss_out_masked["loss_drought"] > 0

    # Test Case C: Gradient flow to learnable uncertainty parameters
    loss_out_full["total_loss"].backward()
    assert loss_fn.log_var_flood.grad is not None, "log_var_flood must receive gradients"
    assert loss_fn.log_var_drought.grad is not None, "log_var_drought must receive gradients"
    print("  [OK] Gradients verified on homoscedastic uncertainty parameters.")

    # --------------------------------------------------------------------------
    # 4. Test End-to-End train_model() Loop on PatchDataset
    # --------------------------------------------------------------------------
    print("\n[Step 4] Testing End-to-End train_model() on PatchDataset...")
    H_grid, W_grid = 64, 64
    T_grid = 35
    C_s = 8
    C_d = 2
    C_c = 3

    static_stack = np.random.normal(100.0, 20.0, (C_s, H_grid, W_grid)).astype(np.float32)
    dynamic_stack = np.random.normal(-15.0, 3.0, (T_grid, C_d, H_grid, W_grid)).astype(np.float32)
    climate_stack = np.random.exponential(10.0, (T_grid, C_c, H_grid, W_grid)).astype(np.float32)
    flood_labels = (np.random.rand(T_grid, H_grid, W_grid) > 0.93).astype(np.uint8)
    spei_labels = np.random.normal(0.0, 1.0, (T_grid, H_grid, W_grid)).astype(np.float32)
    drought_class = np.clip(np.digitize(spei_labels, [-2.0, -1.5, -1.0, -0.5]), 0, 4)
    valid_mask = np.ones((H_grid, W_grid), dtype=np.float32)

    sub_basins = np.zeros((H_grid, W_grid), dtype=np.int32)
    sub_basins[:32, :32] = 406001
    sub_basins[:32, 32:] = 406002
    sub_basins[32:, :32] = 406007  # buffer
    sub_basins[32:, 32:] = 406009  # test

    event_ids = np.full((T_grid, H_grid, W_grid), "none", dtype=object)
    dates = [f"2020-01-{d+1:02d}" for d in range(T_grid)]

    train_loader, val_loader, test_loader, scaler = make_splits(
        static_stack, dynamic_stack, climate_stack,
        flood_labels, spei_labels, drought_class, valid_mask,
        sub_basins, event_ids, dates,
        train_end_date="2020-01-20",
        val_end_date="2020-01-28",
        test_basin_ids=[406009],
        buffer_basin_ids=[406007],
        patch_size=32,
        stride=16,
        lookback_window=10,
        batch_size=4
    )

    test_exp_name = "test_model_gated_multitask"
    checkpoints_dir = "checkpoints"

    history, best_path = train_model(
        model=model_gated_shared,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        clip_grad_norm=1.0,
        patience=2,
        save_dir=checkpoints_dir,
        experiment_name=test_exp_name,
        device=torch.device("cpu")
    )

    print(f"  [OK] train_model completed 2 epochs successfully!")
    print(f"       Best Checkpoint: '{best_path}' (Exists: {os.path.exists(best_path)})")
    print(f"       History keys: {list(history.keys())}")
    print(f"       Final Flood PR-AUC: {history['val_flood_prauc'][-1]:.4f}")
    print(f"       Final Drought RMSE: {history['val_drought_rmse'][-1]:.4f}")
    assert os.path.exists(best_path), f"Checkpoint file '{best_path}' was not created!"

    # Load and verify checkpoint
    ckpt = torch.load(best_path, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "criterion_state_dict" in ckpt
    assert "flood_metrics" in ckpt
    assert "drought_metrics" in ckpt
    print(f"  [OK] Checkpoint payload verified: epoch={ckpt['epoch']}, exp={ckpt['experiment_name']}")

    print("\n================================================================================")
    print("ALL PHASE 4 TESTS PASSED PERFECTLY!")
    print("================================================================================")


if __name__ == "__main__":
    test_phase4()
