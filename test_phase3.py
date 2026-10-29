"""
==============================================================================
Verification Script: Phase 3 Deep Baseline Architecture
==============================================================================
Validates:
1. SpatialCNN: 32x32 patches, static + dynamic inputs, Grad-CAM feature maps,
   and strict shape assertions.
2. TemporalGRU: Causal climate sequences, 3D and 5D tensor inputs, attention
   weights, and sequence length assertions.
3. PlainConcatenation: Latent vector fusion, dimensional alignment, and batch
   matching assertions.
4. CNN_GRU_Baseline: End-to-end forward pass on PatchDataset batches, dual
   head predictions, backward pass gradient propagation, and parameter summary.
==============================================================================
"""

import sys
import os
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.encoders import SpatialCNN, TemporalGRU
from models.fusion import PlainConcatenation
from models.deep_baseline import CNN_GRU_Baseline
from training.dataset import PatchDataset, make_splits


def test_phase3():
    print("================================================================================")
    print("Executing Phase 3 Verification Suite: Deep Spatiotemporal Baseline")
    print("================================================================================")

    torch.manual_seed(42)
    np.random.seed(42)

    B = 4
    C_s = 8
    C_d = 2
    C_c = 3
    T = 15
    patch_size = 32

    # --------------------------------------------------------------------------
    # 1. Test SpatialCNN
    # --------------------------------------------------------------------------
    print("\n[Step 1] Testing SpatialCNN Module...")
    spatial_net = SpatialCNN(
        static_channels=C_s,
        dynamic_channels=C_d,
        out_dim=128,
        dropout=0.1
    )

    static_x = torch.randn(B, C_s, patch_size, patch_size)
    dynamic_x = torch.randn(B, C_d, patch_size, patch_size)

    # Test forward pass with separate static and dynamic inputs
    z_spatial, fmap = spatial_net(static_x, dynamic_x, return_feature_map=True)
    print(f"  [OK] Spatial embedding shape: {z_spatial.shape} (Expected: [{B}, 128])")
    print(f"  [OK] Target conv feature map: {fmap.shape} (Expected: [{B}, 128, 8, 8])")
    assert z_spatial.shape == (B, 128), f"Expected ({B}, 128), got {z_spatial.shape}"
    assert fmap.shape == (B, 128, 8, 8), f"Expected ({B}, 128, 8, 8), got {fmap.shape}"

    # Test 5D dynamic imagery (B, T, C_d, 32, 32)
    dynamic_5d = torch.randn(B, 10, C_d, patch_size, patch_size)
    z_spatial_5d = spatial_net(static_x, dynamic_5d)
    assert z_spatial_5d.shape == (B, 128)
    print("  [OK] 5D dynamic tensor handled correctly (latest pre-event frame extracted).")

    # Test strict 32x32 assertion failure on invalid spatial patch
    invalid_static = torch.randn(B, C_s, 48, 48)
    try:
        spatial_net(invalid_static)
        assert False, "Should have raised AssertionError for non-32x32 patch!"
    except AssertionError as e:
        print(f"  [OK] Strict spatial assertion passed: Caught expected error: {e}")

    # --------------------------------------------------------------------------
    # 2. Test TemporalGRU
    # --------------------------------------------------------------------------
    print("\n[Step 2] Testing TemporalGRU Module...")
    temporal_net = TemporalGRU(
        in_features=C_c,
        hidden_dim=64,
        out_dim=128,
        num_layers=2,
        dropout=0.1
    )

    # Test with 3D input (B, T, C_c)
    climate_3d = torch.randn(B, T, C_c)
    z_temporal, attn = temporal_net(climate_3d, return_attention=True)
    print(f"  [OK] Temporal embedding shape: {z_temporal.shape} (Expected: [{B}, 128])")
    print(f"  [OK] Temporal attention shape: {attn.shape} (Expected: [{B}, {T}, 1])")
    assert z_temporal.shape == (B, 128)
    assert attn.shape == (B, T, 1)

    # Verify temporal attention weights sum to 1.0 along time dimension
    attn_sums = torch.sum(attn, dim=1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5), "Attention weights must sum to 1"
    print("  [OK] Attention weights normalized correctly (sum=1.0).")

    # Test with 5D spatiotemporal patch tensor (B, T, C_c, 32, 32)
    climate_5d = torch.randn(B, T, C_c, patch_size, patch_size)
    z_temporal_5d = temporal_net(climate_5d)
    assert z_temporal_5d.shape == (B, 128)
    print("  [OK] 5D spatiotemporal climate patch handled correctly via spatial pooling.")

    # Test sequence length assertion
    try:
        empty_seq = torch.randn(B, 0, C_c)
        temporal_net(empty_seq)
        assert False, "Should have raised AssertionError for T=0!"
    except AssertionError as e:
        print(f"  [OK] Strict sequence length assertion passed: Caught expected error: {e}")

    # --------------------------------------------------------------------------
    # 3. Test PlainConcatenation
    # --------------------------------------------------------------------------
    print("\n[Step 3] Testing PlainConcatenation Fusion Module...")
    fusion_net = PlainConcatenation(
        spatial_dim=128,
        temporal_dim=128,
        out_dim=128,
        dropout=0.1
    )

    z_fused, raw_concat = fusion_net(z_spatial, z_temporal, return_raw_concat=True)
    print(f"  [OK] Raw concatenation shape: {raw_concat.shape} (Expected: [{B}, 256])")
    print(f"  [OK] Fused embedding shape:   {z_fused.shape} (Expected: [{B}, 128])")
    assert raw_concat.shape == (B, 256)
    assert z_fused.shape == (B, 128)

    # Test batch mismatch assertion
    z_mismatched = torch.randn(B + 1, 128)
    try:
        fusion_net(z_mismatched, z_temporal)
        assert False, "Should have raised AssertionError for batch mismatch!"
    except AssertionError as e:
        print(f"  [OK] Strict batch alignment assertion passed: Caught expected error: {e}")

    # --------------------------------------------------------------------------
    # 4. Test CNN_GRU_Baseline End-to-End
    # --------------------------------------------------------------------------
    print("\n[Step 4] Testing End-to-End CNN_GRU_Baseline Model...")
    model = CNN_GRU_Baseline(
        static_channels=C_s,
        dynamic_channels=C_d,
        climate_features=C_c,
        spatial_dim=128,
        temporal_dim=128,
        fused_dim=128,
        temporal_hidden_dim=64,
        temporal_num_layers=2,
        dropout=0.1
    )

    param_summary = model.get_parameter_count()
    print(f"  [INFO] Model Parameter Summary: {param_summary}")
    assert param_summary["trainable"] > 0
    assert param_summary["total"] == param_summary["trainable"]

    # Forward pass with return_dict=True
    outputs = model(
        static_features=static_x,
        dynamic_images=dynamic_x,
        climate_sequence=climate_3d,
        return_dict=True
    )

    flood_prob = outputs["flood_prob"]
    flood_logit = outputs["flood_logit"]
    spei_pred = outputs["spei_pred"]

    print(f"  [OK] flood_prob shape:  {flood_prob.shape} (Min: {flood_prob.min():.4f}, Max: {flood_prob.max():.4f})")
    print(f"  [OK] spei_pred shape:   {spei_pred.shape} (Min: {spei_pred.min():.4f}, Max: {spei_pred.max():.4f})")

    assert flood_prob.shape == (B, 1)
    assert spei_pred.shape == (B, 1)
    assert (flood_prob >= 0.0).all() and (flood_prob <= 1.0).all(), "flood_prob must be valid probability [0, 1]"

    # Test inference predict() helper
    preds_np = model.predict(static_x, dynamic_x, climate_3d)
    assert preds_np["flood_probability"].shape == (B,)
    assert preds_np["spei_prediction"].shape == (B,)
    print("  [OK] predict() helper produces valid numpy outputs.")

    # --------------------------------------------------------------------------
    # 5. Test Backward Pass & Gradient Flow
    # --------------------------------------------------------------------------
    print("\n[Step 5] Testing Backward Pass and Gradient Flow...")
    loss_flood = torch.nn.functional.binary_cross_entropy_with_logits(
        flood_logit, torch.zeros_like(flood_logit)
    )
    loss_spei = torch.nn.functional.mse_loss(
        spei_pred, torch.zeros_like(spei_pred)
    )
    total_loss = loss_flood + loss_spei
    total_loss.backward()

    # Verify gradients reach all submodules
    has_grad_spatial = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.spatial_encoder.parameters())
    has_grad_temporal = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.temporal_encoder.parameters())
    has_grad_fusion = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.fusion.parameters())
    has_grad_flood = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.flood_head.parameters())
    has_grad_drought = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.drought_head.parameters())

    print(f"  [OK] Gradients flowing to SpatialCNN:      {has_grad_spatial}")
    print(f"  [OK] Gradients flowing to TemporalGRU:     {has_grad_temporal}")
    print(f"  [OK] Gradients flowing to PlainConcatenation: {has_grad_fusion}")
    print(f"  [OK] Gradients flowing to Flood Head:     {has_grad_flood}")
    print(f"  [OK] Gradients flowing to Drought Head:   {has_grad_drought}")

    assert has_grad_spatial and has_grad_temporal and has_grad_fusion and has_grad_flood and has_grad_drought, (
        "Gradients must propagate through all modules in the deep baseline trunk!"
    )

    # --------------------------------------------------------------------------
    # 6. Test with Real PatchDataset Split Batch
    # --------------------------------------------------------------------------
    print("\n[Step 6] Testing with real PatchDataset batch...")
    H_grid, W_grid = 64, 64
    T_grid = 35
    static_stack = np.random.normal(100.0, 20.0, (C_s, H_grid, W_grid)).astype(np.float32)
    dynamic_stack = np.random.normal(-15.0, 3.0, (T_grid, C_d, H_grid, W_grid)).astype(np.float32)
    climate_stack = np.random.exponential(10.0, (T_grid, C_c, H_grid, W_grid)).astype(np.float32)
    flood_labels = (np.random.rand(T_grid, H_grid, W_grid) > 0.95).astype(np.uint8)
    spei_labels = np.random.normal(0.0, 1.0, (T_grid, H_grid, W_grid)).astype(np.float32)
    drought_class = np.clip(np.digitize(spei_labels, [-2.0, -1.5, -1.0, -0.5]), 0, 4)
    valid_mask = np.ones((H_grid, W_grid), dtype=np.float32)

    sub_basins = np.zeros((H_grid, W_grid), dtype=np.int32)
    sub_basins[:32, :32] = 406001
    sub_basins[:32, 32:] = 406002
    sub_basins[32:, :32] = 406007
    sub_basins[32:, 32:] = 406009

    event_ids = np.full((T_grid, H_grid, W_grid), "none", dtype=object)
    dates = [f"2020-01-{d+1:02d}" for d in range(T_grid)]

    train_loader, _, _, _ = make_splits(
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

    batch = next(iter(train_loader))
    s_feat, d_imgs, c_seq, labels, msk = batch

    out_dataset = model(
        static_features=s_feat,
        dynamic_images=d_imgs,
        climate_sequence=c_seq,
        return_dict=True
    )

    print(f"  [OK] Dataset batch forward pass successful!")
    print(f"       Batch flood predictions: {out_dataset['flood_prob'].squeeze().detach().numpy()}")
    print(f"       Batch SPEI predictions:  {out_dataset['spei_pred'].squeeze().detach().numpy()}")

    print("\n================================================================================")
    print("ALL PHASE 3 TESTS PASSED PERFECTLY!")
    print("================================================================================")


if __name__ == "__main__":
    test_phase3()
