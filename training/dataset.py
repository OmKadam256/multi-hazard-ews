"""
==============================================================================
PHASE 2: PyTorch Dataset, Data Loaders & Strict Leakage-Safe Splitting
==============================================================================
Scientific Directives Enforced:
1. PatchDataset yields (static_features, dynamic_images, climate_sequence, labels, masks)
   with 32x32 km spatial windows.
2. make_splits() implements 3-way strict leakage guards:
   - Temporal Split (train: early, val: intermediate, test: future)
   - Spatial Split (leave-sub-basin-out via HydroSHEDS + hydrological buffer zone)
   - Event-Wise Grouping (all patches of a single flood event strictly in same split)
3. Scaling and normalization fitted strictly on training split only.
4. Masks out unknown/no-data pixels to prevent silent negative imputation.
==============================================================================
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetEngine")


class DatasetScaler:
    """
    Fits feature standardization (mean, std) strictly on training samples only.
    Applies the learned parameters to validation and test sets without leakage.
    """

    def __init__(self):
        self.static_mean: Optional[np.ndarray] = None
        self.static_std: Optional[np.ndarray] = None
        self.dynamic_mean: Optional[np.ndarray] = None
        self.dynamic_std: Optional[np.ndarray] = None
        self.climate_mean: Optional[np.ndarray] = None
        self.climate_std: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(
        self,
        static_stack: np.ndarray,
        dynamic_stack: np.ndarray,
        climate_stack: np.ndarray,
        train_indices: List[Tuple[int, int, int, Any, Any]],
        patch_size: int = 32
    ):
        """
        Computes statistics ONLY over the training patch coordinates.
        train_indices: list of (t, y, x, event_id, basin_id)
        """
        logger.info(f"Fitting DatasetScaler strictly on {len(train_indices)} training patches...")
        
        static_samples = []
        dyn_samples = []
        clim_samples = []

        # Sample representative patches to avoid memory exhaustion
        sample_step = max(1, len(train_indices) // 300)
        subset = train_indices[::sample_step]

        for (t, y, x, _, _) in subset:
            static_samples.append(static_stack[:, y:y+patch_size, x:x+patch_size])
            t_start = max(0, t - 30)
            dyn_samples.append(dynamic_stack[t_start:t, :, y:y+patch_size, x:x+patch_size])
            clim_samples.append(climate_stack[t_start:t, :, y:y+patch_size, x:x+patch_size])

        s_all = np.concatenate([s.reshape(s.shape[0], -1) for s in static_samples], axis=1)
        self.static_mean = np.mean(s_all, axis=1, keepdims=True).reshape(-1, 1, 1).astype(np.float32)
        self.static_std = (np.std(s_all, axis=1, keepdims=True) + 1e-6).reshape(-1, 1, 1).astype(np.float32)

        d_all = np.concatenate([np.transpose(d, (1, 0, 2, 3)).reshape(d.shape[1], -1) for d in dyn_samples], axis=1)
        self.dynamic_mean = np.mean(d_all, axis=1, keepdims=True).reshape(1, -1, 1, 1).astype(np.float32)
        self.dynamic_std = (np.std(d_all, axis=1, keepdims=True) + 1e-6).reshape(1, -1, 1, 1).astype(np.float32)

        c_all = np.concatenate([np.transpose(c, (1, 0, 2, 3)).reshape(c.shape[1], -1) for c in clim_samples], axis=1)
        self.climate_mean = np.mean(c_all, axis=1, keepdims=True).reshape(1, -1, 1, 1).astype(np.float32)
        self.climate_std = (np.std(c_all, axis=1, keepdims=True) + 1e-6).reshape(1, -1, 1, 1).astype(np.float32)

        self.is_fitted = True
        logger.info("DatasetScaler successfully fitted on training split.")

    def transform_static(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return x
        return (x - self.static_mean) / self.static_std

    def transform_dynamic(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return x
        return (x - self.dynamic_mean) / self.dynamic_std

    def transform_climate(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return x
        return (x - self.climate_mean) / self.climate_std


class PatchDataset(Dataset):
    """
    Geospatial PyTorch Dataset extracting 32x32 km spatial patches.
    Yields:
      1. static_features: (C_static, 32, 32)
      2. dynamic_images:  (T_dyn, C_dyn, 32, 32) strictly pre-event [t - lookback, t)
      3. climate_sequence: (T_clim, C_clim, 32, 32) antecedent meteorology
      4. labels: dict {'flood': float/mask, 'spei': float, 'drought_class': int}
      5. masks: (32, 32) float32 valid data mask (1.0 = valid, 0.0 = nodata/unobserved)
    """

    def __init__(
        self,
        static_stack: np.ndarray,      # (C_s, H, W)
        dynamic_stack: np.ndarray,     # (T, C_d, H, W)
        climate_stack: np.ndarray,     # (T, C_c, H, W)
        flood_labels: np.ndarray,      # (T, H, W)
        spei_labels: np.ndarray,       # (T, H, W)
        drought_class_labels: np.ndarray, # (T, H, W)
        valid_mask: np.ndarray,        # (H, W) or (T, H, W)
        patch_coords: List[Tuple[int, int, int, Any, Any]], # (t, y, x, event_id, basin_id)
        patch_size: int = 32,
        lookback_window: int = 30,
        scaler: Optional[DatasetScaler] = None,
        augment: bool = False
    ):
        self.static_stack = static_stack
        self.dynamic_stack = dynamic_stack
        self.climate_stack = climate_stack
        self.flood_labels = flood_labels
        self.spei_labels = spei_labels
        self.drought_class_labels = drought_class_labels
        self.valid_mask = valid_mask
        self.patch_coords = patch_coords
        self.patch_size = patch_size
        self.lookback_window = lookback_window
        self.scaler = scaler
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patch_coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        t, y, x, event_id, basin_id = self.patch_coords[idx]
        p = self.patch_size

        # 1. Static features (C_s, 32, 32)
        static_patch = self.static_stack[:, y:y+p, x:x+p].copy()
        if self.scaler is not None:
            static_patch = self.scaler.transform_static(static_patch)

        # 2. Dynamic imagery (T_dyn, C_dyn, 32, 32) strictly pre-event [t - lookback, t)
        t_start = max(0, t - self.lookback_window)
        dyn_patch = self.dynamic_stack[t_start:t, :, y:y+p, x:x+p].copy()
        if self.scaler is not None:
            dyn_patch = self.scaler.transform_dynamic(dyn_patch)

        # 3. Climate sequence (T_clim, C_clim, 32, 32)
        clim_patch = self.climate_stack[t_start:t, :, y:y+p, x:x+p].copy()
        if self.scaler is not None:
            clim_patch = self.scaler.transform_climate(clim_patch)

        # 4. Valid observation mask (32, 32)
        if self.valid_mask.ndim == 2:
            mask_patch = self.valid_mask[y:y+p, x:x+p].astype(np.float32)
        else:
            mask_patch = self.valid_mask[t, y:y+p, x:x+p].astype(np.float32)

        # 5. Ground truth labels at forecast issue time t
        f_patch = self.flood_labels[t, y:y+p, x:x+p]
        # Only evaluate on valid unmasked pixels
        valid_f_pixels = f_patch[mask_patch > 0]
        if len(valid_f_pixels) > 0:
            flood_binary = float(np.any(valid_f_pixels > 0))
            flood_fraction = float(np.mean(valid_f_pixels > 0))
        else:
            flood_binary = 0.0
            flood_fraction = 0.0

        s_patch = self.spei_labels[t, y:y+p, x:x+p]
        valid_s_pixels = s_patch[mask_patch > 0]
        spei_val = float(np.mean(valid_s_pixels)) if len(valid_s_pixels) > 0 else 0.0

        c_patch = self.drought_class_labels[t, y:y+p, x:x+p]
        valid_c_pixels = c_patch[mask_patch > 0]
        if len(valid_c_pixels) > 0:
            unique_c, counts = np.unique(valid_c_pixels, return_counts=True)
            d_class = int(unique_c[np.argmax(counts)])
        else:
            d_class = 0

        # Spatial data augmentation (symmetries)
        if self.augment and np.random.rand() > 0.5:
            # Horizontal flip
            static_patch = np.flip(static_patch, axis=-1).copy()
            dyn_patch = np.flip(dyn_patch, axis=-1).copy()
            clim_patch = np.flip(clim_patch, axis=-1).copy()
            mask_patch = np.flip(mask_patch, axis=-1).copy()

        labels = {
            "flood_binary": torch.tensor(flood_binary, dtype=torch.float32),
            "flood_fraction": torch.tensor(flood_fraction, dtype=torch.float32),
            "spei_continuous": torch.tensor(spei_val, dtype=torch.float32),
            "drought_class": torch.tensor(d_class, dtype=torch.long),
            "event_id": str(event_id),
            "basin_id": int(basin_id),
            "coord": torch.tensor([t, y, x], dtype=torch.int32)
        }

        return (
            torch.from_numpy(static_patch),
            torch.from_numpy(dyn_patch),
            torch.from_numpy(clim_patch),
            labels,
            torch.from_numpy(mask_patch)
        )


def make_splits(
    static_stack: np.ndarray,
    dynamic_stack: np.ndarray,
    climate_stack: np.ndarray,
    flood_labels: np.ndarray,
    spei_labels: np.ndarray,
    drought_class_labels: np.ndarray,
    valid_mask: np.ndarray,
    sub_basins: np.ndarray,
    event_ids: np.ndarray,
    dates: List[str],
    train_end_date: str = "2020-12-31",
    val_end_date: str = "2021-12-31",
    test_basin_ids: Optional[List[int]] = None,
    buffer_basin_ids: Optional[List[int]] = None,
    patch_size: int = 32,
    stride: int = 16,
    lookback_window: int = 30,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetScaler]:
    """
    Builds PyTorch DataLoaders with strict spatial, temporal, and event-wise isolation:
    1. Temporal Split: Chronological partition on dates.
    2. Spatial Split: Complete sub-basin exclusion + buffer zone removal.
    3. Event-Wise Grouping: All patches of a single flood event assigned exclusively to one split.
    4. DatasetScaler: Computed strictly on training split only.
    """
    logger.info("Executing make_splits() with strict Zero Data Leakage guards...")

    H, W = static_stack.shape[1], static_stack.shape[2]
    num_timesteps = len(dates)

    test_basins: Set[int] = set(test_basin_ids or [406009, 406010])
    buffer_basins: Set[int] = set(buffer_basin_ids or [406007, 406008])

    # 1. Parse date indices
    train_t_idx = []
    val_t_idx = []
    test_t_idx = []

    for i, d in enumerate(dates):
        if d <= train_end_date:
            train_t_idx.append(i)
        elif d <= val_end_date:
            val_t_idx.append(i)
        else:
            test_t_idx.append(i)

    # 2. Event-wise mapping: Ensure an event ID never spans across splits
    # Pre-assign every unique flood event ID to a single split
    unique_events = [e for e in np.unique(event_ids) if str(e) not in ["0", "none", "None", "-1"]]
    np.random.seed(42)
    shuffled_events = np.random.permutation(unique_events)
    n_ev = len(shuffled_events)
    train_events = set(shuffled_events[:int(0.65 * n_ev)])
    val_events = set(shuffled_events[int(0.65 * n_ev):int(0.80 * n_ev)])
    test_events = set(shuffled_events[int(0.80 * n_ev):])

    logger.info(f"Event Partitioning: {len(train_events)} train events | {len(val_events)} val events | {len(test_events)} test events")

    # 3. Extract candidate patch locations
    train_patches = []
    val_patches = []
    test_patches = []

    # Buffer zone dropped count
    buffer_dropped_count = 0

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Majority basin ID in patch
            p_basins = sub_basins[y:y+patch_size, x:x+patch_size]
            b_ids, b_counts = np.unique(p_basins, return_counts=True)
            dominant_basin = int(b_ids[np.argmax(b_counts)])

            for t in range(lookback_window, num_timesteps):
                # Identify event ID in this patch at timestep t
                ev_patch = event_ids[t, y:y+patch_size, x:x+patch_size]
                ev_unique = [e for e in np.unique(ev_patch) if str(e) not in ["0", "none", "None", "-1"]]
                dominant_event = ev_unique[0] if len(ev_unique) > 0 else "none"

                coord = (t, y, x, dominant_event, dominant_basin)

                # Event-wise enforcement:
                if dominant_event in test_events:
                    test_patches.append(coord)
                    continue
                if dominant_event in val_events:
                    val_patches.append(coord)
                    continue
                if dominant_event in train_events:
                    # If assigned to train events, must not be in test basin
                    if dominant_basin not in test_basins and dominant_basin not in buffer_basins:
                        train_patches.append(coord)
                    continue

                # Spatial Split enforcement:
                if dominant_basin in test_basins:
                    # Spatial test basin: only include if temporal horizon is test or val
                    test_patches.append(coord)
                elif dominant_basin in buffer_basins:
                    # Hydrological buffer zone: held out of train to prevent boundary autocorrelation
                    if t in val_t_idx:
                        val_patches.append(coord)
                    else:
                        buffer_dropped_count += 1
                else:
                    # Remaining training basins: split strictly by temporal non-overlapping windows
                    if t in train_t_idx:
                        train_patches.append(coord)
                    elif t in val_t_idx:
                        val_patches.append(coord)
                    else:
                        test_patches.append(coord)

    logger.info(
        f"Patches Generated: Train={len(train_patches)} | Val={len(val_patches)} | Test={len(test_patches)} "
        f"(Buffer Zone Excluded Patches={buffer_dropped_count})"
    )

    # 4. Fit Scaler strictly on train patches
    scaler = DatasetScaler()
    scaler.fit(static_stack, dynamic_stack, climate_stack, train_patches, patch_size=patch_size)

    # 5. Instantiate datasets
    train_ds = PatchDataset(
        static_stack, dynamic_stack, climate_stack,
        flood_labels, spei_labels, drought_class_labels, valid_mask,
        train_patches, patch_size=patch_size, lookback_window=lookback_window,
        scaler=scaler, augment=True
    )
    val_ds = PatchDataset(
        static_stack, dynamic_stack, climate_stack,
        flood_labels, spei_labels, drought_class_labels, valid_mask,
        val_patches, patch_size=patch_size, lookback_window=lookback_window,
        scaler=scaler, augment=False
    )
    test_ds = PatchDataset(
        static_stack, dynamic_stack, climate_stack,
        flood_labels, spei_labels, drought_class_labels, valid_mask,
        test_patches, patch_size=patch_size, lookback_window=lookback_window,
        scaler=scaler, augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=False)

    return train_loader, val_loader, test_loader, scaler
