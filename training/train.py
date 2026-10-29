"""
==============================================================================
PHASE 4: PyTorch Multi-Hazard Training Loop with Gradient Clipping & Early Stopping
==============================================================================
Scientific Directives Enforced:
1. train_model() Function:
   - Loops over PatchDataset dataloaders with strict batch unpacking.
   - Dual-T4 PyTorch AMP (Automatic Mixed Precision) and DataParallel support.
   - Gradient clipping (clip_grad_norm) to guarantee recurrent GRU numerical stability.
   - MultiTaskLoss with Homoscedastic Uncertainty Weighting.
2. Scientific Evaluation & Early Stopping:
   - Evaluates validation PR-AUC, POD@FAR=0.10, CSI, and Drought RMSE every epoch.
   - Implements composite early stopping preventing overfit on either hazard.
   - Saves model checkpoints named after specific ablation experiments:
     e.g., 'model_concat_independent.pt' vs 'model_gated_multitask.pt'.
==============================================================================
"""

import os
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from training.loss import MultiTaskLoss
from evaluation.metrics import evaluate_flood, evaluate_drought

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainEngine")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Optional[MultiTaskLoss] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    num_epochs: int = 20,
    clip_grad_norm: float = 1.0,
    patience: int = 5,
    save_dir: str = "checkpoints",
    experiment_name: str = "model_gated_multitask",
    device: Optional[torch.device] = None,
    use_amp: bool = True
) -> Tuple[Dict[str, List[float]], str]:
    """
    Executes multi-hazard training loop with early stopping and checkpointing.
    
    Args:
        model: MultiHazardModel or CNN_GRU_Baseline nn.Module
        train_loader: DataLoader yielding PatchDataset batches
        val_loader: DataLoader for validation split
        criterion: MultiTaskLoss with Homoscedastic Uncertainty Weighting
        optimizer: AdamW or Adam optimizer
        scheduler: Optional learning rate scheduler
        num_epochs: Maximum epochs to train
        clip_grad_norm: Maximum gradient norm for GRU stabilization
        patience: Consecutive epochs without validation improvement before early stop
        save_dir: Directory where model checkpoints are saved
        experiment_name: Experiment identifier for checkpoint naming
        device: torch.device (CUDA or CPU)
        use_amp: Whether to use Automatic Mixed Precision
    
    Returns:
        history: Dictionary of train and validation metric histories
        best_checkpoint_path: Path to the saved best model checkpoint (.pt)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # 1. Initialize criterion with learnable uncertainty parameters
    if criterion is None:
        criterion = MultiTaskLoss()
    criterion = criterion.to(device)

    # 2. Initialize optimizer including both model and uncertainty parameters
    if optimizer is None:
        params = list(model.parameters()) + list(criterion.parameters())
        optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 3. Automatic Mixed Precision Scaler
    amp_enabled = use_amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    os.makedirs(save_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(save_dir, f"{experiment_name}.pt")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_flood_loss": [],
        "train_drought_loss": [],
        "val_loss": [],
        "val_flood_prauc": [],
        "val_flood_csi": [],
        "val_drought_rmse": [],
        "sigma_flood": [],
        "sigma_drought": [],
        "composite_score": []
    }

    best_score = -float("inf")
    patience_counter = 0

    logger.info(f"Initiating Training for Experiment: '{experiment_name}' on {device}")
    logger.info(f"Batches: Train={len(train_loader)} | Val={len(val_loader)} | Mixed Precision={amp_enabled}")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # ======================================================================
        # 1. TRAINING LOOP
        # ======================================================================
        model.train()
        criterion.train()

        running_train_loss = 0.0
        running_flood_loss = 0.0
        running_drought_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            # Unpack PatchDataset batch
            static_feats, dyn_imgs, clim_seq, labels_dict, masks = batch

            static_feats = static_feats.to(device, non_blocking=True)
            dyn_imgs = dyn_imgs.to(device, non_blocking=True)
            clim_seq = clim_seq.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Move label tensors to device
            labels_on_device = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in labels_dict.items()
            }

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                preds = model(
                    static_features=static_feats,
                    dynamic_images=dyn_imgs,
                    climate_sequence=clim_seq,
                    return_dict=True
                )
                loss_dict = criterion(
                    predictions=preds,
                    targets=labels_on_device,
                    masks=masks
                )
                loss = loss_dict["total_loss"]

            # Backward pass with scaled gradients and GRU clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Clip gradients to prevent exploding gradients in recurrent units
            all_params = list(model.parameters()) + list(criterion.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            running_flood_loss += loss_dict["loss_flood"].item()
            running_drought_loss += loss_dict["loss_drought"].item()
            train_batches += 1

        avg_train_loss = running_train_loss / max(1, train_batches)
        avg_flood_loss = running_flood_loss / max(1, train_batches)
        avg_drought_loss = running_drought_loss / max(1, train_batches)

        # ======================================================================
        # 2. VALIDATION LOOP & SCIENTIFIC METRIC EVALUATION
        # ======================================================================
        model.eval()
        criterion.eval()

        running_val_loss = 0.0
        val_batches = 0

        val_flood_preds = []
        val_flood_targets = []
        val_drought_preds = []
        val_drought_targets = []
        val_masks_list = []

        with torch.no_grad():
            for batch in val_loader:
                static_feats, dyn_imgs, clim_seq, labels_dict, masks = batch

                static_feats = static_feats.to(device, non_blocking=True)
                dyn_imgs = dyn_imgs.to(device, non_blocking=True)
                clim_seq = clim_seq.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                labels_on_device = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in labels_dict.items()
                }

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    preds = model(
                        static_features=static_feats,
                        dynamic_images=dyn_imgs,
                        climate_sequence=clim_seq,
                        return_dict=True
                    )
                    loss_dict = criterion(
                        predictions=preds,
                        targets=labels_on_device,
                        masks=masks
                    )

                running_val_loss += loss_dict["total_loss"].item()
                val_batches += 1

                # Gather predictions for scientific evaluation metrics
                flood_prob = preds["flood_prob"].view(-1).cpu().numpy()
                flood_true = labels_dict["flood_binary"].view(-1).cpu().numpy()
                spei_pred = preds["spei_continuous"].view(-1).cpu().numpy()
                spei_true = labels_dict["spei_continuous"].view(-1).cpu().numpy()
                mask_np = (masks.view(masks.shape[0], -1).sum(dim=1) > 0).cpu().numpy().astype(float)

                val_flood_preds.append(flood_prob)
                val_flood_targets.append(flood_true)
                val_drought_preds.append(spei_pred)
                val_drought_targets.append(spei_true)
                val_masks_list.append(mask_np)

        avg_val_loss = running_val_loss / max(1, val_batches)

        # Concatenate full validation set predictions
        all_val_flood_preds = np.concatenate(val_flood_preds) if val_flood_preds else np.array([])
        all_val_flood_targets = np.concatenate(val_flood_targets) if val_flood_targets else np.array([])
        all_val_drought_preds = np.concatenate(val_drought_preds) if val_drought_preds else np.array([])
        all_val_drought_targets = np.concatenate(val_drought_targets) if val_drought_targets else np.array([])
        all_val_masks = np.concatenate(val_masks_list) if val_masks_list else np.array([])

        # Compute rigorous metrics (No raw accuracy!)
        flood_eval = evaluate_flood(
            y_true=all_val_flood_targets,
            y_prob=all_val_flood_preds,
            mask=all_val_masks
        )
        drought_eval = evaluate_drought(
            y_true_spei=all_val_drought_targets,
            y_pred_spei=all_val_drought_preds,
            mask=all_val_masks
        )

        val_prauc = flood_eval.get("pr_auc", 0.0)
        val_csi = flood_eval.get("csi", 0.0)
        val_rmse = drought_eval.get("rmse", 999.0)

        # Learned uncertainty values
        uncertainties = criterion.get_uncertainties()
        s_flood = uncertainties["sigma_flood"]
        s_drought = uncertainties["sigma_drought"]

        # Composite multi-hazard validation score (higher is better)
        # Balances PR-AUC improvements against drought RMSE reduction
        composite_score = float(val_prauc - 0.5 * min(val_rmse, 5.0))

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["train_flood_loss"].append(avg_flood_loss)
        history["train_drought_loss"].append(avg_drought_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_flood_prauc"].append(val_prauc)
        history["val_flood_csi"].append(val_csi)
        history["val_drought_rmse"].append(val_rmse)
        history["sigma_flood"].append(s_flood)
        history["sigma_drought"].append(s_drought)
        history["composite_score"].append(composite_score)

        if scheduler is not None:
            scheduler.step()

        epoch_duration = time.time() - epoch_start
        logger.info(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Flood PR-AUC: {val_prauc:.4f} | Drought RMSE: {val_rmse:.4f} | "
            f"Uncertainty: [sigma_f={s_flood:.2f}, sigma_d={s_drought:.2f}] "
            f"({epoch_duration:.1f}s)"
        )

        # ======================================================================
        # 3. EARLY STOPPING & CHECKPOINTING
        # ======================================================================
        if composite_score > best_score + 1e-4:
            best_score = composite_score
            patience_counter = 0

            checkpoint_payload = {
                "epoch": epoch,
                "experiment_name": experiment_name,
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "composite_score": composite_score,
                "flood_metrics": flood_eval,
                "drought_metrics": drought_eval,
                "uncertainties": uncertainties
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
            logger.info(f"  --> Best Checkpoint Saved: '{best_checkpoint_path}' (Composite Score: {best_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early Stopping triggered at Epoch {epoch:02d} (Patience={patience}).")
                break

    total_training_time = time.time() - start_time
    logger.info(f"Training Complete for '{experiment_name}' in {total_training_time:.1f}s.")
    return history, best_checkpoint_path
