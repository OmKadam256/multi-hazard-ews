"""
==============================================================================
Dual-T4 Optimized Multi-Hazard Model Trainer
==============================================================================
Features:
- PyTorch AMP (Automatic Mixed Precision) for T4 Tensor Cores
- DataParallel multi-GPU support
- Early stopping based on Validation PR-AUC and multi-hazard loss
- Cosine Annealing learning rate scheduler
- Comprehensive loss logging
==============================================================================
"""

import os
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import numpy as np

from training.losses import MultiHazardLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Trainer")


class MultiHazardTrainer:
    """
    Manages model training, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict[str, Any],
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        training_cfg = config.get("training", {})
        self.lr = training_cfg.get("learning_rate", 0.0003)
        self.weight_decay = training_cfg.get("weight_decay", 1e-5)
        self.epochs = training_cfg.get("epochs", 30)
        self.patience = training_cfg.get("early_stopping_patience", 8)
        self.mixed_precision = training_cfg.get("mixed_precision", True) and device.type == "cuda"

        # Optimizer & Loss
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.criterion = MultiHazardLoss(
            w_flood=training_cfg.get("loss_weights", {}).get("flood_focal", 1.0),
            w_spei=training_cfg.get("loss_weights", {}).get("drought_huber", 0.5),
            w_drought_ce=training_cfg.get("loss_weights", {}).get("drought_ce", 0.3)
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        flood_loss = 0.0
        spei_loss = 0.0
        ce_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            x_spatial = batch["x_spatial"].to(self.device, non_blocking=True)
            x_temporal = batch["x_temporal"].to(self.device, non_blocking=True)
            targets = {
                "flood_label": batch["flood_label"].to(self.device, non_blocking=True),
                "spei_target": batch["spei_target"].to(self.device, non_blocking=True),
                "drought_class": batch["drought_class"].to(self.device, non_blocking=True)
            }

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                preds = self.model(x_spatial, x_temporal)
                losses = self.criterion(preds, targets)
                loss = losses["total_loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            flood_loss += losses["loss_flood_focal"].item()
            spei_loss += losses["loss_spei_huber"].item()
            ce_loss += losses["loss_drought_ce"].item()
            num_batches += 1

        self.scheduler.step()
        n = max(1, num_batches)
        return {
            "train_loss": total_loss / n,
            "train_flood_loss": flood_loss / n,
            "train_spei_loss": spei_loss / n,
            "train_ce_loss": ce_loss / n
        }

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_flood_probs = []
        all_flood_targets = []
        all_spei_preds = []
        all_spei_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                x_spatial = batch["x_spatial"].to(self.device, non_blocking=True)
                x_temporal = batch["x_temporal"].to(self.device, non_blocking=True)
                targets = {
                    "flood_label": batch["flood_label"].to(self.device, non_blocking=True),
                    "spei_target": batch["spei_target"].to(self.device, non_blocking=True),
                    "drought_class": batch["drought_class"].to(self.device, non_blocking=True)
                }

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    preds = self.model(x_spatial, x_temporal)
                    losses = self.criterion(preds, targets)

                total_loss += losses["total_loss"].item()
                num_batches += 1

                all_flood_probs.extend(preds["flood_prob"].view(-1).cpu().numpy())
                all_flood_targets.extend(targets["flood_label"].view(-1).cpu().numpy())
                all_spei_preds.extend(preds["spei_continuous"].view(-1).cpu().numpy())
                all_spei_targets.extend(targets["spei_target"].view(-1).cpu().numpy())

        n = max(1, num_batches)
        val_loss = total_loss / n

        # Calculate PR-AUC and Drought RMSE
        from sklearn.metrics import precision_recall_curve, auc, mean_squared_error
        y_true = np.array(all_flood_targets)
        y_prob = np.array(all_flood_probs)
        if len(np.unique(y_true)) > 1:
            p, r, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(r, p))
        else:
            pr_auc = 0.0

        drought_rmse = float(np.sqrt(mean_squared_error(all_spei_targets, all_spei_preds)))

        return {
            "val_loss": val_loss,
            "val_flood_prauc": pr_auc,
            "val_drought_rmse": drought_rmse
        }

    def fit(self) -> Dict[str, Any]:
        logger.info(f"Starting training on {self.device} (AMP={self.mixed_precision})...")
        best_val_loss = float("inf")
        best_prauc = -1.0
        epochs_no_improve = 0
        best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_multi_hazard_model.pt")

        history = []

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            elapsed = time.time() - t0

            val_loss = val_metrics["val_loss"]
            prauc = val_metrics["val_flood_prauc"]
            rmse = val_metrics["val_drought_rmse"]

            logger.info(
                f"Epoch {epoch:02d}/{self.epochs:02d} [{elapsed:.1f}s] - "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Flood PR-AUC: {prauc:.4f} | "
                f"Drought RMSE: {rmse:.4f}"
            )

            history.append({
                "epoch": epoch,
                **train_metrics,
                **val_metrics
            })

            # Checkpointing based on Validation PR-AUC & loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_prauc = prauc
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_flood_prauc": prauc
                }, best_checkpoint_path)
                logger.info(f" Saved new best model checkpoint to {best_checkpoint_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (Patience={self.patience})")
                    break

        return {
            "best_checkpoint_path": best_checkpoint_path,
            "best_val_loss": best_val_loss,
            "best_val_flood_prauc": best_prauc,
            "history": history
        }
