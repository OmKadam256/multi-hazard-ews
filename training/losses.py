"""
Multi-Hazard Multi-Task Loss Functions.
Re-exports from training.loss for compatibility.
"""

from training.loss import MultiTaskLoss, BinaryFocalLoss

# Legacy MultiHazardLoss alias
MultiHazardLoss = MultiTaskLoss

__all__ = ["MultiTaskLoss", "BinaryFocalLoss", "MultiHazardLoss"]
