"""
==============================================================================
Temporal Chronological Splitter (Strict No-Leakage Time Horizons)
==============================================================================
Scientific Directive 2:
Implement Temporal Splits (train on early years, test on future years).
Zero future lookahead guarantee.
==============================================================================
"""

from typing import List, Tuple, Dict
from datetime import datetime
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TemporalSplitter")


class TemporalChronologicalSplitter:
    """
    Partitions time series dates or indices chronologically.
    """

    @staticmethod
    def split_by_dates(
        date_strings: List[str],
        train_end_date: str,
        val_end_date: str
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        date_strings: List of 'YYYY-MM-DD' strings.
        train_end_date: Inclusive last date of training period.
        val_end_date: Inclusive last date of validation period.
        All subsequent dates belong to test set.
        """
        train_dt = datetime.fromisoformat(train_end_date)
        val_dt = datetime.fromisoformat(val_end_date)

        train_indices = []
        val_indices = []
        test_indices = []

        for idx, ds in enumerate(date_strings):
            dt = datetime.fromisoformat(ds)
            if dt <= train_dt:
                train_indices.append(idx)
            elif dt <= val_dt:
                val_indices.append(idx)
            else:
                test_indices.append(idx)

        logger.info(
            f"Chronological Split: Train={len(train_indices)} timesteps | "
            f"Val={len(val_indices)} timesteps | Test={len(test_indices)} timesteps"
        )
        return train_indices, val_indices, test_indices

    @staticmethod
    def split_by_ratios(
        total_timesteps: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[List[int], List[int], List[int]]:
        """Splits an integer range chronologically."""
        n_train = int(total_timesteps * train_ratio)
        n_val = int(total_timesteps * val_ratio)

        train_idx = list(range(0, n_train))
        val_idx = list(range(n_train, n_train + n_val))
        test_idx = list(range(n_train + n_val, total_timesteps))

        logger.info(f"Ratio Split: Train=[0:{n_train}] | Val=[{n_train}:{n_train+n_val}] | Test=[{n_train+n_val}:{total_timesteps}]")
        return train_idx, val_idx, test_idx
