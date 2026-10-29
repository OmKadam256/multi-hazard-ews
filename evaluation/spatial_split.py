"""
==============================================================================
Spatial-Block Cross-Validation & Leave-Sub-Basin-Out Splitter
==============================================================================
Scientific Directive 2:
NEVER randomly split pixels. Implement Spatial Splits (leave-sub-basin-out via HydroSHEDS).
All pixels of a single flood event must remain in the same split.
==============================================================================
"""

import numpy as np
from typing import List, Dict, Tuple, Generator
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SpatialSplitter")


class HydroSHEDSSpatialSplitter:
    """
    Implements Leave-Sub-Basin-Out and Spatial-Block K-Fold Cross Validation.
    """

    def __init__(self, basin_ids: List[int]):
        self.basin_ids = sorted(list(set(basin_ids)))
        logger.info(f"Initialized Spatial Splitter with {len(self.basin_ids)} distinct basins: {self.basin_ids}")

    def leave_sub_basin_out_split(
        self,
        test_basin_ids: List[int],
        val_ratio: float = 0.2,
        seed: int = 42
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits basins into:
        - test_basins: explicitly held out unseen basins
        - val_basins: subset of remaining basins for hyperparameter validation
        - train_basins: remaining training basins
        """
        np.random.seed(seed)
        test_set = set(test_basin_ids)
        remaining = [b for b in self.basin_ids if b not in test_set]

        np.random.shuffle(remaining)
        num_val = max(1, int(len(remaining) * val_ratio))

        val_basins = remaining[:num_val]
        train_basins = remaining[num_val:]

        logger.info(f"Spatial Split: Train Basins={train_basins} | Val Basins={val_basins} | Test Basins={test_basin_ids}")
        return train_basins, val_basins, list(test_set)

    def spatial_k_fold(
        self,
        n_splits: int = 5,
        seed: int = 42
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Yields (train_basin_ids, test_basin_ids) for spatial block cross-validation.
        """
        np.random.seed(seed)
        shuffled = self.basin_ids.copy()
        np.random.shuffle(shuffled)

        folds = np.array_split(shuffled, n_splits)
        for i in range(n_splits):
            test_basins = list(folds[i])
            train_basins = [b for b in self.basin_ids if b not in test_basins]
            yield train_basins, test_basins
