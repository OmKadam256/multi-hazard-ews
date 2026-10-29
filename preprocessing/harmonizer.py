"""
==============================================================================
Geospatial Layer Harmonization Engine
==============================================================================
Scientific Directive 3:
All static (DEM, SoilGrids) and dynamic (ERA5-Land, CHIRPS, Sentinel-1, NDVI) 
layers must be reprojected to a common UTM CRS and resampled to a 1 km analysis grid 
(downscale fine data; bilinear upsample coarse data).
==============================================================================
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.ndimage import zoom

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GeospatialHarmonizer")


class GeospatialHarmonizer:
    """
    Harmonizes heterogeneous multi-sensor geospatial rasters to a standardized
    1 km analysis grid in target UTM CRS.
    """

    def __init__(self, target_crs: str = "EPSG:32644", target_resolution_m: float = 1000.0):
        self.target_crs = target_crs
        self.target_resolution_m = target_resolution_m
        logger.info(f"Initialized Harmonizer: Target CRS={target_crs}, Resolution={target_resolution_m}m")

    def resample_raster(
        self,
        array: np.ndarray,
        source_resolution_m: float,
        target_resolution_m: Optional[float] = None,
        method: str = "bilinear",
        preserve_hydrology: bool = False
    ) -> np.ndarray:
        """
        Resamples a 2D or 3D raster array from source_resolution_m to target_resolution_m.
        - If source_res < target_res: downscaling fine data (e.g. 30m DEM -> 1000m).
        - If source_res > target_res: bilinear upsampling coarse data (e.g. 10km ERA5 -> 1000m).
        - If preserve_hydrology is True (for flow accumulation): uses local maximum/sum.
        """
        if target_resolution_m is None:
            target_resolution_m = self.target_resolution_m

        scale_factor = source_resolution_m / target_resolution_m
        if np.isclose(scale_factor, 1.0, atol=1e-3):
            return array

        logger.info(
            f"Resampling layer: Source Res={source_resolution_m}m -> Target Res={target_resolution_m}m "
            f"(Scale Factor={scale_factor:.4f}, Method={method})"
        )

        order = 1 if method == "bilinear" else 0  # 1 for bilinear, 0 for nearest

        if array.ndim == 2:
            if preserve_hydrology and scale_factor < 1.0:
                # Downscaling flow accumulation: take maximum to preserve stream channel connectivity
                out_shape = (int(array.shape[0] * scale_factor), int(array.shape[1] * scale_factor))
                # Block max pooling approximation
                block_h = max(1, int(round(1.0 / scale_factor)))
                block_w = max(1, int(round(1.0 / scale_factor)))
                resampled = self._block_reduce(array, (block_h, block_w), np.nanmax)
            else:
                resampled = zoom(array, scale_factor, order=order, mode="nearest")
        elif array.ndim == 3:
            # Multi-channel or time series: (T, H, W) or (C, H, W)
            zoom_factors = (1.0, scale_factor, scale_factor)
            resampled = zoom(array, zoom_factors, order=order, mode="nearest")
        else:
            raise ValueError(f"Unsupported array dimension: {array.ndim}")

        return resampled.astype(np.float32)

    def harmonize_cube(
        self,
        layers: Dict[str, Tuple[np.ndarray, float]],
        target_shape: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """
        Harmonizes a dictionary of {layer_name: (array, native_resolution_m)} into
        identical (target_shape[0], target_shape[1]) matrices.
        """
        harmonized = {}
        target_h, target_w = target_shape

        for name, (arr, native_res) in layers.items():
            preserve_hydro = "flow_acc" in name.lower() or "stream" in name.lower()
            method = "nearest" if "basin" in name.lower() or "class" in name.lower() else "bilinear"
            resampled = self.resample_raster(
                arr,
                source_resolution_m=native_res,
                target_resolution_m=self.target_resolution_m,
                method=method,
                preserve_hydrology=preserve_hydro
            )

            # Strict shape alignment
            curr_h, curr_w = resampled.shape[-2], resampled.shape[-1]
            if curr_h != target_h or curr_w != target_w:
                # Crop or pad to match exact target grid
                pad_h = max(0, target_h - curr_h)
                pad_w = max(0, target_w - curr_w)
                if pad_h > 0 or pad_w > 0:
                    pad_width = [(0, pad_h), (0, pad_w)] if resampled.ndim == 2 else [(0, 0), (0, pad_h), (0, pad_w)]
                    resampled = np.pad(resampled, pad_width, mode="edge")
                resampled = resampled[..., :target_h, :target_w]

            harmonized[name] = resampled
            logger.info(f"Harmonized layer '{name}': Shape={resampled.shape}, Dtype={resampled.dtype}")

        return harmonized

    @staticmethod
    def _block_reduce(array: np.ndarray, block_size: Tuple[int, int], func) -> np.ndarray:
        bh, bw = block_size
        h, w = array.shape
        new_h, new_w = h // bh, w // bw
        trimmed = array[:new_h * bh, :new_w * bw]
        reshaped = trimmed.reshape(new_h, bh, new_w, bw)
        return func(func(reshaped, axis=3), axis=1)
