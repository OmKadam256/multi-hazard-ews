"""
==============================================================================
GEE Download & Ingestion Pipeline with Strict Zero Data Leakage Enforcement
==============================================================================
Scientific Rule 1: Sentinel-1 SAR and CHIRPS rainfall must ONLY use 
acquisitions available strictly before the forecast issue time t. 
Post-event flood imagery is strictly forbidden for forecasting.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GEEIngestionEngine")


class ZeroLeakageValidator:
    """Enforces temporal causal boundaries for all dynamic satellite layers."""

    @staticmethod
    def validate_acquisitions(
        issue_time: datetime,
        acquisition_timestamps: List[datetime],
        dataset_name: str
    ) -> List[datetime]:
        """
        Filters and validates that no satellite or reanalysis observation is on or after issue_time.
        Raises ValueError if post-event imagery is detected in a forecast context.
        """
        valid = []
        for ts in acquisition_timestamps:
            if ts >= issue_time:
                msg = (
                    f"CRITICAL LEAKAGE DETECTED in {dataset_name}: Observation timestamp {ts} "
                    f"is >= forecast issue time {issue_time}. Post-event data is strictly forbidden."
                )
                logger.error(msg)
                raise ValueError(msg)
            valid.append(ts)
        return valid

    @staticmethod
    def validate_observation_timestamp(
        obs_time: datetime,
        issue_time: datetime
    ) -> Tuple[bool, str]:
        """
        Validates that an observation timestamp strictly precedes the forecast issue time.
        """
        if obs_time >= issue_time:
            return False, f"Observation {obs_time.isoformat()} is on or after issue time {issue_time.isoformat()}"
        return True, "Observation precedes issue time"


class GEEIngestionEngine:
    """
    Defines Earth Engine collection queries with strict temporal bounds.
    Can export JavaScript/Python EE task specifications or fetch rasters.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.raw_dir = config.get("data", {}).get("raw_dir", "data/raw")
        os.makedirs(self.raw_dir, exist_ok=True)

    def generate_ee_export_manifest(
        self,
        forecast_issue_time: str,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generates an Earth Engine extraction manifest ensuring strict no-leakage constraints.
        """
        issue_dt = datetime.fromisoformat(forecast_issue_time)
        start_dt = issue_dt - timedelta(days=lookback_days)
        # Strict boundary: stop 1 second before midnight of issue_time if daily, or exact issue_dt
        end_str = (issue_dt - timedelta(seconds=1)).isoformat()
        start_str = start_dt.isoformat()

        logger.info(f"Building GEE Manifest: Issue Time={forecast_issue_time}")
        logger.info(f"Causal Window: [{start_str} to {end_str}] (Zero Leakage Guaranteed)")

        manifest = {
            "forecast_issue_time": forecast_issue_time,
            "temporal_window": {
                "start": start_str,
                "end": end_str,
                "lookback_days": lookback_days
            },
            "spatial_bounds": self.config.get("geospatial", {}).get("bounding_box"),
            "target_crs": self.config.get("geospatial", {}).get("target_crs", "EPSG:32644"),
            "target_resolution_m": self.config.get("geospatial", {}).get("analysis_resolution_m", 1000),
            "collections": {
                "sentinel1_sar": {
                    "collection": "COPERNICUS/S1_GRD",
                    "filter": {
                        "instrumentMode": "IW",
                        "polarization": ["VV", "VH"],
                        "orbitProperties_pass": "DESCENDING"
                    },
                    "temporal_filter": [start_str, end_str],
                    "notes": "Acquisitions strictly prior to t. No post-flood extent used as feature."
                },
                "chirps_precipitation": {
                    "collection": "UCSB-CHG/CHIRPS/DAILY",
                    "band": "precipitation",
                    "temporal_filter": [start_str, end_str]
                },
                "era5_land": {
                    "collection": "ECMWF/ERA5_LAND/HOURLY",
                    "bands": [
                        "temperature_2m",
                        "volumetric_soil_water_layer_1",
                        "volumetric_soil_water_layer_2",
                        "total_evaporation"
                    ],
                    "temporal_filter": [start_str, end_str]
                },
                "static_dem": {
                    "collection": "COPERNICUS/DEM/GLO30_2024_1",
                    "band": "DEM"
                },
                "soilgrids": {
                    "clay": "projects/soilgrids-isric/clay_mean",
                    "sand": "projects/soilgrids-isric/sand_mean",
                    "bdod": "projects/soilgrids-isric/bdod_mean"
                },
                "hydrosheds": {
                    "basins": "WWF/HydroSHEDS/v1/Basins/hybas_7",
                    "flow_accumulation": "WWF/HydroSHEDS/03ACC"
                }
            }
        }
        manifest_path = os.path.join(self.raw_dir, f"gee_manifest_{forecast_issue_time[:10]}.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest written to {manifest_path}")
        return manifest


class MockGeospatialDataGenerator:
    """
    Generates physically consistent geospatial synthetic data for unit testing,
    offline development, and CI/CD without requiring live Earth Engine credentials.
    Produces 1 km grid arrays across space (H, W) and time (T).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.H = 192  # 192 km grid
        self.W = 192
        self.patch_size = config.get("geospatial", {}).get("patch_size", 32)
        self.processed_dir = config.get("data", {}).get("processed_dir", "data/processed")
        self.labels_dir = config.get("data", {}).get("labels_dir", "data/labels")
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def generate_synthetic_basin_suite(self, seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Creates synthetic physical terrain:
        - DEM (valleys, slopes, ridges)
        - Flow accumulation
        - Soil properties
        - HydroSHEDS sub-basin masks (10 distinct sub-basins)
        """
        np.random.seed(seed)
        logger.info(f"Generating synthetic physical geospatial domain ({self.H}x{self.W} km)...")

        # 1. Synthetic Elevation Surface (DEM) with main river valley running diagonally
        y, x = np.mgrid[0:self.H, 0:self.W]
        # Valley lowest along a curved line: x - 0.5*y = 30
        dist_to_river = np.abs(x - (0.5 * y + 30 + 15 * np.sin(y / 20.0)))
        elevation = 200.0 + 3.0 * dist_to_river + 50.0 * np.sin(x / 30.0) * np.cos(y / 40.0)
        elevation += np.random.normal(0, 2.0, size=(self.H, self.W))
        elevation = np.clip(elevation, 10.0, 1500.0).astype(np.float32)

        # 2. Flow Accumulation (highest along river channel)
        flow_acc = np.exp(-dist_to_river / 6.0) * 10000.0 + np.random.uniform(1.0, 20.0, (self.H, self.W))
        flow_acc = flow_acc.astype(np.float32)

        # 3. Slope (first derivative of elevation)
        dy, dx_val = np.gradient(elevation)
        slope_rad = np.arctan(np.sqrt(dx_val**2 + dy**2) / 1000.0)
        slope_deg = np.rad2deg(slope_rad).astype(np.float32)

        # 4. Topographic Wetness Index (TWI) = ln(a / tan(beta))
        # Add small epsilon to tan(slope) to avoid div zero
        tan_slope = np.maximum(np.tan(slope_rad), 0.001)
        twi = np.log((flow_acc * 1000.0) / tan_slope).astype(np.float32)
        twi = np.clip(twi, 2.0, 25.0)

        # 5. Soil Properties
        soil_clay = (30.0 + 15.0 * np.cos(x / 45.0) + np.random.normal(0, 3, (self.H, self.W))).astype(np.float32)
        soil_clay = np.clip(soil_clay, 5.0, 65.0)
        soil_sand = (100.0 - soil_clay - 25.0).astype(np.float32)
        soil_bulk_density = (1.35 + 0.15 * np.sin(y / 50.0)).astype(np.float32)

        # 6. HydroSHEDS Sub-basin partitioning (Divide into 10 distinct sub-basins)
        # Using Voronoi-like centroid regions
        basin_centroids = [
            (30, 30, 406001), (30, 100, 406002), (30, 160, 406003),
            (95, 40, 406004), (95, 100, 406005), (95, 160, 406006),
            (160, 35, 406007), (160, 95, 406008), (160, 155, 406009),
            (130, 130, 406010)
        ]
        sub_basins = np.zeros((self.H, self.W), dtype=np.int32)
        for i in range(self.H):
            for j in range(self.W):
                dists = [( (i - cy)**2 + (j - cx)**2, bid ) for cy, cx, bid in basin_centroids]
                dists.sort(key=lambda t: t[0])
                sub_basins[i, j] = dists[0][1]

        data_dict = {
            "dem": elevation,
            "slope": slope_deg,
            "twi": twi,
            "flow_acc": flow_acc,
            "dist_to_river": dist_to_river.astype(np.float32),
            "soil_clay": soil_clay,
            "soil_sand": soil_sand,
            "soil_bulk_density": soil_bulk_density,
            "sub_basins": sub_basins
        }

        # Save static arrays
        np.savez_compressed(os.path.join(self.processed_dir, "static_geospatial_layers.npz"), **data_dict)
        logger.info("Saved static_geospatial_layers.npz")
        return data_dict

    def generate_synthetic_timeseries_dataset(
        self,
        num_timesteps: int = 120,
        lookback_window: int = 30
    ) -> str:
        """
        Generates dynamic temporal observations strictly respecting temporal causality.
        For each timestep t, dynamic variables represent antecedent observations [t - lookback, t).
        Targets:
          - Flood Occurrence at t (binary mask, 0 or 1, high class imbalance ~3% positive)
          - Drought SPEI continuous at t (Z-score from -3.0 to +3.0) and discrete class (0 to 4)
        """
        static = self.generate_basin_suite_if_needed()
        dem = static["dem"]
        twi = static["twi"]
        dist_to_river = static["dist_to_river"]
        sub_basins = static["sub_basins"]

        logger.info(f"Generating {num_timesteps} timesteps of dynamic spatiotemporal grids...")
        
        # Save dates list
        start_date = datetime(2018, 1, 1)
        dates = [(start_date + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(num_timesteps)]

        # Arrays container
        # Dynamic channels: [chirps_rain, s1_sar_vv, s1_sar_vh, era5_soil_moist, era5_temp, era5_pet]
        # Shape per timestep: (lookback_window=30, num_channels=6, H, W)
        # To make it lightweight and fast for dual T4 GPU training, we can store time-slices
        daily_rain = np.zeros((num_timesteps, self.H, self.W), dtype=np.float32)
        daily_sar_vv = np.zeros((num_timesteps, self.H, self.W), dtype=np.float32)
        daily_sar_vh = np.zeros((num_timesteps, self.H, self.W), dtype=np.float32)
        daily_soil_moist = np.zeros((num_timesteps, self.H, self.W), dtype=np.float32)
        daily_temp = np.zeros((num_timesteps, self.H, self.W), dtype=np.float32)
        daily_pet = np.zeros((num_timesteps, self.H, self.W), dtype=np.float32)

        flood_labels = np.zeros((num_timesteps, self.H, self.W), dtype=np.uint8)
        drought_spei_cont = np.zeros((num_timesteps, self.H, self.W), dtype=np.float32)
        drought_class = np.zeros((num_timesteps, self.H, self.W), dtype=np.int64)

        # Baseline climatic seasonality
        for t_idx in range(num_timesteps):
            season_sin = np.sin(2 * np.pi * t_idx / 365.0)
            
            # Monsoon pulse / rainfall storms
            is_monsoon = (t_idx % 60) > 40
            storm_intensity = 35.0 if is_monsoon and (t_idx % 12 in [0, 1, 2]) else 2.0
            
            # Spatially coherent rainfall field
            rain_field = np.maximum(0.0, storm_intensity + 8.0 * np.sin(np.linspace(0, 3, self.H))[:, None] + np.random.exponential(4.0, (self.H, self.W)))
            daily_rain[t_idx] = rain_field.astype(np.float32)

            # Temperature and PET
            temp_field = 25.0 + 8.0 * season_sin - (dem - 200.0) * 0.0065 + np.random.normal(0, 1.0, (self.H, self.W))
            daily_temp[t_idx] = temp_field.astype(np.float32)
            pet_field = np.maximum(0.5, 0.18 * (temp_field + 5.0) + np.random.normal(0, 0.2, (self.H, self.W)))
            daily_pet[t_idx] = pet_field.astype(np.float32)

            # Soil moisture (running decay of rainfall + soil properties)
            if t_idx == 0:
                sm = 0.22 + 0.002 * twi
            else:
                sm = 0.88 * daily_soil_moist[t_idx - 1] + 0.003 * daily_rain[t_idx] - 0.01 * daily_pet[t_idx]
            daily_soil_moist[t_idx] = np.clip(sm, 0.05, 0.55).astype(np.float32)

            # Sentinel-1 SAR Backscatter (VV/VH): wet soil/inundation drops backscatter
            daily_sar_vv[t_idx] = -12.0 - 15.0 * (daily_soil_moist[t_idx] > 0.45).astype(np.float32) + np.random.normal(0, 1.0, (self.H, self.W))
            daily_sar_vh[t_idx] = daily_sar_vv[t_idx] - 6.5 + np.random.normal(0, 0.8, (self.H, self.W))

            # Ground Truth Inundation Label (Physics-based threshold: extreme antecedent rain + high TWI + river proximity)
            if t_idx >= 7:
                rain_7d = np.sum(daily_rain[t_idx-7:t_idx], axis=0)
            else:
                rain_7d = daily_rain[t_idx] * 7.0
            
            flood_susceptibility = (rain_7d > 65.0) & (twi > 10.5) & (dist_to_river < 25.0) & (daily_soil_moist[t_idx] > 0.32)
            flood_labels[t_idx] = flood_susceptibility.astype(np.uint8)

            # Ground Truth Drought SPEI Label
            # Water balance D = Rain - PET over 30 days
            if t_idx >= lookback_window:
                p_sum = np.sum(daily_rain[t_idx-lookback_window:t_idx], axis=0)
                pet_sum = np.sum(daily_pet[t_idx-lookback_window:t_idx], axis=0)
            else:
                p_sum = daily_rain[t_idx] * lookback_window
                pet_sum = daily_pet[t_idx] * lookback_window
            
            water_balance = p_sum - pet_sum
            # Standardize to SPEI Z-score roughly mean 0 std 1
            spei_val = (water_balance - 40.0) / 45.0
            spei_val = np.clip(spei_val, -3.0, 3.0)
            drought_spei_cont[t_idx] = spei_val.astype(np.float32)

            # Discrete classification
            # 0: Normal (SPEI >= -0.5)
            # 1: Mild (-1.0 < SPEI <= -0.5)
            # 2: Moderate (-1.5 < SPEI <= -1.0)
            # 3: Severe (-2.0 < SPEI <= -1.5)
            # 4: Extreme (SPEI <= -2.0)
            d_class = np.zeros_like(spei_val, dtype=np.int64)
            d_class[spei_val <= -0.5] = 1
            d_class[spei_val <= -1.0] = 2
            d_class[spei_val <= -1.5] = 3
            d_class[spei_val <= -2.0] = 4
            drought_class[t_idx] = d_class

        dataset_path = os.path.join(self.processed_dir, "spatiotemporal_cube.npz")
        np.savez_compressed(
            dataset_path,
            dates=np.array(dates),
            daily_rain=daily_rain,
            daily_sar_vv=daily_sar_vv,
            daily_sar_vh=daily_sar_vh,
            daily_soil_moist=daily_soil_moist,
            daily_temp=daily_temp,
            daily_pet=daily_pet,
            flood_labels=flood_labels,
            drought_spei_cont=drought_spei_cont,
            drought_class=drought_class,
            sub_basins=sub_basins
        )
        logger.info(f"Saved complete spatio-temporal dataset cube to {dataset_path}")
        return dataset_path

    def generate_basin_suite_if_needed(self) -> Dict[str, np.ndarray]:
        static_path = os.path.join(self.processed_dir, "static_geospatial_layers.npz")
        if os.path.exists(static_path):
            with np.load(static_path) as data:
                return {k: data[k] for k in data.files}
        return self.generate_synthetic_basin_suite()
