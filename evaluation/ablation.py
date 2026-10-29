"""
==============================================================================
Master Ablation & Model Comparison Engine
==============================================================================
Strict Baseline-to-Advanced Ablation Strategy:
1. Logistic Regression Baseline
2. Random Forest Baseline
3. XGBoost Baseline (scale_pos_weight)
4. Spatial CNN Only (no temporal history)
5. Temporal GRU Only (no spatial terrain)
6. MultiTask Deep Model with Plain Concatenation Fusion
7. MultiTask Deep Model with Gated Cross-Modal Attention Fusion (Proposed Master)

All models evaluated strictly on unseen future time horizon AND held-out sub-basins.
Metrics reported: PR-AUC, POD@FAR=0.10, POD@FAR=0.20, CSI, Brier Score, Drought RMSE,
and Skill Score vs. Persistence.
==============================================================================
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

from evaluation.metrics import ScientificMetricsCalculator
from models.baselines.logistic import BaselineLinearSystem
from models.baselines.random_forest import BaselineRandomForestSystem
from models.baselines.xgboost_model import BaselineXGBoostSystem
from models.multitask import MultiHazardDeepModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AblationEngine")


class MultiHazardAblationEngine:
    """
    Orchestrates training and comparative evaluation across all 7 ablation variants.
    """

    def __init__(self, config: Dict[str, Any], output_dir: str = "evaluation/results"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_tabular_arrays(
        self,
        cube_path: str,
        static_path: str,
        sample_indices: List[Tuple[int, int, int]],
        lookback_window: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts patch locations into tabular feature rows for baseline models.
        Returns:
            X: (N, num_features)
            y_flood: (N,)
            y_spei: (N,)
            y_persist: (N,) persistence baseline SPEI
        """
        with np.load(static_path) as s:
            dem = s["dem"]
            slope = s["slope"]
            twi = s["twi"]
            flow_acc = s["flow_acc"]
            dist_to_river = s["dist_to_river"]
            soil_clay = s["soil_clay"]
            soil_sand = s["soil_sand"]

        with np.load(cube_path) as c:
            daily_rain = c["daily_rain"]
            daily_soil_moist = c["daily_soil_moist"]
            daily_sar_vv = c["daily_sar_vv"]
            flood_labels = c["flood_labels"]
            spei_cont = c["drought_spei_cont"]

        rows = []
        y_flood_list = []
        y_spei_list = []
        y_persist_list = []

        patch_size = self.config.get("geospatial", {}).get("patch_size", 32)

        for (t, y, x) in sample_indices:
            # Mean static values in patch
            m_dem = float(np.mean(dem[y:y+patch_size, x:x+patch_size]))
            m_slope = float(np.mean(slope[y:y+patch_size, x:x+patch_size]))
            m_twi = float(np.mean(twi[y:y+patch_size, x:x+patch_size]))
            m_acc = float(np.mean(np.log1p(flow_acc[y:y+patch_size, x:x+patch_size])))
            m_dist = float(np.mean(dist_to_river[y:y+patch_size, x:x+patch_size]))
            m_clay = float(np.mean(soil_clay[y:y+patch_size, x:x+patch_size]))
            m_sand = float(np.mean(soil_sand[y:y+patch_size, x:x+patch_size]))

            # Dynamic antecedent summaries
            rain_patch_seq = np.mean(daily_rain[t-lookback_window:t, y:y+patch_size, x:x+patch_size], axis=(1, 2))
            sm_patch_seq = np.mean(daily_soil_moist[t-lookback_window:t, y:y+patch_size, x:x+patch_size], axis=(1, 2))
            sar_patch_seq = np.mean(daily_sar_vv[t-lookback_window:t, y:y+patch_size, x:x+patch_size], axis=(1, 2))

            r_3d = float(np.sum(rain_patch_seq[-3:]))
            r_7d = float(np.sum(rain_patch_seq[-7:]))
            r_15d = float(np.sum(rain_patch_seq[-15:]))
            r_30d = float(np.sum(rain_patch_seq))
            r_max = float(np.max(rain_patch_seq))
            sm_curr = float(sm_patch_seq[-1])
            sar_curr = float(sar_patch_seq[-1])

            row = [
                m_dem, m_slope, m_twi, m_acc, m_dist, m_clay, m_sand,
                r_3d, r_7d, r_15d, r_30d, r_max, sm_curr, sar_curr
            ]
            rows.append(row)

            # Targets
            f_val = float(np.any(flood_labels[t, y:y+patch_size, x:x+patch_size] > 0))
            s_val = float(np.mean(spei_cont[t, y:y+patch_size, x:x+patch_size]))
            # Persistence SPEI is value at start of forecast window (t - 1)
            p_val = float(np.mean(spei_cont[t - 1, y:y+patch_size, x:x+patch_size])) if t > 0 else s_val

            y_flood_list.append(f_val)
            y_spei_list.append(s_val)
            y_persist_list.append(p_val)

        return (
            np.array(rows, dtype=np.float32),
            np.array(y_flood_list, dtype=np.float32),
            np.array(y_spei_list, dtype=np.float32),
            np.array(y_persist_list, dtype=np.float32)
        )

    def run_full_ablation_benchmark(
        self,
        X_train: np.ndarray,
        y_f_train: np.ndarray,
        y_s_train: np.ndarray,
        X_test: np.ndarray,
        y_f_test: np.ndarray,
        y_s_test: np.ndarray,
        y_p_test: np.ndarray,
        deep_model_gated: MultiHazardDeepModel,
        deep_model_concat: MultiHazardDeepModel,
        test_loader: Any,
        device: torch.device
    ) -> Dict[str, Dict[str, float]]:
        """
        Runs the full 7-model ablation battery and computes evaluation scores.
        """
        results = {}

        # 1. Baseline 1: Logistic Regression & Ridge
        logger.info("Evaluating Baseline 1: Logistic Regression...")
        lr_sys = BaselineLinearSystem()
        lr_sys.fit(X_train, y_f_train, y_s_train)
        p_flood_lr = lr_sys.predict_flood_prob(X_test)
        p_spei_lr = lr_sys.predict_spei(X_test)
        results["1_Logistic_Regression"] = {
            **ScientificMetricsCalculator.compute_flood_metrics(y_f_test, p_flood_lr),
            **ScientificMetricsCalculator.compute_drought_metrics(y_s_test, p_spei_lr, y_p_test)
        }

        # 2. Baseline 2: Random Forest
        logger.info("Evaluating Baseline 2: Random Forest...")
        rf_sys = BaselineRandomForestSystem(n_estimators=100)
        rf_sys.fit(X_train, y_f_train, y_s_train)
        p_flood_rf = rf_sys.predict_flood_prob(X_test)
        p_spei_rf = rf_sys.predict_spei(X_test)
        results["2_Random_Forest"] = {
            **ScientificMetricsCalculator.compute_flood_metrics(y_f_test, p_flood_rf),
            **ScientificMetricsCalculator.compute_drought_metrics(y_s_test, p_spei_rf, y_p_test)
        }

        # 3. Baseline 3: XGBoost
        logger.info("Evaluating Baseline 3: XGBoost...")
        xgb_sys = BaselineXGBoostSystem(n_estimators=150, scale_pos_weight=10.0)
        xgb_sys.fit(X_train, y_f_train, y_s_train)
        p_flood_xgb = xgb_sys.predict_flood_prob(X_test)
        p_spei_xgb = xgb_sys.predict_spei(X_test)
        results["3_XGBoost"] = {
            **ScientificMetricsCalculator.compute_flood_metrics(y_f_test, p_flood_xgb),
            **ScientificMetricsCalculator.compute_drought_metrics(y_s_test, p_spei_xgb, y_p_test)
        }

        # 4. Deep Models Inference
        deep_model_gated.eval()
        deep_model_concat.eval()

        all_f_true = []
        all_s_true = []
        gated_f_prob = []
        gated_s_pred = []
        concat_f_prob = []
        concat_s_pred = []
        spatial_only_f_prob = []
        spatial_only_s_pred = []
        temporal_only_f_prob = []
        temporal_only_s_pred = []

        with torch.no_grad():
            for batch in test_loader:
                x_sp = batch["x_spatial"].to(device)
                x_tp = batch["x_temporal"].to(device)
                f_true = batch["flood_label"].numpy()
                s_true = batch["spei_target"].numpy()

                all_f_true.extend(f_true)
                all_s_true.extend(s_true)

                # Proposed Gated Cross-Modal
                out_g = deep_model_gated(x_sp, x_tp)
                gated_f_prob.extend(out_g["flood_prob"].view(-1).cpu().numpy())
                gated_s_pred.extend(out_g["spei_continuous"].view(-1).cpu().numpy())

                # Concat Baseline
                out_c = deep_model_concat(x_sp, x_tp)
                concat_f_prob.extend(out_c["flood_prob"].view(-1).cpu().numpy())
                concat_s_pred.extend(out_c["spei_continuous"].view(-1).cpu().numpy())

                # Spatial Only Ablation (zero out temporal inputs)
                out_sp_only = deep_model_gated(x_sp, torch.zeros_like(x_tp))
                spatial_only_f_prob.extend(out_sp_only["flood_prob"].view(-1).cpu().numpy())
                spatial_only_s_pred.extend(out_sp_only["spei_continuous"].view(-1).cpu().numpy())

                # Temporal Only Ablation (zero out spatial inputs)
                out_tp_only = deep_model_gated(torch.zeros_like(x_sp), x_tp)
                temporal_only_f_prob.extend(out_tp_only["flood_prob"].view(-1).cpu().numpy())
                temporal_only_s_pred.extend(out_tp_only["spei_continuous"].view(-1).cpu().numpy())

        # Length alignment
        min_len = min(len(all_f_true), len(y_p_test))
        y_f_eval = np.array(all_f_true[:min_len])
        y_s_eval = np.array(all_s_true[:min_len])
        y_p_eval = np.array(y_p_test[:min_len])

        # 4. Spatial CNN Only
        results["4_Spatial_CNN_Only"] = {
            **ScientificMetricsCalculator.compute_flood_metrics(y_f_eval, np.array(spatial_only_f_prob[:min_len])),
            **ScientificMetricsCalculator.compute_drought_metrics(y_s_eval, np.array(spatial_only_s_pred[:min_len]), y_p_eval)
        }

        # 5. Temporal GRU Only
        results["5_Temporal_GRU_Only"] = {
            **ScientificMetricsCalculator.compute_flood_metrics(y_f_eval, np.array(temporal_only_f_prob[:min_len])),
            **ScientificMetricsCalculator.compute_drought_metrics(y_s_eval, np.array(temporal_only_s_pred[:min_len]), y_p_eval)
        }

        # 6. Deep Concat Fusion
        results["6_MultiTask_Concatenation"] = {
            **ScientificMetricsCalculator.compute_flood_metrics(y_f_eval, np.array(concat_f_prob[:min_len])),
            **ScientificMetricsCalculator.compute_drought_metrics(y_s_eval, np.array(concat_s_pred[:min_len]), y_p_eval)
        }

        # 7. Proposed Master Model (Gated Cross-Modal Attention)
        results["7_MultiTask_GatedAttention_PROPOSED"] = {
            **ScientificMetricsCalculator.compute_flood_metrics(y_f_eval, np.array(gated_f_prob[:min_len])),
            **ScientificMetricsCalculator.compute_drought_metrics(y_s_eval, np.array(gated_s_pred[:min_len]), y_p_eval)
        }

        # Save scorecard
        res_path = os.path.join(self.output_dir, "ablation_scorecard.json")
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)

        self._generate_markdown_report(results)
        return results

    def _generate_markdown_report(self, results: Dict[str, Dict[str, float]]):
        md_path = os.path.join(self.output_dir, "ablation_summary.md")
        lines = [
            "# Model Ablation & Evaluation Scorecard",
            "Strict Spatial Split (Leave-Sub-Basin-Out) & Chronological Test Horizon",
            "",
            "| Model Architecture | Flood PR-AUC | POD @ FAR=0.10 | CSI | Brier Score | Drought RMSE | Skill Score vs Persist |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]
        for name, m in results.items():
            lines.append(
                f"| **{name}** | {m.get('pr_auc', 0):.4f} | {m.get('pod_at_far_10', 0):.4f} | "
                f"{m.get('csi', 0):.4f} | {m.get('brier_score', 0):.4f} | {m.get('drought_rmse', 0):.4f} | "
                f"{m.get('skill_score_vs_persistence', 0):.4f} |"
            )

        with open(md_path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Ablation summary report written to {md_path}")
