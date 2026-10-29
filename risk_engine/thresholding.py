"""
==============================================================================
PHASE 5: Operational Early Warning Engine, Threshold Selection & Hysteresis
==============================================================================
Scientific Directives Enforced:
1. Operational WarningEngine:
   - Evaluates calibrated flood probabilities and continuous drought SPEI forecasts.
   - Strictly processes hazard probability; keeps exposure/vulnerability separate.
2. Scientific Threshold Selection:
   - determine_alert_thresholds() computes alert cutoffs based on validation POD
     and False Alarm Rate (FAR) trade-offs.
3. Hysteresis & Anti-Flapping State Machine:
   - Escalation and de-escalation use asymmetric dead-band (delta) and persistence
     rules to eliminate flapping between alert levels.
   - CRITICAL alerts require verified persistence or extreme confidence.
4. Data Quality & Uncertainty Flags:
   - Stale or high-uncertainty observations explicitly yield 'ADVISORY_UNCERTAIN'
     instead of silently degrading to 'LOW'.
5. Pydantic Structured Output:
   - Yields standardized EarlyWarningRecord with metadata, top drivers, and action guidelines.
==============================================================================
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WarningEngine")


class AlertLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    ADVISORY_UNCERTAIN = "ADVISORY_UNCERTAIN"


class EarlyWarningRecord(BaseModel):
    """
    Standardized operational early warning record emitted per spatial cell.
    Strictly encapsulates hazard probability and diagnostic attribution.
    """
    grid_id: str = Field(..., description="Unique spatial grid / sub-basin identifier")
    hazard_type: str = Field(..., description="Hazard type: flood, drought, or compound")
    issue_timestamp: str = Field(..., description="ISO 8601 timestamp of forecast issue")
    lead_time_hours: int = Field(24, description="Forecast lead time in hours")
    calibrated_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated flood probability")
    spei_forecast: Optional[float] = Field(None, description="Predicted future continuous SPEI index")
    warning_level: AlertLevel = Field(..., description="Operational alert level")
    color_code: str = Field(..., description="Hex color code for map rendering")
    top_explanation_factors: List[Dict[str, Any]] = Field(default_factory=list, description="Top XAI driving factors")
    data_quality_flags: Dict[str, Any] = Field(default_factory=dict, description="Staleness and uncertainty flags")
    recommended_action: str = Field(..., description="Standard operating procedure recommendation")
    hazard_only_metric: bool = Field(True, description="Strictly separates hazard from downstream exposure data")


def determine_alert_thresholds(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    target_fars: Tuple[float, float, float] = (0.25, 0.15, 0.05)
) -> Dict[str, float]:
    """
    Computes empirical alert thresholds (MODERATE, HIGH, CRITICAL)
    based on validation set Probability of Detection (POD) vs False Alarm Rate (FAR) trade-offs.
    
    Args:
        y_val_true: Validation ground truth binary labels {0, 1}
        y_val_prob: Validation predicted probabilities [0, 1]
        target_fars: Tuple of target FAR cutoffs for (MODERATE, HIGH, CRITICAL)
    
    Returns:
        Dict with thresholds: {"LOW": 0.0, "MODERATE": ..., "HIGH": ..., "CRITICAL": ...}
    """
    y_t = np.asarray(y_val_true, dtype=int).flatten()
    y_p = np.asarray(y_val_prob, dtype=float).flatten()

    candidate_thresholds = np.linspace(0.05, 0.95, 91)
    far_targets = target_fars

    selected_thresholds = {
        "LOW": 0.0,
        "MODERATE": 0.20,
        "HIGH": 0.45,
        "CRITICAL": 0.70
    }

    if len(y_t) == 0 or np.sum(y_t) == 0:
        return selected_thresholds

    # Search for thresholds matching target FARs
    for t in candidate_thresholds:
        preds = (y_p >= t).astype(int)
        tp = np.sum((preds == 1) & (y_t == 1))
        fp = np.sum((preds == 1) & (y_t == 0))

        far = fp / (tp + fp) if (tp + fp) > 0 else 0.0

        if far <= far_targets[0] and selected_thresholds["MODERATE"] == 0.20:
            selected_thresholds["MODERATE"] = round(float(t), 3)
        if far <= far_targets[1] and selected_thresholds["HIGH"] == 0.45:
            selected_thresholds["HIGH"] = round(float(t), 3)
        if far <= far_targets[2] and selected_thresholds["CRITICAL"] == 0.70:
            selected_thresholds["CRITICAL"] = round(float(t), 3)

    # Ensure strictly increasing thresholds
    selected_thresholds["MODERATE"] = max(0.10, selected_thresholds["MODERATE"])
    selected_thresholds["HIGH"] = max(selected_thresholds["MODERATE"] + 0.10, selected_thresholds["HIGH"])
    selected_thresholds["CRITICAL"] = max(selected_thresholds["HIGH"] + 0.15, selected_thresholds["CRITICAL"])

    logger.info(f"Empirical alert thresholds derived from validation FAR trade-offs: {selected_thresholds}")
    return selected_thresholds


class WarningEngine:
    """
    Operational Multi-Hazard Early Warning Engine.
    
    Features:
      - POD / FAR trade-off alert thresholding
      - Asymmetric temporal hysteresis state machine (anti-flapping)
      - Critical alert persistence verification
      - Stale and uncertain data detection (ADVISORY_UNCERTAIN)
      - Exposure-separated, standardized EarlyWarningRecord emission
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        hysteresis_delta: float = 0.05,
        persistence_steps: int = 2,
        max_staleness_hours: float = 24.0,
        max_uncertainty_sigma: float = 0.35
    ):
        self.thresholds = thresholds or {
            "LOW": 0.0,
            "MODERATE": 0.20,
            "HIGH": 0.45,
            "CRITICAL": 0.70
        }
        self.delta = hysteresis_delta
        self.persistence_steps = persistence_steps
        self.max_staleness_hours = max_staleness_hours
        self.max_uncertainty_sigma = max_uncertainty_sigma

        # Temporal entity state tracking: {grid_id: {"level": AlertLevel, "cooling": int, "consecutive_critical": int}}
        self.history: Dict[str, Dict[str, Any]] = {}

    def _get_raw_level(self, prob: float) -> AlertLevel:
        """Determines un-stabilized alert level from probability."""
        if prob >= self.thresholds["CRITICAL"]:
            return AlertLevel.CRITICAL
        elif prob >= self.thresholds["HIGH"]:
            return AlertLevel.HIGH
        elif prob >= self.thresholds["MODERATE"]:
            return AlertLevel.MODERATE
        return AlertLevel.LOW

    def _apply_hysteresis(self, grid_id: str, raw_level: AlertLevel, prob: float) -> AlertLevel:
        """
        Applies dead-band hysteresis and persistence rules to prevent alert level flickering.
        """
        level_order = [AlertLevel.LOW, AlertLevel.MODERATE, AlertLevel.HIGH, AlertLevel.CRITICAL]

        if grid_id not in self.history:
            self.history[grid_id] = {
                "level": raw_level,
                "cooling": 0,
                "consecutive_critical": 1 if raw_level == AlertLevel.CRITICAL else 0
            }
            return raw_level

        state = self.history[grid_id]
        curr_level = state["level"]
        curr_idx = level_order.index(curr_level)
        raw_idx = level_order.index(raw_level)

        # 1. Escalation:
        if raw_idx > curr_idx:
            # Special persistence guard for CRITICAL:
            # Immediate escalation to CRITICAL requires either severe confidence (> thresh + 0.10)
            # or persistence over consecutive steps
            if raw_level == AlertLevel.CRITICAL:
                crit_thresh = self.thresholds["CRITICAL"]
                if prob >= (crit_thresh + 0.10) or state["consecutive_critical"] >= (self.persistence_steps - 1):
                    stabilized = AlertLevel.CRITICAL
                    state["consecutive_critical"] = state.get("consecutive_critical", 0) + 1
                else:
                    # Hold at HIGH while building persistence
                    stabilized = AlertLevel.HIGH
                    state["consecutive_critical"] = state.get("consecutive_critical", 0) + 1
                    logger.info(f"[{grid_id}] Critical threshold met; confirming persistence (Step {state['consecutive_critical']}/{self.persistence_steps}).")
            else:
                stabilized = raw_level
                state["consecutive_critical"] = 0

            state["level"] = stabilized
            state["cooling"] = 0
            return stabilized

        # 2. De-escalation:
        elif raw_idx < curr_idx:
            # De-escalation requires probability strictly below threshold - delta
            thresh_name = curr_level.value
            thresh = self.thresholds.get(thresh_name, 0.0)

            if prob < (thresh - self.delta):
                state["cooling"] += 1
                if state["cooling"] >= self.persistence_steps:
                    stabilized = raw_level
                    state["cooling"] = 0
                    state["consecutive_critical"] = 0
                    logger.info(f"[{grid_id}] Verified de-escalation: {curr_level.value} -> {raw_level.value}")
                else:
                    stabilized = curr_level  # Hold previous higher level
            else:
                # Within dead-band margin: hold level
                stabilized = curr_level
                state["cooling"] = 0

            state["level"] = stabilized
            return stabilized

        # 3. Sustained level
        if raw_level == AlertLevel.CRITICAL:
            state["consecutive_critical"] = state.get("consecutive_critical", 0) + 1
        state["cooling"] = 0
        return curr_level

    def process_prediction(
        self,
        grid_id: str,
        calibrated_prob: float,
        spei_forecast: Optional[float] = None,
        top_factors: Optional[List[Dict[str, Any]]] = None,
        issue_timestamp: str = "2026-10-24T21:00:00Z",
        lead_time_hours: int = 24,
        data_staleness_hours: float = 0.0,
        uncertainty_sigma: Optional[float] = None
    ) -> EarlyWarningRecord:
        """
        Processes model inference into an operational early warning record.
        """
        prob = float(calibrated_prob)
        factors = top_factors or []
        unc = float(uncertainty_sigma) if uncertainty_sigma is not None else 0.15

        # 1. Check Data Quality & Uncertainty Flags
        is_stale = bool(data_staleness_hours > self.max_staleness_hours)
        is_uncertain = bool(unc > self.max_uncertainty_sigma)

        data_flags = {
            "is_stale": is_stale,
            "data_staleness_hours": round(data_staleness_hours, 1),
            "uncertainty_sigma": round(unc, 4),
            "high_uncertainty_flag": is_uncertain,
            "sensor_confidence": round(max(0.0, 1.0 - (data_staleness_hours / 72.0)), 2)
        }

        # 2. Check Compound Hazard Conditions
        # Severe antecedent drought (SPEI <= -1.5) baked soil + sudden heavy rain -> flash flood amplifier
        is_compound = False
        hazard_type = "flood"
        if spei_forecast is not None:
            if spei_forecast <= -1.5 and prob >= 0.40:
                is_compound = True
                hazard_type = "compound"

        # 3. Determine Warning Level
        if is_stale or is_uncertain:
            warning_level = AlertLevel.ADVISORY_UNCERTAIN
            color = "#eab308"
            action = (
                "DATA QUALITY ADVISORY: Satellite imagery or rainfall gauge inputs exceed staleness threshold. "
                "Verify ground hydrological stations before deploying assets."
            )
        else:
            raw_level = self._get_raw_level(prob)
            if is_compound:
                # Compound dry-to-deluge condition escalates alert
                raw_level = AlertLevel.CRITICAL

            # Apply temporal hysteresis state stabilization
            warning_level = self._apply_hysteresis(grid_id, raw_level, prob)

            if warning_level == AlertLevel.CRITICAL:
                color = "#ef4444"
                action = "CRITICAL WARNING: Immediate evacuation of low-lying floodplains; dispatch emergency pumps and flood gates."
            elif warning_level == AlertLevel.HIGH:
                color = "#f97316"
                action = "HIGH ALERT: Pre-position search-and-rescue teams; deploy mobile barriers along riverbanks."
            elif warning_level == AlertLevel.MODERATE:
                color = "#eab308"
                action = "MODERATE WATCH: Inspect drainage infrastructure, culverts, and reservoir sluice gates."
            else:
                color = "#22c55e"
                action = "ROUTINE MONITORING: Normal baseline conditions. Continue automated hydrometeorological tracking."

        return EarlyWarningRecord(
            grid_id=grid_id,
            hazard_type=hazard_type,
            issue_timestamp=issue_timestamp,
            lead_time_hours=lead_time_hours,
            calibrated_probability=round(prob, 4),
            spei_forecast=round(spei_forecast, 3) if spei_forecast is not None else None,
            warning_level=warning_level,
            color_code=color,
            top_explanation_factors=factors,
            data_quality_flags=data_flags,
            recommended_action=action,
            hazard_only_metric=True
        )


# Backward compatibility aliases
MultiHazardThresholdEngine = WarningEngine
