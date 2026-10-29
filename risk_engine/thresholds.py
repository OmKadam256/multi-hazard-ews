"""
==============================================================================
Multi-Hazard Risk Thresholds & Compound Threat Scorer
==============================================================================
Categorizes continuous hazard estimates into standard actionable emergency alert levels:
- Flood: LOW, MODERATE, HIGH, CRITICAL
- Drought: NORMAL, MILD, MODERATE, SEVERE, EXTREME
- Compound Risk Index: Evaluates drought-induced soil hydrophobicity compounding flood runoff.
==============================================================================
"""

from enum import Enum
from typing import Dict, Any

from risk_engine.thresholding import (
    WarningEngine,
    EarlyWarningRecord,
    AlertLevel,
    determine_alert_thresholds
)


class FloodAlertLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DroughtAlertLevel(str, Enum):
    NORMAL = "NORMAL"
    MILD = "MILD"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    EXTREME = "EXTREME"


class MultiHazardThresholdEngine:
    """
    Evaluates hazard severity levels and compound hazard amplification.
    """

    @staticmethod
    def classify_flood_risk(calibrated_prob: float) -> Dict[str, Any]:
        """
        Maps calibrated probability to early warning alert level and operational action.
        """
        p = float(calibrated_prob)
        if p >= 0.70:
            level = FloodAlertLevel.CRITICAL
            color = "#ef4444"
            action = "Immediate evacuation of floodplains and low-lying river corridors."
        elif p >= 0.45:
            level = FloodAlertLevel.HIGH
            color = "#f97316"
            action = "Deploy flood barriers; stage emergency pumps and rescue assets."
        elif p >= 0.20:
            level = FloodAlertLevel.MODERATE
            color = "#eab308"
            action = "Issue advisory; inspect drainage culverts and river gauges."
        else:
            level = FloodAlertLevel.LOW
            color = "#22c55e"
            action = "Routine hydrometeorological monitoring."

        return {
            "level": level.value,
            "color": color,
            "calibrated_prob": round(p, 4),
            "recommended_action": action
        }

    @staticmethod
    def classify_drought_risk(spei_continuous: float) -> Dict[str, Any]:
        """
        Maps continuous SPEI to WMO drought categories.
        """
        s = float(spei_continuous)
        if s <= -2.0:
            level = DroughtAlertLevel.EXTREME
            color = "#7f1d1d"
            action = "Enact emergency agricultural water rationing and feed distribution."
        elif s <= -1.5:
            level = DroughtAlertLevel.SEVERE
            color = "#dc2626"
            action = "Restrict non-essential irrigation; monitor reservoir conservation pools."
        elif s <= -1.0:
            level = DroughtAlertLevel.MODERATE
            color = "#ea580c"
            action = "Advise farmers on soil moisture retention and drought-resilient crops."
        elif s <= -0.5:
            level = DroughtAlertLevel.MILD
            color = "#d97706"
            action = "Heightened soil moisture and vegetation vigor tracking."
        else:
            level = DroughtAlertLevel.NORMAL
            color = "#10b981"
            action = "Sufficient water balance; normal agricultural water usage."

        return {
            "level": level.value,
            "color": color,
            "spei_continuous": round(s, 3),
            "recommended_action": action
        }

    @staticmethod
    def compute_compound_hazard_index(
        flood_prob: float,
        spei: float
    ) -> Dict[str, Any]:
        """
        Calculates compound hazard index.
        Compounding scenario: prolonged drought (SPEI <= -1.0) leads to dry, crusted soils.
        When sudden high precipitation occurs, infiltration drops, accelerating flash runoff.
        """
        p_fl = float(flood_prob)
        s_dr = float(spei)

        # Baseline compound score (geometric blend of risks)
        # Invert SPEI deficit into drought risk [0, 1]
        drought_deficit_norm = min(1.0, max(0.0, (-s_dr) / 2.5))
        
        # Compound interaction bonus: when severe drought is followed by high flood risk
        interaction_bonus = 0.0
        if s_dr <= -1.0 and p_fl >= 0.35:
            interaction_bonus = 0.25 * drought_deficit_norm * p_fl

        compound_score = min(1.0, 0.55 * p_fl + 0.45 * drought_deficit_norm + interaction_bonus)

        is_compound_alert = compound_score >= 0.50

        return {
            "compound_hazard_score": round(compound_score, 3),
            "is_compound_threat": is_compound_alert,
            "soil_hydrophobicity_risk": s_dr <= -1.0,
            "summary": "Compound Flash Flood & Hydrophobic Soil Threat" if is_compound_alert else "Standard Multi-Hazard Profile"
        }
