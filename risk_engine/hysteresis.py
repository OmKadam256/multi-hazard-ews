"""
==============================================================================
Temporal Hysteresis State Machine (Alert Flapping Prevention)
==============================================================================
Operational Problem:
When flood probability hovers around an alert boundary (e.g. 0.45), raw alerts 
flicker between Moderate and High across consecutive hours, confusing emergency managers.

Solution:
Asymmetric state transition with dead-band hysteresis:
- Escalation: Instantaneous transition when p >= T.
- De-escalation: Requires p < T - delta for k consecutive timesteps.
==============================================================================
"""

from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HysteresisEngine")


class HysteresisStateMachine:
    """
    Maintains alert stability across time sequences for each geographic entity (sub-basin).
    """

    def __init__(
        self,
        hysteresis_delta: float = 0.05,
        required_cooling_steps: int = 2
    ):
        self.delta = hysteresis_delta
        self.required_cooling_steps = required_cooling_steps

        # State storage: {entity_id: {"current_level": str, "cooling_counter": int}}
        self.states: Dict[str, Dict[str, Any]] = {}

        # Alert boundaries
        self.thresholds = {
            "CRITICAL": 0.70,
            "HIGH": 0.45,
            "MODERATE": 0.20,
            "LOW": 0.0
        }

    def update_entity_state(
        self,
        entity_id: str,
        prob: float
    ) -> Dict[str, Any]:
        """
        Updates and stabilizes the alert level for entity_id at the current timestep.
        """
        p = float(prob)
        if entity_id not in self.states:
            # Initial state
            initial_level = self._get_level_for_prob(p)
            self.states[entity_id] = {
                "current_level": initial_level,
                "cooling_counter": 0,
                "history": [p]
            }
            return {
                "entity_id": entity_id,
                "stabilized_level": initial_level,
                "raw_level": initial_level,
                "is_stable": True
            }

        curr_state = self.states[entity_id]
        curr_level = curr_state["current_level"]
        raw_level = self._get_level_for_prob(p)
        cooling = curr_state["cooling_counter"]

        level_order = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        curr_idx = level_order.index(curr_level)
        raw_idx = level_order.index(raw_level)

        if raw_idx > curr_idx:
            # Immediate escalation
            stabilized_level = raw_level
            new_cooling = 0
            logger.info(f"[{entity_id}] Alert Escalation: {curr_level} -> {stabilized_level} (p={p:.3f})")
        elif raw_idx < curr_idx:
            # De-escalation candidate: check dead-band margin
            # Must be strictly below threshold - delta
            thresh = self.thresholds[curr_level]
            if p < (thresh - self.delta):
                new_cooling = cooling + 1
                if new_cooling >= self.required_cooling_steps:
                    stabilized_level = raw_level
                    new_cooling = 0
                    logger.info(f"[{entity_id}] Alert De-escalation verified: {curr_level} -> {stabilized_level} (p={p:.3f})")
                else:
                    stabilized_level = curr_level  # Hold prior level until cooling period verified
            else:
                # Within dead-band margin: reset cooling and hold
                stabilized_level = curr_level
                new_cooling = 0
        else:
            stabilized_level = curr_level
            new_cooling = 0

        curr_state["current_level"] = stabilized_level
        curr_state["cooling_counter"] = new_cooling
        curr_state["history"].append(p)

        return {
            "entity_id": entity_id,
            "stabilized_level": stabilized_level,
            "raw_level": raw_level,
            "cooling_counter": new_cooling,
            "hysteresis_active": (stabilized_level != raw_level)
        }

    def _get_level_for_prob(self, p: float) -> str:
        if p >= 0.70:
            return "CRITICAL"
        elif p >= 0.45:
            return "HIGH"
        elif p >= 0.20:
            return "MODERATE"
        return "LOW"
