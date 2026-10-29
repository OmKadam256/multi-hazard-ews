"""
==============================================================================
Central Multi-Channel Disaster Alert Dispatcher (Phase 3)
==============================================================================
Orchestrates emergency alert dissemination across all national channels:
- Channel integration: Telegram Bot, SMTP Email, TRAI DLT SMS, OASIS CAP v1.2 Feed.
- Human-in-the-Loop review: Holds CRITICAL alerts in OperatorApprovalQueue.
- Hysteresis transition filtering: Suppresses duplicate dispatch when an alert
  state remains STABLE; triggers immediate broadcast upon NEW or ESCALATED events.
- Rate-limiting & deduplication: Eliminates alert fatigue across major basins.
==============================================================================
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from api.alerts.telegram import TelegramAlertNotifier
from api.alerts.email import EmailAlertNotifier
from api.alerts.sms import SMSAlertNotifier
from api.alerts.cap_feed import CAPFeedGenerator
from api.alerts.approval import OperatorApprovalQueue, ApprovalStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AlertDispatcher")

# Numeric hierarchy for evaluating escalation vs de-escalation
TIER_HIERARCHY = {
    "LOW": 1,
    "MODERATE": 2,
    "HIGH": 3,
    "CRITICAL": 4
}


class AlertDispatcher:
    """
    Central dispatch router with hysteresis state filtering and human-in-the-loop gating.
    """

    def __init__(
        self,
        telegram_notifier: Optional[TelegramAlertNotifier] = None,
        email_notifier: Optional[EmailAlertNotifier] = None,
        sms_notifier: Optional[SMSAlertNotifier] = None,
        cap_generator: Optional[CAPFeedGenerator] = None,
        approval_queue: Optional[OperatorApprovalQueue] = None,
        rate_limit_hours: float = 6.0
    ):
        self.telegram = telegram_notifier or TelegramAlertNotifier()
        self.email = email_notifier or EmailAlertNotifier()
        self.sms = sms_notifier or SMSAlertNotifier()
        self.cap = cap_generator or CAPFeedGenerator()
        self.rate_limit_seconds = rate_limit_hours * 3600.0

        # Operator approval queue for CRITICAL alerts
        self.approval_queue = approval_queue or OperatorApprovalQueue(
            auto_release_timeout_seconds=900.0,
            dispatch_callback=self._execute_immediate_dispatch
        )

        # Catchment alert state tracking for hysteresis & deduplication
        # key: (region_id, hazard_type) -> {"tier": str, "last_dispatched_ts": float}
        self._catchment_states: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.active_cap_alerts: List[Dict[str, Any]] = []
        self.dispatch_log: List[Dict[str, Any]] = []

    def evaluate_transition(
        self,
        region_id: str,
        hazard_type: str,
        current_tier: str
    ) -> Tuple[str, bool]:
        """
        Evaluates hysteresis transition relative to prior known state.
        Returns (transition_type, should_dispatch).
        transition_type in: NEW, ESCALATED, STABLE, DE_ESCALATED
        """
        state_key = (str(region_id), str(hazard_type).lower())
        prev_state = self._catchment_states.get(state_key)
        now_ts = time.time()

        if not prev_state:
            # First time this catchment triggered an alert
            return "NEW", True

        prev_tier = prev_state["tier"]
        last_dispatched = prev_state.get("last_dispatched_ts", 0.0)
        time_elapsed = now_ts - last_dispatched

        curr_rank = TIER_HIERARCHY.get(current_tier.upper(), 1)
        prev_rank = TIER_HIERARCHY.get(prev_tier.upper(), 1)

        if curr_rank > prev_rank:
            # Escalation always bypasses rate-limiting to save lives
            return "ESCALATED", True

        if curr_rank < prev_rank:
            # De-escalated; update state but suppress emergency siren
            return "DE_ESCALATED", False

        # Alert tier is identical (STABLE)
        if time_elapsed >= self.rate_limit_seconds:
            # 6-hour refresh interval elapsed; allow periodic reminder
            return "STABLE_REFRESH", True

        # Within rate limit window and unchanged tier -> suppress to prevent alert fatigue
        return "STABLE_SUPPRESSED", False

    def dispatch_alert(
        self,
        alert_data: Dict[str, Any],
        bypass_approval: bool = False
    ) -> Dict[str, Any]:
        """
        Main entry point for dispatching a single early warning alert.
        Handles approval queuing for CRITICAL alerts, hysteresis state checks,
        and multi-channel distribution.
        """
        tier = alert_data.get("tier", "HIGH").upper()
        region_id = str(alert_data.get("region_id", "Unknown"))
        hazard_type = alert_data.get("hazard_type", "flood").lower()
        alert_id = alert_data.get("id", f"alt_{int(time.time())}")

        # 1. Hysteresis Transition Evaluation
        transition, should_dispatch = self.evaluate_transition(region_id, hazard_type, tier)
        alert_data["transition"] = transition

        # Always register in public CAP XML repository
        self._record_cap_alert(alert_data)

        if not should_dispatch and not bypass_approval:
            logger.info(
                f"[DISPATCH SUPPRESSED] Catchment {region_id} ({hazard_type}) tier '{tier}' "
                f"transition={transition}. Suppressing duplicate notification."
            )
            return {
                "alert_id": alert_id,
                "status": "SUPPRESSED_FATIGUE_PREVENTION",
                "transition": transition,
                "tier": tier,
                "reason": "Alert tier unchanged within 6-hour rate limit window."
            }

        # 2. Human-in-the-Loop Gate for CRITICAL alerts
        if tier == "CRITICAL" and not bypass_approval:
            logger.warning(f"[DISPATCH] CRITICAL alert {alert_id} redirected to Operator Approval Queue.")
            queue_record = self.approval_queue.enqueue_alert(alert_data)
            return {
                "alert_id": alert_id,
                "status": "HELD_FOR_OPERATOR_APPROVAL",
                "tier": tier,
                "transition": transition,
                "queue_record": queue_record
            }

        # 3. Direct Multi-Channel Dissemination (HIGH, MODERATE or Approved CRITICAL)
        return self._execute_immediate_dispatch(alert_data)

    def _execute_immediate_dispatch(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal execution of channel delivery after approval or auto-bypass.
        """
        tier = alert_data.get("tier", "HIGH").upper()
        region_id = str(alert_data.get("region_id", "Unknown"))
        hazard_type = alert_data.get("hazard_type", "flood").lower()
        alert_id = alert_data.get("id", f"alt_{int(time.time())}")

        channel_results = {}
        now_ts = time.time()

        # Multi-channel routing based on severity tier
        if tier == "CRITICAL":
            # CRITICAL: Telegram + Email + SMS + CAP
            channel_results["telegram"] = self.telegram.send_alert(alert_data)
            channel_results["email"] = self.email.send_alert(alert_data)
            channel_results["sms"] = self.sms.send_alert(alert_data)
        elif tier == "HIGH":
            # HIGH: Telegram + Email + CAP
            channel_results["telegram"] = self.telegram.send_alert(alert_data)
            channel_results["email"] = self.email.send_alert(alert_data)
        elif tier == "MODERATE":
            # MODERATE: Email + CAP
            channel_results["email"] = self.email.send_alert(alert_data)
        else: # LOW
            # LOW: CAP feed only
            channel_results["cap_feed"] = {"status": "INDEXED_IN_CAP_FEED"}

        # Update catchment state memory
        self._catchment_states[(region_id, hazard_type)] = {
            "tier": tier,
            "last_dispatched_ts": now_ts,
            "alert_id": alert_id
        }

        dispatch_record = {
            "alert_id": alert_id,
            "region_id": region_id,
            "hazard_type": hazard_type,
            "tier": tier,
            "status": "DISPATCHED",
            "channels_contacted": list(channel_results.keys()),
            "channel_results": channel_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.dispatch_log.append(dispatch_record)
        logger.info(f"[DISPATCH COMPLETE] Alert {alert_id} ({tier}) disseminated to {list(channel_results.keys())}")
        return dispatch_record

    def _record_cap_alert(self, alert_data: Dict[str, Any]):
        """Maintains active CAP alerts collection."""
        alert_id = alert_data.get("id")
        # Remove old record for same alert_id if present
        self.active_cap_alerts = [a for a in self.active_cap_alerts if a.get("id") != alert_id]
        self.active_cap_alerts.append(alert_data)
        if len(self.active_cap_alerts) > 100:
            self.active_cap_alerts.pop(0)

    def get_cap_feed_xml(self) -> str:
        """Returns the full OASIS CAP v1.2 XML feed string."""
        return self.cap.generate_cap_feed(self.active_cap_alerts)

    def dispatch_batch(self, alerts_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Dispatches a collection of alerts (e.g. from the 6-hour scheduler pipeline).
        """
        results = []
        for alert in alerts_list:
            res = self.dispatch_alert(alert)
            results.append(res)

        return {
            "total_processed": len(alerts_list),
            "dispatched_count": sum(1 for r in results if r.get("status") == "DISPATCHED"),
            "held_for_approval_count": sum(1 for r in results if r.get("status") == "HELD_FOR_OPERATOR_APPROVAL"),
            "suppressed_count": sum(1 for r in results if r.get("status") == "SUPPRESSED_FATIGUE_PREVENTION"),
            "results": results
        }


# Global shared dispatcher instance for FastAPI and Scheduler
alert_dispatcher = AlertDispatcher()
