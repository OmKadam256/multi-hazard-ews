"""
==============================================================================
Human-in-the-Loop Operator Approval Queue for CRITICAL Alerts (Phase 3)
==============================================================================
National Disaster Management Authority (NDMA) operational protocols require
that highest-severity (CRITICAL) automated alarms undergo human verification
by an on-duty hydro-informatics operator before public sirens / mass broadcast:
- Non-critical alerts (HIGH, MODERATE, LOW): Auto-dispatch without blocking.
- CRITICAL alerts: Enqueued in OperatorApprovalQueue.
- Actions:
  - approve_alert(): Operator signs off; immediately triggers multi-channel dispatch.
  - reject_alert(): Operator flags false alarm / sensor artifact; cancels broadcast.
  - auto_release(): Configurable safety countdown (default 15 mins). If duty officer
    fails to review in 15 mins during active flood wave, alert is automatically
    released to preserve human life.
==============================================================================
"""

import time
import logging
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OperatorApprovalQueue")


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    AUTO_RELEASED = "AUTO_RELEASED"
    EXPIRED = "EXPIRED"


class OperatorApprovalQueue:
    """
    Manages human review lifecycle for critical disaster warnings.
    """

    def __init__(
        self,
        auto_release_timeout_seconds: float = 900.0, # 15 minutes
        dispatch_callback: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ):
        self.timeout_seconds = auto_release_timeout_seconds
        self.dispatch_callback = dispatch_callback
        self._queue: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.audit_trail: List[Dict[str, Any]] = []

    def enqueue_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enqueues an alert for human review.
        """
        alert_id = str(alert_data.get("id", time.time()))
        now_ts = time.time()
        expires_at = now_ts + self.timeout_seconds

        record = {
            "alert_id": alert_id,
            "status": ApprovalStatus.PENDING,
            "alert_data": alert_data,
            "enqueued_at": datetime.utcnow().isoformat(),
            "enqueued_ts": now_ts,
            "expires_at_ts": expires_at,
            "auto_release_seconds": self.timeout_seconds,
            "reviewed_by": None,
            "reviewed_at": None,
            "decision_notes": None
        }

        with self._lock:
            self._queue[alert_id] = record

        logger.warning(
            f"[APPROVAL QUEUE] CRITICAL Alert {alert_id} held for operator sign-off. "
            f"Auto-release timeout in {self.timeout_seconds}s."
        )
        return record

    def list_pending(self) -> List[Dict[str, Any]]:
        """Returns all currently pending alerts with countdown remaining."""
        now = time.time()
        pending = []
        with self._lock:
            for alert_id, item in self._queue.items():
                if item["status"] == ApprovalStatus.PENDING:
                    rem = max(0.0, item["expires_at_ts"] - now)
                    copy_item = dict(item)
                    copy_item["seconds_remaining"] = round(rem, 1)
                    pending.append(copy_item)
        return pending

    def approve_alert(
        self,
        alert_id: str,
        operator_id: str = "DutyOfficer-1",
        notes: str = "Verified with CWC gauge telemetry"
    ) -> Dict[str, Any]:
        """
        Approves an enqueued alert and triggers immediate downstream dispatch.
        """
        with self._lock:
            item = self._queue.get(alert_id)
            if not item:
                return {"status": "ERROR", "message": f"Alert {alert_id} not in approval queue."}

            if item["status"] != ApprovalStatus.PENDING:
                return {
                    "status": "ERROR",
                    "message": f"Alert {alert_id} is already in state '{item['status']}'."
                }

            item["status"] = ApprovalStatus.APPROVED
            item["reviewed_by"] = operator_id
            item["reviewed_at"] = datetime.utcnow().isoformat()
            item["decision_notes"] = notes

        # Trigger downstream dispatch callback if registered
        dispatch_receipt = None
        if self.dispatch_callback:
            try:
                dispatch_receipt = self.dispatch_callback(item["alert_data"])
            except Exception as e:
                logger.error(f"Error invoking dispatch callback after approval: {e}")
                dispatch_receipt = {"status": "CALLBACK_ERROR", "error": str(e)}

        audit_entry = {
            "event": "ALERT_APPROVED",
            "alert_id": alert_id,
            "operator_id": operator_id,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat(),
            "dispatch_receipt": dispatch_receipt
        }
        self.audit_trail.append(audit_entry)
        logger.info(f"[APPROVAL QUEUE] Alert {alert_id} APPROVED by {operator_id}.")

        return {
            "status": "APPROVED",
            "alert_id": alert_id,
            "reviewed_by": operator_id,
            "dispatch_receipt": dispatch_receipt
        }

    def reject_alert(
        self,
        alert_id: str,
        operator_id: str = "DutyOfficer-1",
        reason: str = "Sensor glitch / False cloudburst alarm"
    ) -> Dict[str, Any]:
        """
        Rejects an alert as false alarm, cancelling mass dissemination.
        """
        with self._lock:
            item = self._queue.get(alert_id)
            if not item:
                return {"status": "ERROR", "message": f"Alert {alert_id} not in approval queue."}

            if item["status"] != ApprovalStatus.PENDING:
                return {
                    "status": "ERROR",
                    "message": f"Alert {alert_id} is already in state '{item['status']}'."
                }

            item["status"] = ApprovalStatus.REJECTED
            item["reviewed_by"] = operator_id
            item["reviewed_at"] = datetime.utcnow().isoformat()
            item["decision_notes"] = f"REJECTED: {reason}"

        audit_entry = {
            "event": "ALERT_REJECTED",
            "alert_id": alert_id,
            "operator_id": operator_id,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.audit_trail.append(audit_entry)
        logger.warning(f"[APPROVAL QUEUE] Alert {alert_id} REJECTED by {operator_id}: {reason}")

        return {
            "status": "REJECTED",
            "alert_id": alert_id,
            "reviewed_by": operator_id,
            "reason": reason
        }

    def process_timeouts(self) -> List[Dict[str, Any]]:
        """
        Checks for timed-out pending alerts and auto-releases them to prevent loss of life.
        """
        now = time.time()
        released = []

        with self._lock:
            for alert_id, item in self._queue.items():
                if item["status"] == ApprovalStatus.PENDING and now >= item["expires_at_ts"]:
                    item["status"] = ApprovalStatus.AUTO_RELEASED
                    item["reviewed_by"] = "SYSTEM_FAILSAFE"
                    item["reviewed_at"] = datetime.utcnow().isoformat()
                    item["decision_notes"] = "Auto-released after operator timeout expiry."
                    released.append(item)

        # Trigger dispatch for auto-released alerts
        for item in released:
            logger.warning(f"[APPROVAL QUEUE] Alert {item['alert_id']} AUTO-RELEASED on timeout!")
            receipt = None
            if self.dispatch_callback:
                try:
                    receipt = self.dispatch_callback(item["alert_data"])
                except Exception as e:
                    logger.error(f"Error in auto-release dispatch: {e}")
                    receipt = {"status": "ERROR", "error": str(e)}

            self.audit_trail.append({
                "event": "ALERT_AUTO_RELEASED",
                "alert_id": item["alert_id"],
                "timestamp": datetime.utcnow().isoformat(),
                "dispatch_receipt": receipt
            })

        return released

