"""
==============================================================================
AI Multi-Hazard Early Warning & GeoIntelligence Platform (India)
Emergency Alert Dispatch & Multi-Channel Notification Package (Phase 3)
==============================================================================
Provides decoupled, multi-channel disaster alert dissemination:
- Telegram Bot Notifications (Free Telegram Bot API)
- SMTP Emergency Email Broadcast (HTML Dossier + Links)
- TRAI DLT-Compliant SMS Interface (Zero-cost stub + extensible gateway)
- OASIS CAP v1.2 (Common Alerting Protocol) XML Feed for NDMA / CWC
- Human-in-the-Loop Operator Approval Queue for CRITICAL alerts
- Hysteresis-aware Dispatch Engine with Rate-Limiting & Deduplication
==============================================================================
"""

from api.alerts.telegram import TelegramAlertNotifier
from api.alerts.email import EmailAlertNotifier
from api.alerts.sms import SMSAlertNotifier, BaseSMSProvider, MockSMSProvider
from api.alerts.cap_feed import CAPFeedGenerator
from api.alerts.approval import OperatorApprovalQueue, ApprovalStatus
from api.alerts.dispatcher import AlertDispatcher, alert_dispatcher

__all__ = [
    "TelegramAlertNotifier",
    "EmailAlertNotifier",
    "SMSAlertNotifier",
    "BaseSMSProvider",
    "MockSMSProvider",
    "CAPFeedGenerator",
    "OperatorApprovalQueue",
    "ApprovalStatus",
    "AlertDispatcher",
    "alert_dispatcher",
]
