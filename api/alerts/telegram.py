"""
==============================================================================
Telegram Bot Alert Dissemination Channel (100% Free Telegram Bot API)
==============================================================================
Provides automated emergency broadcast to national/regional Telegram channels:
- Zero paid API keys: uses the official free Telegram Bot API.
- Configurable via TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.
- Safe zero-cost fallback: if credentials are omitted or network fails, logs
  full formatted broadcast to audit log without halting execution.
- Includes mandatory CWC/NDMA emergency advisory disclaimer.
==============================================================================
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TelegramAlertNotifier")

NDMA_DISCLAIMER = (
    "DISCLAIMER: AI-generated advisory for operational decision support under CWC / NDMA guidelines. "
    "Verify with local disaster management authorities (SEOC/DEOC) prior to public evacuation."
)


class TelegramAlertNotifier:
    """
    Dispatches early warning alerts via Telegram Bot API with graceful mock fallback.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout_seconds: float = 5.0
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.timeout_seconds = timeout_seconds
        self.is_configured = bool(self.bot_token and self.chat_id)
        self.dispatch_history: List[Dict[str, Any]] = []

    def format_alert_message(self, alert_data: Dict[str, Any]) -> str:
        """
        Formats alert payload into Telegram HTML/Markdown notification.
        """
        tier = alert_data.get("tier", "HIGH").upper()
        hazard_type = alert_data.get("hazard_type", "flood").upper()
        region_id = alert_data.get("region_id", "Unknown")
        region_name = alert_data.get("region_name", f"Catchment {region_id}")
        lead_time = alert_data.get("lead_time_h", 48)
        prob = alert_data.get("probability_calibrated", alert_data.get("calibrated_prob", 0.0))
        spei = alert_data.get("spei_forecast", alert_data.get("forecast_index"))
        action = alert_data.get("recommended_action", "Monitor local hydro-meteorological advisories.")
        issued_at = alert_data.get("issued_at", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

        # Icon mapping
        icon_map = {
            "CRITICAL": "🚨🔴 [CRITICAL EMERGENCY ALERT]",
            "HIGH": "⚠️🟠 [HIGH THREAT WARNING]",
            "MODERATE": "⚡🟡 [MODERATE WATCH ADVISORY]",
            "LOW": "ℹ️🟢 [LOW RISK MONITORING]"
        }
        header = icon_map.get(tier, "⚠️ [HAZARD ADVISORY]")

        lines = [
            f"<b>{header}</b>",
            f"<b>Hazard:</b> {hazard_type} FORECAST",
            f"<b>Basin / Catchment:</b> {region_name} (ID: {region_id})",
            f"<b>Alert Tier:</b> {tier}",
            f"<b>Forecast Horizon:</b> +{lead_time} Hours",
        ]

        if hazard_type == "FLOOD" or prob is not None:
            lines.append(f"<b>Calibrated Flood Probability:</b> {prob * 100:.1f}%")
        if spei is not None:
            lines.append(f"<b>Drought SPEI Deficit:</b> {float(spei):.2f} Z-Score")

        lines.extend([
            f"<b>Operational Action:</b> {action}",
            f"<b>Issue Time:</b> {issued_at}",
            "",
            f"<i>{NDMA_DISCLAIMER}</i>"
        ])

        return "\n".join(lines)

    def send_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends alert via Telegram Bot API or logs in mock mode if unconfigured.
        """
        formatted_text = self.format_alert_message(alert_data)
        alert_id = alert_data.get("id", str(time.time()))
        receipt: Dict[str, Any] = {
            "channel": "telegram",
            "alert_id": alert_id,
            "region_id": alert_data.get("region_id"),
            "tier": alert_data.get("tier"),
            "timestamp": datetime.utcnow().isoformat(),
            "formatted_message": formatted_text,
        }

        if not self.is_configured:
            receipt["status"] = "MOCK_LOGGED"
            receipt["notes"] = "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID unset; recorded in mock log."
            logger.info(f"[Telegram MOCK DISPATCH] Alert {alert_id} logged (Tokens not configured).")
            self.dispatch_history.append(receipt)
            return receipt

        # Real Telegram Bot API call
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": formatted_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
            if resp.status_code == 200:
                receipt["status"] = "DELIVERED"
                receipt["message_id"] = resp.json().get("result", {}).get("message_id")
                logger.info(f"[Telegram DELIVERED] Alert {alert_id} successfully sent to chat {self.chat_id}")
            else:
                receipt["status"] = "FAILED"
                receipt["error"] = f"Telegram API error {resp.status_code}: {resp.text}"
                logger.warning(f"[Telegram ERROR] Failed to send alert {alert_id}: {resp.text}")
        except Exception as e:
            receipt["status"] = "FAILED"
            receipt["error"] = str(e)
            logger.warning(f"[Telegram EXCEPTION] {e}; falling back to mock audit record.")

        self.dispatch_history.append(receipt)
        return receipt

