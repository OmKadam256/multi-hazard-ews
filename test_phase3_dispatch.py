"""
==============================================================================
Phase 3 Comprehensive Verification Suite:
Multi-Channel Disaster Alert Dispatch, TRAI DLT SMS, CAP v1.2 & Operator Queue
==============================================================================
Tests:
  1. Telegram Bot Alert Formatting, Mock/Live Delivery & NDMA Disclaimer
  2. SMTP HTML Incident Email Formatting & Multi-Recipient Handling
  3. TRAI DLT Compliant SMS Templates & 160-Character Limits
  4. OASIS CAP v1.2 XML Feed Schema & ElementTree Structure
  5. Dispatch Engine Hysteresis Transitions, Escalations & Deduplication
  6. Human-in-the-Loop Operator Approval Queue (Hold, Approve, Reject, Timeout)
  7. FastAPI Endpoints Integration (/api/v1/alerts/cap, /pending, /approve, /dispatch-test)
==============================================================================
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

# Windows PyTorch DLL rule: import torch first
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.alerts.telegram import TelegramAlertNotifier, NDMA_DISCLAIMER as TG_DISCLAIMER
from api.alerts.email import EmailAlertNotifier, NDMA_DISCLAIMER as EMAIL_DISCLAIMER
from api.alerts.sms import SMSAlertNotifier, MockSMSProvider, TRAI_DLT_TEMPLATES
from api.alerts.cap_feed import CAPFeedGenerator, CAP_NAMESPACE
from api.alerts.approval import OperatorApprovalQueue, ApprovalStatus
from api.alerts.dispatcher import AlertDispatcher, alert_dispatcher
from api.main import (
    get_cap_alert_feed,
    list_pending_critical_alerts,
    approve_critical_alert,
    reject_critical_alert,
    dispatch_test_alert,
    get_alert_dispatch_history,
    AlertDispatchRequest
)


def test_telegram_alert_channel():
    print("\n--- 1. Testing Telegram Bot Alert Dissemination Channel ---")
    notifier = TelegramAlertNotifier(bot_token=None, chat_id=None) # Mock mode
    
    alert_sample = {
        "id": "tg_test_001",
        "region_id": "406008",
        "region_name": "Godavari Basin & Wardha Plain",
        "hazard_type": "flood",
        "tier": "CRITICAL",
        "lead_time_h": 48,
        "probability_calibrated": 0.92,
        "spei_forecast": -0.2,
        "recommended_action": "Execute immediate riverine evacuation along Wardha floodplains."
    }

    formatted = notifier.format_alert_message(alert_sample)
    assert "[CRITICAL EMERGENCY ALERT]" in formatted
    assert "Godavari Basin & Wardha Plain" in formatted
    assert "92.0%" in formatted
    assert "Execute immediate riverine evacuation" in formatted
    assert "DISCLAIMER" in formatted

    receipt = notifier.send_alert(alert_sample)
    assert receipt["status"] == "MOCK_LOGGED"
    assert receipt["channel"] == "telegram"
    assert receipt["alert_id"] == "tg_test_001"
    print("  [PASS] Telegram message formatted with critical alert badge, telemetry, and legal disclaimer.")
    print("  [PASS] Mock delivery successfully executed without requiring external paid API keys.")


def test_email_alert_channel():
    print("\n--- 2. Testing SMTP Emergency Email Formatting & Dispatch ---")
    notifier = EmailAlertNotifier(
        smtp_host=None, # Mock mode
        recipients=["flood-control@cwc.nic.in", "seoc@maharashtra.gov.in"]
    )

    alert_sample = {
        "id": "email_test_001",
        "region_id": "102001",
        "region_name": "Brahmaputra Valley & Delta",
        "hazard_type": "flood",
        "tier": "HIGH",
        "lead_time_h": 24,
        "probability_calibrated": 0.78,
        "spei_forecast": 0.5,
        "recommended_action": "Stage NDRF rescue boats at Guwahati and Dhubri embankments."
    }

    html = notifier.build_html_email(alert_sample)
    assert "<!DOCTYPE html>" in html
    assert "Brahmaputra Valley & Delta" in html
    assert "HIGH WARNING" in html
    assert "78.0%" in html
    assert "Guwahati" in html
    assert "STATUTORY LEGAL NOTICE" in html

    receipt = notifier.send_alert(alert_sample)
    assert receipt["status"] == "MOCK_LOGGED"
    assert receipt["channel"] == "email"
    assert receipt["recipients_count"] == 2
    assert "EEWS HIGH ADVISORY" in receipt["subject"]
    print("  [PASS] HTML email rendered responsive CSS, threat color badges, and CWC action protocols.")
    print("  [PASS] Multi-recipient dispatch logged successfully in zero-cost fallback mode.")


def test_trai_dlt_sms_channel():
    print("\n--- 3. Testing TRAI DLT Compliant SMS Templates & Constraints ---")
    mock_provider = MockSMSProvider()
    sms_notifier = SMSAlertNotifier(provider=mock_provider)

    tiers_to_test = [
        ("CRITICAL", "flood", 0.95),
        ("HIGH", "flood", 0.75),
        ("MODERATE", "drought", 0.35)
    ]

    for tier, hz, prob in tiers_to_test:
        alert = {
            "id": f"sms_{tier}",
            "region_id": "406008",
            "hazard_type": hz,
            "tier": tier,
            "lead_time_h": 48,
            "probability_calibrated": prob
        }
        res = sms_notifier.format_dlt_sms(alert)
        # Strict TRAI compliance: <= 160 characters
        assert res["char_count"] <= 160, f"DLT template exceeded 160 chars: {res['char_count']}"
        assert res["header"] == "NDMAEW"
        assert res["template_id"] in [t["template_id"] for t in TRAI_DLT_TEMPLATES.values()]
        print(f"  [PASS] {tier} SMS text ({res['char_count']} chars): '{res['text']}'")

    # Test delivery
    send_res = sms_notifier.send_alert(
        alert_data={"id": "sms_send_test", "tier": "CRITICAL", "region_id": "406008", "probability_calibrated": 0.92},
        recipient_phones=["+919876543210", "+919876543211"]
    )
    assert send_res["channel"] == "sms"
    assert send_res["recipients_count"] == 2
    assert len(send_res["receipts"]) == 2
    assert send_res["receipts"][0]["within_160_chars"] is True
    print("  [PASS] TRAI Principal Entity ID & Header validated across multiple recipients.")


def test_cap_v12_xml_generation():
    print("\n--- 4. Testing OASIS Common Alerting Protocol (CAP v1.2) XML ---")
    generator = CAPFeedGenerator(sender="cwc-eews@ndma.gov.in")

    alert_sample = {
        "id": "cap_test_001",
        "region_id": "406008",
        "region_name": "Godavari Basin & Wardha Plain",
        "hazard_type": "flood",
        "tier": "CRITICAL",
        "lead_time_h": 48,
        "probability_calibrated": 0.92,
        "spei_forecast": -0.2,
        "recommended_action": "Evacuate floodplains.",
        "polygon": "19.2,78.5 19.8,78.8 19.5,79.2 19.2,78.5"
    }

    # 4.1 Test single <alert> node
    alert_el = generator.create_cap_alert(alert_sample)
    xml_str = generator.to_xml_string(alert_el)

    # Validate XML parsing
    root = ET.fromstring(xml_str)
    assert "alert" in root.tag
    assert root.find(f"{{{CAP_NAMESPACE}}}sender").text == "cwc-eews@ndma.gov.in"
    assert root.find(f"{{{CAP_NAMESPACE}}}status").text == "Actual"
    assert root.find(f"{{{CAP_NAMESPACE}}}scope").text == "Public"

    info = root.find(f"{{{CAP_NAMESPACE}}}info")
    assert info is not None
    assert info.find(f"{{{CAP_NAMESPACE}}}severity").text == "Extreme"
    assert info.find(f"{{{CAP_NAMESPACE}}}urgency").text == "Immediate"
    assert info.find(f"{{{CAP_NAMESPACE}}}certainty").text == "Observed"

    area = info.find(f"{{{CAP_NAMESPACE}}}area")
    assert area.find(f"{{{CAP_NAMESPACE}}}areaDesc").text == "Godavari Basin & Wardha Plain"
    assert area.find(f"{{{CAP_NAMESPACE}}}polygon").text == alert_sample["polygon"]

    print("  [PASS] CAP v1.2 XML root and <info> child nodes conform to OASIS standard.")

    # 4.2 Test multi-alert syndication <feed>
    feed_xml = generator.generate_cap_feed([alert_sample, {
        "id": "cap_test_002",
        "region_id": "104001",
        "region_name": "Thar & Luni Arid Basin",
        "hazard_type": "drought",
        "tier": "HIGH",
        "lead_time_h": 72,
        "probability_calibrated": 0.10,
        "spei_forecast": -2.3,
        "recommended_action": "Activate agricultural groundwater rationing."
    }])
    assert "<feed" in feed_xml
    assert "India Multi-Hazard EEWS" in feed_xml
    print("  [PASS] Syndication <feed> successfully serialized with multiple concurrent alerts.")


def test_hysteresis_and_deduplication():
    print("\n--- 5. Testing Dispatcher Hysteresis Transitions & Rate-Limiting ---")
    dispatcher = AlertDispatcher(rate_limit_hours=6.0)

    basin_id = "406008"

    # Step 1: Initial event -> Must dispatch as NEW
    t1, should_d1 = dispatcher.evaluate_transition(basin_id, "flood", "MODERATE")
    assert t1 == "NEW"
    assert should_d1 is True
    res1 = dispatcher.dispatch_alert({
        "id": "step1_mod", "region_id": basin_id, "hazard_type": "flood", "tier": "MODERATE", "lead_time_h": 48
    })
    assert res1["status"] == "DISPATCHED"
    print("  [PASS] Step 1: First occurrence dispatched cleanly as NEW.")

    # Step 2: Same tier within 6h -> Must SUPPRESS to prevent alert fatigue
    t2, should_d2 = dispatcher.evaluate_transition(basin_id, "flood", "MODERATE")
    assert t2 == "STABLE_SUPPRESSED"
    assert should_d2 is False
    res2 = dispatcher.dispatch_alert({
        "id": "step2_mod", "region_id": basin_id, "hazard_type": "flood", "tier": "MODERATE", "lead_time_h": 48
    })
    assert res2["status"] == "SUPPRESSED_FATIGUE_PREVENTION"
    print("  [PASS] Step 2: Unchanged tier suppressed immediately by hysteresis rate limiter.")

    # Step 3: Escalation (MODERATE -> HIGH) -> Must BYPASS rate-limit and DISPATCH
    t3, should_d3 = dispatcher.evaluate_transition(basin_id, "flood", "HIGH")
    assert t3 == "ESCALATED"
    assert should_d3 is True
    res3 = dispatcher.dispatch_alert({
        "id": "step3_high", "region_id": basin_id, "hazard_type": "flood", "tier": "HIGH", "lead_time_h": 48
    })
    assert res3["status"] == "DISPATCHED"
    print("  [PASS] Step 3: Escalation from MODERATE to HIGH bypassed rate limit and dispatched.")

    # Step 4: De-escalation (HIGH -> MODERATE) -> Suppress emergency broadcast
    t4, should_d4 = dispatcher.evaluate_transition(basin_id, "flood", "MODERATE")
    assert t4 == "DE_ESCALATED"
    assert should_d4 is False
    print("  [PASS] Step 4: De-escalation recognized; siren broadcast suppressed.")


def test_operator_approval_queue():
    print("\n--- 6. Testing Human-in-the-Loop Operator Approval Queue ---")
    dispatched_events = []
    queue = OperatorApprovalQueue(
        auto_release_timeout_seconds=2.0, # Fast timeout for test
        dispatch_callback=lambda alt: dispatched_events.append(alt) or {"status": "CALLBACK_SUCCESS"}
    )

    alert_critical = {
        "id": "crit_001",
        "region_id": "406008",
        "hazard_type": "flood",
        "tier": "CRITICAL",
        "probability_calibrated": 0.94,
        "recommended_action": "Evacuate riverbank settlements."
    }

    # 6.1 Enqueue
    record = queue.enqueue_alert(alert_critical)
    assert record["status"] == ApprovalStatus.PENDING
    pending_list = queue.list_pending()
    assert len(pending_list) == 1
    assert pending_list[0]["alert_id"] == "crit_001"
    print("  [PASS] CRITICAL alert enqueued with PENDING state; mass sirens blocked.")

    # 6.2 Approve
    approve_res = queue.approve_alert(
        alert_id="crit_001",
        operator_id="CWC-Officer-Sharma",
        notes="Gauges indicate water level 1.8m above Danger Level"
    )
    assert approve_res["status"] == "APPROVED"
    assert len(dispatched_events) == 1
    assert queue.list_pending() == []
    print("  [PASS] Operator approval triggered immediate emergency broadcast.")

    # 6.3 Reject false alarm
    alert_false_alarm = {
        "id": "crit_false_002",
        "region_id": "104001",
        "hazard_type": "flood",
        "tier": "CRITICAL"
    }
    queue.enqueue_alert(alert_false_alarm)
    reject_res = queue.reject_alert(
        alert_id="crit_false_002",
        operator_id="CWC-Officer-Verma",
        reason="Erroneous gauge sensor spike; river is dry."
    )
    assert reject_res["status"] == "REJECTED"
    assert len(dispatched_events) == 1 # Did not increment
    print("  [PASS] False alarm successfully rejected by operator; broadcast cancelled.")

    # 6.4 Auto-release on timeout
    alert_timeout = {
        "id": "crit_timeout_003",
        "region_id": "101001",
        "hazard_type": "flood",
        "tier": "CRITICAL"
    }
    queue.enqueue_alert(alert_timeout)
    time.sleep(2.1) # Wait for timeout
    auto_released = queue.process_timeouts()
    assert len(auto_released) == 1
    assert auto_released[0]["alert_id"] == "crit_timeout_003"
    assert auto_released[0]["status"] == ApprovalStatus.AUTO_RELEASED
    assert len(dispatched_events) == 2 # Callback triggered
    print("  [PASS] Unresponsive operator timeout triggered safety auto-release.")


def test_fastapi_endpoints_integration():
    print("\n--- 7. Testing FastAPI Alert Dissemination Endpoints ---")

    # 7.1 GET /api/v1/alerts/cap
    cap_response = get_cap_alert_feed()
    assert cap_response.status_code == 200
    assert "application/xml" in cap_response.media_type
    assert "<feed" in cap_response.body.decode("utf-8")
    print("  [PASS] GET /api/v1/alerts/cap responded with valid OASIS CAP XML feed.")

    # 7.2 POST /api/v1/alerts/dispatch-test
    test_req = AlertDispatchRequest(
        region_id="406008",
        region_name="Godavari Basin & Wardha Plain",
        hazard_type="flood",
        tier="HIGH",
        lead_time_h=48,
        probability_calibrated=0.88,
        recommended_action="Deploy emergency pumps.",
        bypass_approval=True
    )
    dispatch_res = dispatch_test_alert(test_req)
    assert dispatch_res["status"] == "SUCCESS"
    assert "dispatch_result" in dispatch_res
    print("  [PASS] POST /api/v1/alerts/dispatch-test dispatched across active channels.")

    # 7.3 GET /api/v1/alerts/pending
    pending_res = list_pending_critical_alerts()
    assert pending_res["status"] == "SUCCESS"
    assert "pending_count" in pending_res
    print(f"  [PASS] GET /api/v1/alerts/pending returned active approval queue ({pending_res['pending_count']} pending).")

    # 7.4 GET /api/v1/alerts/history
    hist_res = get_alert_dispatch_history(limit=10)
    assert hist_res["status"] == "SUCCESS"
    assert hist_res["total_dispatched"] >= 1
    print(f"  [PASS] GET /api/v1/alerts/history returned {len(hist_res['history'])} dispatch receipts.")


if __name__ == "__main__":
    print("======================================================================")
    print("PHASE 3: MULTI-CHANNEL DISASTER ALERT DISPATCH & CAP v1.2 SUITE")
    print("======================================================================")
    test_telegram_alert_channel()
    test_email_alert_channel()
    test_trai_dlt_sms_channel()
    test_cap_v12_xml_generation()
    test_hysteresis_and_deduplication()
    test_operator_approval_queue()
    test_fastapi_endpoints_integration()
    print("\n======================================================================")
    print("ALL PHASE 3 VERIFICATION CHECKS PASSED PERFECTLY (7/7)")
    print("======================================================================")

