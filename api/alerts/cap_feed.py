"""
==============================================================================
OASIS Common Alerting Protocol (CAP v1.2) XML Feed Generator (Phase 3)
==============================================================================
Generates standard CAP v1.2 emergency XML alerts compliant with National Disaster
Management Authority (NDMA) and Central Water Commission (CWC) protocols:
- OASIS Standard: urn:oasis:names:tc:emergency:cap:1.2
- Maps internal tiers (CRITICAL, HIGH, MODERATE, LOW) to CAP urgency/severity/certainty.
- Encodes geospatial bounding polygons, HydroSHEDS identifiers, and telemetry.
- Outputs schema-valid XML string or multi-alert syndication feed.
==============================================================================
"""

import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from xml.dom import minidom

CAP_NAMESPACE = "urn:oasis:names:tc:emergency:cap:1.2"
SENDER_ID = "cwc-eews@ndma.gov.in"

# Mapping from EEWS alert tiers to OASIS CAP v1.2 attributes
TIER_TO_CAP = {
    "CRITICAL": {
        "urgency": "Immediate",
        "severity": "Extreme",
        "certainty": "Observed",
        "event_code": "FFW" # Flash Flood Warning
    },
    "HIGH": {
        "urgency": "Expected",
        "severity": "Severe",
        "certainty": "Likely",
        "event_code": "FLW" # Flood Warning
    },
    "MODERATE": {
        "urgency": "Future",
        "severity": "Moderate",
        "certainty": "Possible",
        "event_code": "FLA" # Flood Watch / Advisory
    },
    "LOW": {
        "urgency": "Past",
        "severity": "Minor",
        "certainty": "Unlikely",
        "event_code": "FLS" # Flood Statement
    }
}


class CAPFeedGenerator:
    """
    Constructs and serializes OASIS CAP v1.2 XML alerts and feeds.
    """

    def __init__(self, sender: str = SENDER_ID):
        self.sender = sender

    def create_cap_alert(self, alert_data: Dict[str, Any]) -> ET.Element:
        """
        Builds an ElementTree XML node representing a single CAP v1.2 <alert>.
        """
        tier = alert_data.get("tier", "HIGH").upper()
        cap_attrs = TIER_TO_CAP.get(tier, TIER_TO_CAP["HIGH"])
        hazard_type = alert_data.get("hazard_type", "flood").capitalize()
        region_id = str(alert_data.get("region_id", "406008"))
        region_name = alert_data.get("region_name", f"Catchment {region_id}")
        lead_time = alert_data.get("lead_time_h", 48)
        prob = alert_data.get("probability_calibrated", alert_data.get("calibrated_prob", 0.0))
        spei = alert_data.get("spei_forecast", alert_data.get("forecast_index"))
        action = alert_data.get("recommended_action", "Initiate localized emergency staging.")

        now_dt = datetime.utcnow()
        sent_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%S-00:00")
        expires_iso = (now_dt + timedelta(hours=lead_time)).strftime("%Y-%m-%dT%H:%M:%S-00:00")
        identifier = f"IN-EEWS-CWC-{now_dt.strftime('%Y%m%d%H%M')}-{region_id}-{str(uuid.uuid4())[:8]}"

        # Root: <alert>
        alert = ET.Element("alert", xmlns=CAP_NAMESPACE)
        ET.SubElement(alert, "identifier").text = identifier
        ET.SubElement(alert, "sender").text = self.sender
        ET.SubElement(alert, "sent").text = sent_iso
        ET.SubElement(alert, "status").text = "Actual"
        ET.SubElement(alert, "msgType").text = "Alert"
        ET.SubElement(alert, "scope").text = "Public"
        ET.SubElement(alert, "code").text = "IPAWS-1.0"

        # Child: <info>
        info = ET.SubElement(alert, "info")
        ET.SubElement(info, "language").text = "en-IN"
        ET.SubElement(info, "category").text = "Met" if hazard_type.lower() == "flood" else "Other"
        ET.SubElement(info, "event").text = f"{hazard_type} Early Warning ({tier})"
        ET.SubElement(info, "urgency").text = cap_attrs["urgency"]
        ET.SubElement(info, "severity").text = cap_attrs["severity"]
        ET.SubElement(info, "certainty").text = cap_attrs["certainty"]

        event_code = ET.SubElement(info, "eventCode")
        ET.SubElement(event_code, "valueName").text = "SAME"
        ET.SubElement(event_code, "value").text = cap_attrs["event_code"]

        ET.SubElement(info, "expires").text = expires_iso
        ET.SubElement(info, "headline").text = f"{tier} {hazard_type} Advisory for {region_name} (+{lead_time}h)"
        ET.SubElement(info, "description").text = (
            f"Physical hydro-meteorological model projects calibrated {hazard_type.lower()} "
            f"probability of {float(prob)*100:.1f}% within the next {lead_time} hours in {region_name}."
        )
        ET.SubElement(info, "instruction").text = action
        ET.SubElement(info, "web").text = f"https://eews-india.gov.in/map?basin={region_id}"
        ET.SubElement(info, "contact").text = "Central Water Commission Floods Control Desk: cwc-flood@nic.in"

        # Parameters
        param_prob = ET.SubElement(info, "parameter")
        ET.SubElement(param_prob, "valueName").text = "Calibrated_Probability"
        ET.SubElement(param_prob, "value").text = f"{float(prob):.4f}"

        if spei is not None:
            param_spei = ET.SubElement(info, "parameter")
            ET.SubElement(param_spei, "valueName").text = "SPEI_Deficit"
            ET.SubElement(param_spei, "value").text = f"{float(spei):.2f}"

        # Child: <area>
        area = ET.SubElement(info, "area")
        ET.SubElement(area, "areaDesc").text = region_name

        # Coordinates/Polygon if provided
        polygon_coords = alert_data.get("polygon")
        if polygon_coords:
            ET.SubElement(area, "polygon").text = str(polygon_coords)

        # Geocode (HydroSHEDS Basin identifier)
        geocode = ET.SubElement(area, "geocode")
        ET.SubElement(geocode, "valueName").text = "HYBAS_ID"
        ET.SubElement(geocode, "value").text = region_id

        return alert

    def to_xml_string(self, alert_element: ET.Element, pretty: bool = True) -> str:
        """Serializes XML element to string with optional indentation."""
        raw_bytes = ET.tostring(alert_element, encoding="utf-8")
        if pretty:
            reparsed = minidom.parseString(raw_bytes)
            return reparsed.toprettyxml(indent="  ")
        return raw_bytes.decode("utf-8")

    def generate_cap_feed(self, alerts_list: List[Dict[str, Any]]) -> str:
        """
        Generates an XML feed containing all active CAP v1.2 alerts.
        """
        feed = ET.Element("feed", xmlns="http://www.w3.org/2005/Atom")
        ET.SubElement(feed, "title").text = "India Multi-Hazard EEWS - OASIS CAP v1.2 Public Alert Feed"
        ET.SubElement(feed, "updated").text = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        ET.SubElement(feed, "id").text = "urn:uuid:india-eews-cap-feed"

        for alert_data in alerts_list:
            entry = ET.SubElement(feed, "entry")
            ET.SubElement(entry, "id").text = f"urn:uuid:{alert_data.get('id', str(uuid.uuid4()))}"
            ET.SubElement(entry, "title").text = f"{alert_data.get('tier')} {alert_data.get('hazard_type', 'Flood')} Warning"
            ET.SubElement(entry, "updated").text = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            content = ET.SubElement(entry, "content", type="application/xml")
            cap_alert = self.create_cap_alert(alert_data)
            content.append(cap_alert)

        return self.to_xml_string(feed, pretty=True)

