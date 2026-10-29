"""
==============================================================================
PostgreSQL / PostGIS Spatial Database Models (Phase 1)
==============================================================================
Defines the 10 spatial and operational tables for multi-hazard early warning:
1. basins (Major river catchments)
2. sub_basins (Pfafstetter HydroSHEDS sub-basins with MultiPolygon & GIST index)
3. river_segments (Stream channels, Strahler order, MultiLineString)
4. stations (CWC / IMD river gauges & telemetry, Point)
5. forecasts (Physical hazard predictions strictly separated from exposure)
6. observations (Satellite rain, SAR flood-extent, NDVI, SPEI & confidence)
7. alerts (State machine, hysteresis state, dispatch status)
8. exposure (Demographic, educational, healthcare & agricultural exposure)
9. subscribers (Multi-channel emergency alert subscribers with consent)
10. audit_log (Compliance & operational dispatch audit trail)
==============================================================================
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import (
    Column, String, Integer, BigInteger, Float, Boolean, DateTime,
    ForeignKey, Text, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

try:
    from geoalchemy2 import Geometry
    HAS_GEOALCHEMY = True
except ImportError:
    HAS_GEOALCHEMY = False
    Geometry = None

Base = declarative_base()


class Basin(Base):
    """1. Major River Basin (e.g. Godavari, Ganga, Brahmaputra)."""
    __tablename__ = "basins"

    id = Column(BigInteger, primary_key=True, autoincrement=False)
    basin_name = Column(String(120), nullable=False)
    river_system = Column(String(120), nullable=False)
    area_km2 = Column(Float, nullable=False)
    
    if HAS_GEOALCHEMY:
        geom = Column(Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=True), nullable=True)
    else:
        geom = Column(Text, nullable=True)

    sub_basins = relationship("SubBasin", back_populates="basin", cascade="all, delete-orphan")


class SubBasin(Base):
    """2. HydroSHEDS Pfafstetter Sub-basin watershed."""
    __tablename__ = "sub_basins"

    hybas_id = Column(BigInteger, primary_key=True, autoincrement=False)
    basin_id = Column(BigInteger, ForeignKey("basins.id", ondelete="SET NULL"), nullable=True)
    pfaf_code = Column(String(50), nullable=True, index=True)
    sub_basin_name = Column(String(150), nullable=False)
    area_km2 = Column(Float, nullable=False)
    upstream_area_km2 = Column(Float, nullable=True)

    if HAS_GEOALCHEMY:
        geom = Column(Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=True), nullable=False)
    else:
        geom = Column(Text, nullable=False)

    basin = relationship("Basin", back_populates="sub_basins")
    exposure = relationship("Exposure", back_populates="sub_basin", uselist=False, cascade="all, delete-orphan")


class RiverSegment(Base):
    """3. River Stream Segment (Strahler hydrographic order)."""
    __tablename__ = "river_segments"

    segment_id = Column(String(64), primary_key=True)
    hybas_id = Column(BigInteger, ForeignKey("sub_basins.hybas_id", ondelete="CASCADE"), nullable=True)
    river_name = Column(String(120), nullable=False)
    order_strahler = Column(Integer, default=1)
    length_km = Column(Float, nullable=False)

    if HAS_GEOALCHEMY:
        geom = Column(Geometry(geometry_type="MULTILINESTRING", srid=4326, spatial_index=True), nullable=False)
    else:
        geom = Column(Text, nullable=False)


class Station(Base):
    """4. Central Water Commission (CWC) or IMD Hydrometeorological Station."""
    __tablename__ = "stations"

    station_id = Column(String(64), primary_key=True)
    name = Column(String(150), nullable=False)
    agency = Column(String(50), default="CWC")
    hybas_id = Column(BigInteger, ForeignKey("sub_basins.hybas_id", ondelete="SET NULL"), nullable=True)
    warning_level_m = Column(Float, nullable=True)
    danger_level_m = Column(Float, nullable=True)

    if HAS_GEOALCHEMY:
        geom = Column(Geometry(geometry_type="POINT", srid=4326, spatial_index=True), nullable=False)
    else:
        geom = Column(Text, nullable=False)


class Forecast(Base):
    """
    5. Physical Hazard Prediction Record.
    SCIENTIFIC RULE: Decoupled strictly from exposure metrics.
    """
    __tablename__ = "forecasts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    hazard_type = Column(String(30), nullable=False, index=True) # flood, drought, compound
    region_id = Column(String(64), nullable=False, index=True)   # hybas_id or grid id
    lead_time_h = Column(Integer, nullable=False)
    probability_raw = Column(Float, nullable=True)
    probability_calibrated = Column(Float, nullable=True)
    forecast_index = Column(Float, nullable=True)                # SPEI continuous Z-score
    tier = Column(String(30), nullable=False)                    # LOW, MODERATE, HIGH, CRITICAL
    uncertainty = Column(Float, nullable=False, default=0.14)
    model_version = Column(String(64), nullable=False)
    issued_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    valid_until = Column(DateTime, nullable=False)


class Observation(Base):
    """6. Earth Observation Ingestion Record (Rain, SAR, NDVI, SPEI & Confidence)."""
    __tablename__ = "observations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    hybas_id = Column(BigInteger, ForeignKey("sub_basins.hybas_id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    rain_24h_mm = Column(Float, nullable=True)
    sar_inundation_ref = Column(String(120), nullable=True)
    ndvi = Column(Float, nullable=True)
    spei = Column(Float, nullable=True)
    confidence = Column(Float, nullable=False, default=1.0)
    source = Column(String(64), nullable=False)                  # CHIRPS, Sentinel-1, MODIS, ERA5-Land


class Alert(Base):
    """7. Active Early Warning Alert with Temporal Hysteresis & Dispatch Status."""
    __tablename__ = "alerts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    region_id = Column(String(64), nullable=False, index=True)
    hazard_type = Column(String(30), nullable=False)
    tier = Column(String(30), nullable=False)                    # LOW, MODERATE, HIGH, CRITICAL
    state = Column(String(30), nullable=False, default="NEW")    # NEW, ACKNOWLEDGED, RESOLVED, EXPIRED
    hysteresis_state = Column(String(30), nullable=False, default="STABLE")
    dispatch_status = Column(String(30), nullable=False, default="PENDING") # PENDING, SENT, FAILED
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    triggered_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class Exposure(Base):
    """
    8. Demographic and Critical Infrastructure Exposure.
    SCIENTIFIC RULE: Stored in a separate table to guarantee physical hazard decoupling.
    """
    __tablename__ = "exposure"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    hybas_id = Column(BigInteger, ForeignKey("sub_basins.hybas_id", ondelete="CASCADE"), unique=True, nullable=False)
    sub_basin_name = Column(String(150), nullable=False)
    population_total = Column(Integer, nullable=False, default=0)
    schools_count = Column(Integer, nullable=False, default=0)
    hospitals_count = Column(Integer, nullable=False, default=0)
    cropland_ha = Column(Float, nullable=False, default=0.0)
    road_km = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    sub_basin = relationship("SubBasin", back_populates="exposure")


class Subscriber(Base):
    """9. Multi-Channel Alert Subscriber with Consent."""
    __tablename__ = "subscribers"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    channel = Column(String(30), nullable=False)                 # telegram, email, sms, cap_webhook
    destination = Column(String(255), nullable=False)            # phone, email, chat_id, or webhook URL
    region_id = Column(String(64), nullable=False, index=True)   # subscribed sub-basin or 'ALL'
    consent_flag = Column(Boolean, nullable=False, default=True) # Explicit TRAI / GDPR opt-in
    is_verified = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class AuditLog(Base):
    """10. System and Dispatch Compliance Audit Trail."""
    __tablename__ = "audit_log"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type = Column(String(64), nullable=False, index=True)
    actor = Column(String(64), nullable=False, default="SYSTEM")
    payload_json = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
