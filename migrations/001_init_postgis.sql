-- =============================================================================
-- Migration 001 (UP): PostGIS Multi-Hazard Early Warning Core Schema
-- Platform: PostgreSQL 15 + PostGIS 3.3
-- Coordinate Reference System: WGS 84 (EPSG:4326)
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- 1. Major River Basins
CREATE TABLE IF NOT EXISTS basins (
    id BIGINT PRIMARY KEY,
    basin_name VARCHAR(120) NOT NULL,
    river_system VARCHAR(120) NOT NULL,
    area_km2 DOUBLE PRECISION NOT NULL,
    geom GEOMETRY(MultiPolygon, 4326)
);
CREATE INDEX IF NOT EXISTS idx_basins_geom ON basins USING GIST (geom);

-- 2. HydroSHEDS Pfafstetter Sub-basins
CREATE TABLE IF NOT EXISTS sub_basins (
    hybas_id BIGINT PRIMARY KEY,
    basin_id BIGINT REFERENCES basins(id) ON DELETE SET NULL,
    pfaf_code VARCHAR(50),
    sub_basin_name VARCHAR(150) NOT NULL,
    area_km2 DOUBLE PRECISION NOT NULL,
    upstream_area_km2 DOUBLE PRECISION,
    geom GEOMETRY(MultiPolygon, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sub_basins_geom ON sub_basins USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_sub_basins_pfaf ON sub_basins (pfaf_code);

-- 3. River Stream Segments
CREATE TABLE IF NOT EXISTS river_segments (
    segment_id VARCHAR(64) PRIMARY KEY,
    hybas_id BIGINT REFERENCES sub_basins(hybas_id) ON DELETE CASCADE,
    river_name VARCHAR(120) NOT NULL,
    order_strahler INT DEFAULT 1,
    length_km DOUBLE PRECISION NOT NULL,
    geom GEOMETRY(MultiLineString, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_river_segments_geom ON river_segments USING GIST (geom);

-- 4. Hydrometeorological Gauging Stations
CREATE TABLE IF NOT EXISTS stations (
    station_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(150) NOT NULL,
    agency VARCHAR(50) DEFAULT 'CWC',
    hybas_id BIGINT REFERENCES sub_basins(hybas_id) ON DELETE SET NULL,
    warning_level_m DOUBLE PRECISION,
    danger_level_m DOUBLE PRECISION,
    geom GEOMETRY(Point, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stations_geom ON stations USING GIST (geom);

-- 5. Physical Hazard Forecasts (Decoupled from human exposure)
CREATE TABLE IF NOT EXISTS forecasts (
    id VARCHAR(36) PRIMARY KEY,
    hazard_type VARCHAR(30) NOT NULL,
    region_id VARCHAR(64) NOT NULL,
    lead_time_h INT NOT NULL,
    probability_raw DOUBLE PRECISION,
    probability_calibrated DOUBLE PRECISION,
    forecast_index DOUBLE PRECISION,
    tier VARCHAR(30) NOT NULL,
    uncertainty DOUBLE PRECISION NOT NULL DEFAULT 0.14,
    model_version VARCHAR(64) NOT NULL,
    issued_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    valid_until TIMESTAMP WITH TIME ZONE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_forecasts_lookup ON forecasts (region_id, hazard_type, issued_at);

-- 6. Ingested Earth Observations
CREATE TABLE IF NOT EXISTS observations (
    id VARCHAR(36) PRIMARY KEY,
    hybas_id BIGINT REFERENCES sub_basins(hybas_id) ON DELETE CASCADE NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    rain_24h_mm DOUBLE PRECISION,
    sar_inundation_ref VARCHAR(120),
    ndvi DOUBLE PRECISION,
    spei DOUBLE PRECISION,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    source VARCHAR(64) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observations_hybas_time ON observations (hybas_id, timestamp);

-- 7. Early Warning Alerts (with Hysteresis & Dispatch Status)
CREATE TABLE IF NOT EXISTS alerts (
    id VARCHAR(36) PRIMARY KEY,
    region_id VARCHAR(64) NOT NULL,
    hazard_type VARCHAR(30) NOT NULL,
    tier VARCHAR(30) NOT NULL,
    state VARCHAR(30) DEFAULT 'NEW' NOT NULL,
    hysteresis_state VARCHAR(30) DEFAULT 'STABLE' NOT NULL,
    dispatch_status VARCHAR(30) DEFAULT 'PENDING' NOT NULL,
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts (is_active, region_id);

-- 8. Exposure & Demographics (Decoupled from hazard likelihood)
CREATE TABLE IF NOT EXISTS exposure (
    id VARCHAR(36) PRIMARY KEY,
    hybas_id BIGINT REFERENCES sub_basins(hybas_id) ON DELETE CASCADE UNIQUE NOT NULL,
    sub_basin_name VARCHAR(150) NOT NULL,
    population_total INT DEFAULT 0 NOT NULL,
    schools_count INT DEFAULT 0 NOT NULL,
    hospitals_count INT DEFAULT 0 NOT NULL,
    cropland_ha DOUBLE PRECISION DEFAULT 0.0 NOT NULL,
    road_km DOUBLE PRECISION DEFAULT 0.0 NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- 9. Alert Notification Subscribers
CREATE TABLE IF NOT EXISTS subscribers (
    id VARCHAR(36) PRIMARY KEY,
    channel VARCHAR(30) NOT NULL,
    destination VARCHAR(255) NOT NULL,
    region_id VARCHAR(64) NOT NULL,
    consent_flag BOOLEAN DEFAULT TRUE NOT NULL,
    is_verified BOOLEAN DEFAULT FALSE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_subscribers_region ON subscribers (region_id, channel);

-- 10. Audit Log & Compliance Ledger
CREATE TABLE IF NOT EXISTS audit_log (
    id VARCHAR(36) PRIMARY KEY,
    event_type VARCHAR(64) NOT NULL,
    actor VARCHAR(64) DEFAULT 'SYSTEM' NOT NULL,
    payload_json TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log (timestamp);

-- Schema Migration Metadata
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(32) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

INSERT INTO schema_migrations (version) VALUES ('001_init_postgis')
ON CONFLICT (version) DO NOTHING;
