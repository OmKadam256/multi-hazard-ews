-- =============================================================================
-- Migration 001 (DOWN): Revert PostGIS Multi-Hazard Early Warning Schema
-- =============================================================================

DROP TABLE IF EXISTS audit_log CASCADE;
DROP TABLE IF EXISTS subscribers CASCADE;
DROP TABLE IF EXISTS exposure CASCADE;
DROP TABLE IF EXISTS alerts CASCADE;
DROP TABLE IF EXISTS observations CASCADE;
DROP TABLE IF EXISTS forecasts CASCADE;
DROP TABLE IF EXISTS stations CASCADE;
DROP TABLE IF EXISTS river_segments CASCADE;
DROP TABLE IF EXISTS sub_basins CASCADE;
DROP TABLE IF EXISTS basins CASCADE;

DELETE FROM schema_migrations WHERE version = '001_init_postgis';
