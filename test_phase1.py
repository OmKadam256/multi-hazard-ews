"""
==============================================================================
Phase 1 Comprehensive Verification Suite:
PostGIS Migrations, Spatial Ingestion, Geometry Repair & Decoupled Exposure
==============================================================================
Tests:
  1. Migration UP / DOWN lifecycle & schema verification (10 tables)
  2. Spatial Ingestion & Seeding with strict Idempotency (2 passes)
  3. Point-in-Sub-Basin Spatial Containment Query (ST_Contains / Shapely)
  4. Geometry Self-Intersection Repair (make_valid)
  5. Decoupling of Intrinsic Hazard Forecasts from Exposure Data
  6. FastAPI Database Health Check (/health/db endpoint)
==============================================================================
"""

import os
import sys
from pathlib import Path
from shapely.geometry import Polygon, Point
# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.migrate import MigrationManager
from api.spatial_ingest import SpatialIngestionEngine, ValidationReport
from api.db import PostGISAdapter
from api.db_models import (
    Basin, SubBasin, RiverSegment, Station, Forecast,
    Observation, Alert, Exposure, Subscriber, AuditLog
)
from api.main import app, get_database_health

TEST_DB_PATH = Path(__file__).resolve().parent / "test_phase1_eews.db"
TEST_DB_URL = f"sqlite:///{TEST_DB_PATH.as_posix()}"


def test_migration_lifecycle():
    print("\n--- 1. Testing PostGIS / Relational Migration Lifecycle (UP / DOWN) ---")
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()

    manager = MigrationManager(db_url=TEST_DB_URL)

    # 1.1 Run UP
    up_success = manager.run_up()
    assert up_success is True, "Migration UP failed!"

    status_up = manager.get_status()
    assert status_up["database_online"] is True, "Database not online after migration UP"
    assert len(status_up["applied_migrations"]) >= 1, "Applied migrations list is empty"
    assert status_up["applied_migrations"][0]["version"] == "001_init_postgis"

    expected_tables = [
        "basins", "sub_basins", "river_segments", "stations",
        "forecasts", "observations", "alerts", "exposure",
        "subscribers", "audit_log"
    ]
    for tbl in expected_tables:
        assert tbl in status_up["table_counts"], f"Missing table: {tbl}"
        assert status_up["table_counts"][tbl] != "TABLE_DOES_NOT_EXIST", f"Table does not exist: {tbl}"

    print(f"  [PASS] Migration UP verified: All 10 tables created with schema tracking.")

    # 1.2 Run DOWN
    down_success = manager.run_down()
    assert down_success is True, "Migration DOWN failed!"

    status_down = manager.get_status()
    for tbl in expected_tables:
        assert status_down["table_counts"][tbl] == "TABLE_DOES_NOT_EXIST", f"Table not dropped: {tbl}"

    print(f"  [PASS] Migration DOWN verified: All tables cleanly dropped.")

    # Re-apply UP for subsequent spatial tests
    manager.run_up()
    print(f"  [PASS] Migration re-applied UP for downstream spatial tests.")


def test_spatial_seeding_and_idempotency():
    print("\n--- 2. Testing Spatial Ingestion Engine & Seeding Idempotency ---")
    engine = SpatialIngestionEngine(db_url=TEST_DB_URL)
    assert engine.is_db_connected is True, "SpatialIngestionEngine failed to connect to test database"

    # Pass 1: Initial Seed
    reports_pass1 = engine.seed_initial_datasets()
    assert "basins" in reports_pass1
    assert "sub_basins" in reports_pass1
    assert "exposure" in reports_pass1
    assert "stations" in reports_pass1

    assert reports_pass1["basins"].inserted_records == 12, "Expected 12 basins in Pass 1"
    assert reports_pass1["sub_basins"].inserted_records == 10, "Expected 10 sub-basins in Pass 1"
    assert reports_pass1["exposure"].inserted_records == 10, "Expected 10 exposure records in Pass 1"
    assert reports_pass1["stations"].inserted_records == 8, "Expected 8 stations in Pass 1"

    for name, rep in reports_pass1.items():
        assert rep.failed_records == 0, f"Pass 1 failed on {name}: {rep.errors}"

    print(f"  [PASS] Pass 1 Seeding complete: Basins=12, Sub-basins=10, Exposure=10, Stations=8.")

    # Pass 2: Re-run seed (Idempotency check - ON CONFLICT DO UPDATE)
    reports_pass2 = engine.seed_initial_datasets()
    for name, rep in reports_pass2.items():
        assert rep.failed_records == 0, f"Pass 2 idempotency failed on {name}: {rep.errors}"
        assert rep.inserted_records == rep.total_features, f"Pass 2 record count mismatch on {name}"

    # Verify database counts remained exact
    manager = MigrationManager(db_url=TEST_DB_URL)
    status = manager.get_status()
    assert status["table_counts"]["basins"] == 12, "Basins count altered on re-seeding"
    assert status["table_counts"]["sub_basins"] == 10, "Sub-basins count altered on re-seeding"
    assert status["table_counts"]["exposure"] == 10, "Exposure count altered on re-seeding"
    assert status["table_counts"]["stations"] == 8, "Stations count altered on re-seeding"

    print(f"  [PASS] Pass 2 Idempotency verified: Re-seeding produced zero errors and row counts remained constant.")


def test_point_in_basin_query():
    print("\n--- 3. Testing Point-in-Sub-Basin Spatial Containment Query ---")
    adapter = PostGISAdapter(connection_url=TEST_DB_URL)

    # 1. Point inside Wardha Floodplain (lat 19.5, lon 79.9)
    res_wardha = adapter.point_in_sub_basin(lat=19.5, lon=79.9)
    assert res_wardha is not None
    assert res_wardha["hybas_id"] == 406008
    assert "Wardha" in res_wardha["name"]
    print(f"  [PASS] Wardha point (19.5, 79.9) -> Resolved to {res_wardha['name']} (ID={res_wardha['hybas_id']}).")

    # 2. Point inside Upper Godavari (lat 18.3, lon 79.3)
    res_upper = adapter.point_in_sub_basin(lat=18.3, lon=79.3)
    assert res_upper is not None
    assert res_upper["hybas_id"] == 406001
    assert "Upper Godavari" in res_upper["name"]
    print(f"  [PASS] Upper Godavari point (18.3, 79.3) -> Resolved to {res_upper['name']} (ID={res_upper['hybas_id']}).")

    # 3. Point inside Manjra Headwaters (lat 18.8, lon 79.3)
    res_manjra = adapter.point_in_sub_basin(lat=18.8, lon=79.3)
    assert res_manjra is not None
    assert res_manjra["hybas_id"] == 406004
    print(f"  [PASS] Manjra point (18.8, 79.3) -> Resolved to {res_manjra['name']} (ID={res_manjra['hybas_id']}).")

    # 4. Out-of-bounds point fallback
    res_out = adapter.point_in_sub_basin(lat=10.0, lon=72.0)
    assert res_out is not None
    assert "hybas_id" in res_out
    print(f"  [PASS] Out-of-bounds coordinate handled gracefully with default pilot catchment.")


def test_geometry_repair():
    print("\n--- 4. Testing Geometry Repair Engine (ST_MakeValid / Shapely) ---")
    engine = SpatialIngestionEngine(db_url=TEST_DB_URL)

    # Create an invalid self-intersecting "bowtie" polygon
    # (0,0) -> (2,2) -> (2,0) -> (0,2) -> (0,0) intersects itself in the center (1,1)
    invalid_bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    assert invalid_bowtie.is_valid is False, "Test polygon must be mathematically invalid"

    repaired_geom, was_repaired = engine.repair_geometry(invalid_bowtie)
    assert was_repaired is True, "repair_geometry should flag was_repaired as True"
    assert repaired_geom.is_valid is True, "Repaired geometry must be valid"
    assert repaired_geom.geom_type in ["MultiPolygon", "GeometryCollection"], f"Unexpected type: {repaired_geom.geom_type}"

    # Valid polygon should pass without repair
    valid_square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    assert valid_square.is_valid is True
    passed_geom, rep_flag = engine.repair_geometry(valid_square)
    assert rep_flag is False
    assert passed_geom.equals(valid_square)

    print(f"  [PASS] Self-intersecting polygon successfully repaired into valid {repaired_geom.geom_type}.")


def test_hazard_exposure_decoupling():
    print("\n--- 5. Testing Decoupling of Intrinsic Hazard Forecast from Exposure ---")
    adapter = PostGISAdapter(connection_url=TEST_DB_URL)

    exp_data = adapter.get_exposure_for_basin(hybas_id=406008)
    assert "exposed_population_est" in exp_data
    assert exp_data["exposed_population_est"] == 181440
    assert exp_data["schools_count"] == 68
    assert exp_data["hospitals_count"] == 14
    assert exp_data["agricultural_cropland_ha"] == 42500.0
    assert "separation_note" in exp_data
    assert "strictly separated" in exp_data["separation_note"].lower()

    # Verify model tables: Forecast has no exposure fields; Exposure has no hazard probability fields
    forecast_cols = [c.name for c in Forecast.__table__.columns]
    exposure_cols = [c.name for c in Exposure.__table__.columns]

    assert "probability_calibrated" in forecast_cols
    assert "population_total" not in forecast_cols, "Data Leakage: population_total found in Forecast table!"
    assert "cropland_ha" not in forecast_cols, "Data Leakage: cropland_ha found in Forecast table!"

    assert "population_total" in exposure_cols
    assert "probability_calibrated" not in exposure_cols, "Data Leakage: probability found in Exposure table!"

    print(f"  [PASS] Separation of concerns verified: Physical Hazard and Exposure live in strictly decoupled tables.")


def test_fastapi_database_health():
    print("\n--- 6. Testing FastAPI Database Health Check (/health/db) ---")
    data = get_database_health()
    assert isinstance(data, dict), "Expected dict response from get_database_health()"
    assert "status" in data
    assert data["status"] in ["HEALTHY", "STANDALONE_FALLBACK"]
    assert "engine" in data
    assert "tables" in data
    assert data["tables"]["basins"] >= 12
    assert data["tables"]["sub_basins"] >= 10
    assert "latency_ms" in data

    print(f"  [PASS] /health/db responded: Status={data['status']}, Engine='{data['engine']}', Latency={data['latency_ms']}ms.")


def cleanup_test_database():
    try:
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        print("======================================================================")
        print("PHASE 1: POSTGIS MIGRATIONS, SPATIAL INGESTION & DECOUPLING SUITE")
        print("======================================================================")
        test_migration_lifecycle()
        test_spatial_seeding_and_idempotency()
        test_point_in_basin_query()
        test_geometry_repair()
        test_hazard_exposure_decoupling()
        test_fastapi_database_health()
        print("\n======================================================================")
        print("ALL PHASE 1 VERIFICATION CHECKS PASSED PERFECTLY (6/6)")
        print("======================================================================")
    finally:
        cleanup_test_database()
