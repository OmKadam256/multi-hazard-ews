"""
==============================================================================
CLI Script: Spatial Data Ingestion & Seeding Entry Point (Phase 1)
==============================================================================
Usage:
  python scripts/ingest_spatial_data.py --seed
  python scripts/ingest_spatial_data.py --file data/hydrobasins_sample.geojson --table sub_basins
==============================================================================
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.spatial_ingest import SpatialIngestionEngine


def main():
    parser = argparse.ArgumentParser(description="Multi-Hazard PostGIS Spatial Data Ingestion")
    parser.add_argument("--seed", action="store_true", help="Seed 12 Indian river basins, pilot sub-basins, CWC stations & exposure")
    parser.add_argument("--file", type=str, default=None, help="Path to GeoJSON, Shapefile or GeoPackage to ingest")
    parser.add_argument("--table", type=str, default="sub_basins", help="Target table (sub_basins, river_segments, stations)")
    parser.add_argument("--db-url", type=str, default=None, help="Override database URL")

    args = parser.parse_args()
    engine = SpatialIngestionEngine(db_url=args.db_url)

    if args.seed or (not args.file):
        print("\n--- SEEDING OFFICIAL INDIAN RIVER BASINS & CWC GAUGES ---")
        reports = engine.seed_initial_datasets()
        for name, rep in reports.items():
            print(f"\n[REPORT: {name}]")
            print(f"  Target Table: {rep.target_table}")
            print(f"  Total Features: {rep.total_features}")
            print(f"  Valid Geometries: {rep.valid_geometries}")
            print(f"  Repaired Geometries: {rep.repaired_geometries}")
            print(f"  Inserted / Upserted: {rep.inserted_records}")
            print(f"  Failed: {rep.failed_records}")
            if rep.errors:
                print(f"  Errors: {rep.errors}")
    elif args.file:
        print(f"\n--- INGESTING SPATIAL DATA FROM: {args.file} ---")
        rep = engine.ingest_geojson_sub_basins(args.file)
        print(f"[REPORT: {rep.source_name}]")
        print(f"  Total Features: {rep.total_features}")
        print(f"  Valid: {rep.valid_geometries}, Repaired: {rep.repaired_geometries}")
        print(f"  Upserted: {rep.inserted_records}, Failed: {rep.failed_records}")
        if rep.errors:
            print(f"  Errors: {rep.errors}")


if __name__ == "__main__":
    main()
