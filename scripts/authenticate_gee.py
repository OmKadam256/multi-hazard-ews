"""
Google Earth Engine (GEE) Authentication & Live Connectivity Script
==================================================================
This script connects your local environment to the live Google Earth Engine servers.
"""

import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import ee
except ImportError:
    print("[ERROR] earthengine-api is not installed. Please run:")
    print("        .venv\\Scripts\\pip install earthengine-api")
    sys.exit(1)


def main():
    print("=" * 70)
    print("GOOGLE EARTH ENGINE (GEE) - LIVE CONNECTIVITY SETUP")
    print("=" * 70)
    print("\n[Step 1] Initiating Earth Engine OAuth authentication...")
    print("A browser window will open (or a link will appear below).")
    print("1. Log in with your Google Account that has Earth Engine access.")
    print("2. Accept the permissions and copy the authorization code.")
    print("3. Paste the authorization code back here when prompted.\n")

    try:
        # Use notebook/code auth mode which is most reliable across all Windows shells
        ee.Authenticate(auth_mode="notebook")
        print("\n[SUCCESS] OAuth authorization code received and saved successfully!")
    except Exception as e:
        print(f"\n[WARNING] OAuth authentication step encountered: {e}")
        print("Retrying with default browser flow...")
        try:
            ee.Authenticate()
        except Exception as err:
            print(f"[ERROR] Could not complete authentication: {err}")
            sys.exit(1)

    print("\n" + "-" * 70)
    print("[Step 2] Initializing Earth Engine Client & Live Test")
    print("-" * 70)
    
    project_id = input("\nEnter your Google Cloud Project ID (or press ENTER to use default): ").strip()
    
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("\n[SUCCESS] ee.Initialize() succeeded!")
    except Exception as e:
        print(f"\n[ERROR] Initialization failed: {e}")
        print("Tip: If you do not have a GCP project registered for Earth Engine,")
        print("visit: https://code.earthengine.google.com to register a free non-commercial project.")
        sys.exit(1)

    # Live Data Probing
    print("\n[Step 3] Querying live satellite collections on Google Cloud...")
    try:
        # Test Copernicus DEM
        dem_info = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").first().select("DEM").getInfo()
        print(f"  ✓ Copernicus GLO-30 (2024) DEM Live: Connected (Bands: {dem_info.get('bands', [{}])[0].get('id', 'DEM')})")

        # Test CHIRPS daily precipitation
        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate("2024-01-01", "2024-01-07")
        chirps_count = chirps.size().getInfo()
        print(f"  ✓ CHIRPS v2.0 Daily Precipitation: Connected ({chirps_count} tiles found in 7-day query)")

        # Test Sentinel-1 SAR GRD
        s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate("2024-01-01", "2024-01-07")
        s1_count = s1.size().getInfo()
        print(f"  ✓ Sentinel-1 C-SAR GRD: Connected ({s1_count} acquisitions found in 7-day query)")

        print("\n" + "=" * 70)
        print("ALL LIVE EARTH ENGINE DATASETS CONNECTED & OPERATIONAL!")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] Probing live datasets failed: {e}")


if __name__ == "__main__":
    main()
