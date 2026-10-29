"""
==============================================================================
Unified Migration Runner: PostgreSQL / PostGIS (Phase 1)
==============================================================================
Provides single-command database migration lifecycle:
  python scripts/migrate.py --up      (Apply pending migrations)
  python scripts/migrate.py --down    (Revert applied migrations)
  python scripts/migrate.py --status  (Check migration status & table counts)
  python scripts/migrate.py --reset   (Revert down then migrate up)
==============================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MigrateRunner")

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


class MigrationManager:
    """Manages versioned SQL migrations across PostgreSQL and local engines."""

    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/eews_db"
        )
        self.engine = create_engine(self.db_url, isolation_level="AUTOCOMMIT")

    def run_up(self) -> bool:
        """Applies 001_init_postgis.sql."""
        sql_path = MIGRATIONS_DIR / "001_init_postgis.sql"
        if not sql_path.exists():
            logger.error(f"Migration file missing: {sql_path}")
            return False

        logger.info(f"Applying migration UP: {sql_path.name} to {self._mask_url(self.db_url)}...")
        content = sql_path.read_text(encoding="utf-8")

        # Split statements cleanly
        statements = [stmt.strip() for stmt in content.split(";") if stmt.strip()]
        is_sqlite = (self.engine.dialect.name == "sqlite")

        with self.engine.connect() as conn:
            for s in statements:
                if is_sqlite:
                    # Translate PostGIS constructs for pure SQLite environments
                    if "extension" in s.lower() and "postgis" in s.lower():
                        continue
                    if "using gist" in s.lower():
                        s = s.replace("USING GIST", "").replace("using gist", "")
                    s = s.replace("GEOMETRY(MultiPolygon, 4326)", "TEXT")
                    s = s.replace("GEOMETRY(MultiLineString, 4326)", "TEXT")
                    s = s.replace("GEOMETRY(Point, 4326)", "TEXT")
                    s = s.replace("TIMESTAMP WITH TIME ZONE", "TIMESTAMP")
                    s = s.replace("DOUBLE PRECISION", "FLOAT")
                try:
                    conn.execute(text(s))
                except Exception as e:
                    # Ignore duplicate extension or index creation if already present
                    if "already exists" in str(e).lower():
                        continue
                    logger.warning(f"Warning on SQL execution: {e}")

        logger.info("[SUCCESS] Migration 001_init_postgis applied successfully.")
        return True

    def run_down(self) -> bool:
        """Reverts 001_down.sql."""
        sql_path = MIGRATIONS_DIR / "001_down.sql"
        if not sql_path.exists():
            logger.error(f"Migration file missing: {sql_path}")
            return False

        logger.info(f"Applying migration DOWN: {sql_path.name}...")
        content = sql_path.read_text(encoding="utf-8")
        statements = [stmt.strip() for stmt in content.split(";") if stmt.strip()]
        is_sqlite = (self.engine.dialect.name == "sqlite")

        with self.engine.connect() as conn:
            for s in statements:
                if is_sqlite and "cascade" in s.lower():
                    s = s.replace("CASCADE", "").replace("cascade", "")
                try:
                    conn.execute(text(s))
                except Exception as e:
                    logger.warning(f"Warning on DROP statement: {e}")

        logger.info("[SUCCESS] Migration 001_down executed. All tables dropped.")
        return True

    def get_status(self) -> Dict[str, Any]:
        """Checks applied version and row counts."""
        tables = [
            "basins", "sub_basins", "river_segments", "stations",
            "forecasts", "observations", "alerts", "exposure",
            "subscribers", "audit_log"
        ]
        status = {"applied_migrations": [], "table_counts": {}, "database_online": False}

        try:
            with self.engine.connect() as conn:
                status["database_online"] = True
                # Check migration history
                try:
                    res = conn.execute(text("SELECT version, applied_at FROM schema_migrations ORDER BY applied_at"))
                    status["applied_migrations"] = [{"version": r[0], "applied_at": str(r[1])} for r in res.fetchall()]
                except Exception:
                    status["applied_migrations"] = []

                # Check table counts
                for t in tables:
                    try:
                        res = conn.execute(text(f"SELECT COUNT(*) FROM {t}"))
                        status["table_counts"][t] = res.scalar()
                    except Exception:
                        status["table_counts"][t] = "TABLE_DOES_NOT_EXIST"
        except Exception as e:
            logger.error(f"Could not connect to database: {e}")
            status["error"] = str(e)

        return status

    def _mask_url(self, url: str) -> str:
        if "@" in url:
            parts = url.split("@")
            return f"{parts[0].split(':')[0]}://****:****@{parts[1]}"
        return url


def main():
    parser = argparse.ArgumentParser(description="Multi-Hazard PostGIS Migration Runner")
    parser.add_argument("--up", action="store_true", help="Apply pending migrations")
    parser.add_argument("--down", action="store_true", help="Revert migrations")
    parser.add_argument("--status", action="store_true", help="Display migration status and table counts")
    parser.add_argument("--reset", action="store_true", help="Revert down and migrate up")
    parser.add_argument("--db-url", type=str, default=None, help="Override database connection URL")

    args = parser.parse_args()
    manager = MigrationManager(db_url=args.db_url)

    if args.down:
        manager.run_down()
    elif args.reset:
        manager.run_down()
        manager.run_up()
    elif args.status:
        st = manager.get_status()
        print("\n--- DATABASE MIGRATION STATUS ---")
        print(f"Online: {st.get('database_online')}")
        print(f"Applied Migrations: {st.get('applied_migrations')}")
        print("Table Counts:")
        for t, count in st.get("table_counts", {}).items():
            print(f"  - {t}: {count}")
    else:
        # Default action is --up
        manager.run_up()


if __name__ == "__main__":
    main()
