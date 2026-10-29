"""
==============================================================================
Standalone Scheduler Daemon Runner (Phase 2)
==============================================================================
Entrypoint for dedicated Docker container or systemd background service.
Runs independently of the FastAPI web process to prevent blocking HTTP workers.

Usage:
  python -m scheduler.runner
==============================================================================
"""

import sys
import time
import signal
import logging
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scheduler.daemon import scheduler_daemon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SchedulerDaemon] %(message)s"
)
logger = logging.getLogger("SchedulerRunner")


def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}. Initiating graceful shutdown...")
    scheduler_daemon.shutdown(wait=True)
    logger.info("Scheduler daemon stopped cleanly. Exiting.")
    sys.exit(0)


def main():
    logger.info("======================================================================")
    logger.info("AI-Driven Multi-Hazard Early Warning System: 6-Hour Scheduler Daemon")
    logger.info("======================================================================")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Initializing operational jobs and APScheduler engine...")
    scheduler_daemon.start()

    logger.info("Registered recurring jobs:")
    for j in scheduler_daemon.list_jobs():
        logger.info(f"  - [{j['job_id']}] {j['name']} (every {j['interval_hours']}h) Next: {j['next_run']}")

    logger.info("Scheduler daemon running. Press Ctrl+C to terminate.")

    try:
        while True:
            time.sleep(60)
            staleness = scheduler_daemon.get_staleness_status()
            if staleness["is_stale"]:
                logger.warning(f"Data Staleness Notice: System data is {staleness['staleness_hours']}h old.")
    except (KeyboardInterrupt, SystemExit):
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()

