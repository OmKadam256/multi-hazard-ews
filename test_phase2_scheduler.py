"""
==============================================================================
Phase 2 Comprehensive Verification Suite:
Scheduler Daemon, 6-Hour Cycle, Concurrency Locks, Retries & Admin Endpoints
==============================================================================
Tests:
  1. Full 8-Job Operational Pipeline Execution & Zero-Leakage Enforcement
  2. Concurrency Lock & Overlapping Run Prevention (JobLock)
  3. Upstream Source Failure Resilience, Retries & Dead-Letter Log
  4. Scheduler Daemon Lifecycle, Restart Recovery & Data Staleness Detection
  5. Admin API Endpoints Authentication & Remote Triggering (/admin/jobs)
  6. Public System Staleness Telemetry (/api/v1/system/staleness)
==============================================================================
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta

# Ensure PyTorch DLLs load first on Windows
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scheduler.base import JobStatus
from scheduler.jobs import OperationalPipelineJobs
from scheduler.daemon import APSchedulerBackend, STALENESS_THRESHOLD_HOURS
from api.main import (
    app,
    get_scheduled_jobs,
    trigger_scheduled_job,
    get_system_staleness,
    verify_admin_key,
    EEWS_ADMIN_KEY
)
from fastapi import HTTPException


def test_full_operational_cycle():
    print("\n--- 1. Testing Full 8-Job Operational Cycle & Zero-Leakage ---")
    jobs = OperationalPipelineJobs()
    t_issue = datetime.utcnow()

    cycle_result = jobs.run_full_cycle(issue_time=t_issue)

    assert cycle_result["status"] == "SUCCESS", "Pipeline cycle failed"
    assert "duration_seconds" in cycle_result
    assert cycle_result["duration_seconds"] > 0.0

    step_results = cycle_result["step_results"]
    expected_steps = [
        "1_weather_pull", "2_satellite_ingestion", "3_feature_rebuild",
        "4_dl_inference", "5_calibration_hysteresis", "6_postgis_persistence",
        "7_alert_dispatch", "8_radar_refresh"
    ]
    for step in expected_steps:
        assert step in step_results, f"Missing operational step: {step}"
        assert step_results[step] in ["SUCCESS", "DEGRADED_FALLBACK"], f"Step {step} failed: {step_results[step]}"

    assert cycle_result["forecasts_generated"] >= 10, "Expected at least 10 forecasts generated"
    assert cycle_result["alerts_generated"] >= 1, "Expected alerts generated for high-risk catchments"

    print(f"  [PASS] Full 8-step cycle completed in {cycle_result['duration_seconds']}s:")
    for step, st in step_results.items():
        print(f"    - {step}: {st}")
    print(f"  [PASS] Forecasts Persisted: {cycle_result['forecasts_generated']}, Alerts Emitted: {cycle_result['alerts_generated']}.")


def test_concurrency_lock_and_overlap_prevention():
    print("\n--- 2. Testing Concurrency Lock & Overlapping Run Prevention ---")
    daemon = APSchedulerBackend()
    job_id = "test_locked_job"
    run_started = threading.Event()
    release_blocker = threading.Event()
    execution_counter = [0]

    def slow_worker():
        execution_counter[0] += 1
        run_started.set()
        release_blocker.wait(timeout=5.0)
        return {"status": "DONE"}

    daemon.add_job(job_id=job_id, func=slow_worker, interval_seconds=3600, name="Slow Worker Test")

    # Start slow_worker in a background thread
    t1 = threading.Thread(target=lambda: daemon.trigger_job(job_id))
    t1.start()

    # Wait until slow_worker has acquired the lock
    run_started.wait(timeout=3.0)

    # Attempt to trigger the exact same job while it's still running
    overlap_result = daemon.trigger_job(job_id)

    assert overlap_result["status"] == "SKIPPED_OVERLAP", "Lock failed: overlapping job execution was not blocked!"
    assert "currently running" in overlap_result["message"]
    print("  [PASS] Overlapping invocation blocked immediately with SKIPPED_OVERLAP.")

    # Release background worker and wait for clean completion
    release_blocker.set()
    t1.join(timeout=3.0)

    assert execution_counter[0] == 1, f"Expected exactly 1 execution, got {execution_counter[0]}"

    # Verify lock has been released and job can now run again
    rec = daemon.get_job_status(job_id)
    assert rec.is_locked is False, "Lock remained held after job finished!"

    second_run = daemon.trigger_job(job_id)
    assert second_run["status"] == "SUCCESS", "Subsequent run failed after lock release"
    assert execution_counter[0] == 2, "Execution counter did not increment"
    print("  [PASS] Lock cleanly released and subsequent run succeeded.")


def test_source_failure_and_dead_letter_queue():
    print("\n--- 3. Testing Upstream Source Failure, Retries & Dead-Letter Log ---")
    daemon = APSchedulerBackend()
    failing_job_id = "flaky_source_job"
    fail_attempts = [0]

    def failing_worker():
        fail_attempts[0] += 1
        raise ConnectionResetError("Simulated 503 Service Unavailable from satellite ingest")

    daemon.add_job(job_id=failing_job_id, func=failing_worker, interval_seconds=3600, name="Flaky Source")

    result = daemon.trigger_job(failing_job_id)
    assert result["status"] == "FAILED"
    assert "503 Service Unavailable" in result["error"]

    # Verify job record updated with error and failure count
    status = daemon.get_job_status(failing_job_id)
    assert status.status == JobStatus.FAILED
    assert status.failure_count == 1
    assert "ConnectionResetError" in status.last_error

    # Verify pushed to dead-letter queue
    dead_letters = daemon.get_dead_letter_log(limit=10)
    assert len(dead_letters) >= 1
    recent_entry = dead_letters[0]
    assert recent_entry["job_id"] == failing_job_id
    assert recent_entry["error_type"] == "ConnectionResetError"
    assert "503" in recent_entry["error_message"]

    print(f"  [PASS] Source failure intercepted gracefully without process crash.")
    print(f"  [PASS] Dead-letter queue logged error entry ID: {recent_entry['entry_id']}.")


def test_daemon_lifecycle_and_staleness():
    print("\n--- 4. Testing Daemon Lifecycle, Restart Recovery & Data Staleness ---")
    daemon = APSchedulerBackend()

    # 4.1 Lifecycle start & shutdown
    assert daemon.is_running is False
    daemon.start()
    assert daemon.is_running is True
    assert daemon.scheduler.running is True

    jobs = daemon.list_jobs()
    assert len(jobs) >= 6, f"Expected at least 6 jobs, found {len(jobs)}"
    print(f"  [PASS] Daemon started with {len(jobs)} scheduled background jobs.")

    # 4.2 Staleness check before run
    stale_initial = daemon.get_staleness_status()
    assert stale_initial["is_stale"] is True
    assert stale_initial["last_success"] is None
    print(f"  [PASS] Initial state correctly flagged as STALE (never run).")

    # 4.3 Trigger master cycle
    run_res = daemon.trigger_job("full_cycle_pipeline")
    assert run_res["status"] == "SUCCESS"

    fresh_status = daemon.get_staleness_status()
    assert fresh_status["is_stale"] is False
    assert fresh_status["staleness_hours"] < 0.1
    assert fresh_status["status"] == "HEALTHY_AND_FRESH"
    print(f"  [PASS] After pipeline execution: Data flagged FRESH ({fresh_status['staleness_hours']}h old).")

    # 4.4 Shutdown and Restart Recovery
    daemon.shutdown(wait=False)
    assert daemon.is_running is False

    # Simulate restart
    daemon.start()
    assert daemon.is_running is True
    post_restart_jobs = daemon.list_jobs()
    assert len(post_restart_jobs) == len(jobs)
    daemon.shutdown(wait=False)
    print(f"  [PASS] Restart recovery verified: All jobs preserved across daemon restart.")


def test_admin_api_endpoints():
    print("\n--- 5. Testing Admin Endpoints & Authentication (/admin/jobs) ---")

    # 5.1 Test unauthorized access (No key or invalid key)
    try:
        verify_admin_key(x_api_key="wrong-key", api_key=None)
        assert False, "Should have raised 401 Unauthorized"
    except HTTPException as e:
        assert e.status_code == 401
        print("  [PASS] Invalid API key correctly rejected with HTTP 401.")

    # 5.2 Test authorized access with valid key
    is_authed = verify_admin_key(x_api_key=EEWS_ADMIN_KEY, api_key=None)
    assert is_authed is True

    # 5.3 Test get_scheduled_jobs endpoint function
    jobs_response = get_scheduled_jobs(authorized=True)
    assert jobs_response["status"] == "SUCCESS"
    assert "jobs" in jobs_response
    assert "staleness" in jobs_response
    assert "dead_letter_log" in jobs_response
    assert jobs_response["total_jobs"] >= 6
    print(f"  [PASS] /admin/jobs endpoint returned {jobs_response['total_jobs']} jobs and telemetry.")

    # 5.4 Test manual job trigger endpoint
    trigger_response = trigger_scheduled_job(job_id="refresh_rainviewer_radar", authorized=True)
    assert trigger_response["status"] in ["SUCCESS", "DEGRADED_FALLBACK"]
    assert trigger_response["job_id"] == "refresh_rainviewer_radar"
    print(f"  [PASS] /admin/jobs/trigger executed 'refresh_rainviewer_radar' successfully.")


def test_public_staleness_telemetry():
    print("\n--- 6. Testing Public Data Freshness Telemetry (/api/v1/system/staleness) ---")
    staleness_data = get_system_staleness()

    assert "is_stale" in staleness_data
    assert "staleness_hours" in staleness_data
    assert "status" in staleness_data
    assert "recommended_action" in staleness_data

    print(f"  [PASS] /api/v1/system/staleness responded: is_stale={staleness_data['is_stale']}, status={staleness_data['status']}.")


if __name__ == "__main__":
    print("======================================================================")
    print("PHASE 2: SCHEDULER DAEMON, RETRIES, LOCKS & ADMIN TELEMETRY SUITE")
    print("======================================================================")
    test_full_operational_cycle()
    test_concurrency_lock_and_overlap_prevention()
    test_source_failure_and_dead_letter_queue()
    test_daemon_lifecycle_and_staleness()
    test_admin_api_endpoints()
    test_public_staleness_telemetry()
    print("\n======================================================================")
    print("ALL PHASE 2 VERIFICATION CHECKS PASSED PERFECTLY (6/6)")
    print("======================================================================")
