"""
==============================================================================
Scheduler Base Abstraction (Phase 2)
==============================================================================
Defines the scheduler interface and data structures.
Enables transparent swapping between APScheduler (standalone daemon)
and Celery/Redis distributed workers without breaking business workflows.
==============================================================================
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable


class JobStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRYING = "RETRYING"


@dataclass
class JobRecord:
    job_id: str
    name: str
    status: JobStatus = JobStatus.IDLE
    interval_hours: float = 6.0
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    last_success: Optional[str] = None
    duration_seconds: float = 0.0
    run_count: int = 0
    failure_count: int = 0
    last_error: Optional[str] = None
    is_locked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status.value,
            "interval_hours": self.interval_hours,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "last_success": self.last_success,
            "duration_seconds": round(self.duration_seconds, 2),
            "run_count": self.run_count,
            "failure_count": self.failure_count,
            "last_error": self.last_error,
            "is_locked": self.is_locked,
        }


@dataclass
class DeadLetterEntry:
    entry_id: str
    job_id: str
    timestamp: str
    error_type: str
    error_message: str
    retry_attempts: int
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "job_id": self.job_id,
            "timestamp": self.timestamp,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "retry_attempts": self.retry_attempts,
            "payload": self.payload,
        }


class BaseScheduler(ABC):
    """
    Abstract scheduler protocol. APScheduler or Celery backends must implement
    these methods to ensure uniform job management and reliability telemetry.
    """

    @abstractmethod
    def start(self) -> None:
        """Starts the scheduler engine."""
        pass

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Gracefully halts the scheduler engine."""
        pass

    @abstractmethod
    def add_job(
        self,
        job_id: str,
        func: Callable,
        interval_seconds: int,
        name: Optional[str] = None
    ) -> None:
        """Schedules a recurring job."""
        pass

    @abstractmethod
    def trigger_job(self, job_id: str) -> Dict[str, Any]:
        """Manually executes a job immediately."""
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> Optional[JobRecord]:
        """Returns metadata for a specific job."""
        pass

    @abstractmethod
    def list_jobs(self) -> List[Dict[str, Any]]:
        """Returns metadata for all registered jobs."""
        pass

    @abstractmethod
    def get_dead_letter_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves dead-letter records for inspection."""
        pass

    @abstractmethod
    def get_staleness_status(self) -> Dict[str, Any]:
        """Returns overall data freshness and stale flags for UI display."""
        pass

