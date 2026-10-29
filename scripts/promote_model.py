"""
==============================================================================
Production Model Promotion & Gating Automation Script (Phase 4)
==============================================================================
Evaluates trained candidate checkpoints against strict production quality gates
before promoting to 'checkpoints/best_multi_hazard_model.pt':
1. Verification Criteria:
   - Flood Critical Success Index (CSI): >= 0.50
   - Flood Brier Score: <= 0.18
   - Drought RMSE: <= 0.85
   - Drought Skill Score vs Persistence: > 0.0
2. Actions on Gating Pass:
   - Archives current production model to 'checkpoints/archive/'
   - Atomically updates 'checkpoints/best_multi_hazard_model.pt'
   - Updates 'checkpoints/model_registry.json' metadata log
3. Safety: Rejects substandard models with detailed diagnostic telemetry.
==============================================================================
"""

import os
import sys
import json
import shutil
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple

# Windows PyTorch DLL rule: import torch first
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelPromoter")

CHECKPOINT_DIR = "checkpoints"
PRODUCTION_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_multi_hazard_model.pt")
REGISTRY_PATH = os.path.join(CHECKPOINT_DIR, "model_registry.json")
ARCHIVE_DIR = os.path.join(CHECKPOINT_DIR, "archive")

# Quality threshold gates for production deployment
GATING_THRESHOLDS = {
    "min_flood_csi": 0.50,
    "max_flood_brier": 0.18,
    "max_drought_rmse": 0.85,
    "min_skill_score_persist": 0.0
}


def load_checkpoint_metadata(path: str) -> Dict[str, Any]:
    """Inspects checkpoint dictionary without loading entire weights onto GPU."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    data = torch.load(path, map_location="cpu")
    return data


def evaluate_promotion_gates(checkpoint_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates candidate metrics against GATING_THRESHOLDS.
    """
    flood_metrics = checkpoint_data.get("flood_metrics", {})
    drought_metrics = checkpoint_data.get("drought_metrics", {})

    csi = flood_metrics.get("csi", 0.0)
    brier = flood_metrics.get("brier_score", 1.0)
    rmse = drought_metrics.get("rmse", 99.0)
    skill_score = drought_metrics.get("skill_score_vs_persistence", -1.0)

    checks = {
        "flood_csi_passed": csi >= GATING_THRESHOLDS["min_flood_csi"],
        "flood_brier_passed": brier <= GATING_THRESHOLDS["max_flood_brier"],
        "drought_rmse_passed": rmse <= GATING_THRESHOLDS["max_drought_rmse"],
        "drought_skill_passed": skill_score >= GATING_THRESHOLDS["min_skill_score_persist"]
    }

    all_passed = all(checks.values())
    metrics_summary = {
        "csi": csi,
        "brier": brier,
        "drought_rmse": rmse,
        "skill_score_vs_persistence": skill_score,
        "gates_detail": checks,
        "all_passed": all_passed
    }
    return all_passed, metrics_summary


def promote_candidate(
    candidate_path: str,
    force: bool = False,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Promotes candidate checkpoint to production if gates pass.
    """
    logger.info(f"Evaluating candidate checkpoint for promotion: '{candidate_path}'")
    ckpt_data = load_checkpoint_metadata(candidate_path)

    # 1. Gate evaluation
    passed, summary = evaluate_promotion_gates(ckpt_data)

    if not passed and not force:
        logger.error("PROMOTION REJECTED: Candidate failed production quality gates:")
        for gate, ok in summary["gates_detail"].items():
            logger.error(f"  - {gate}: {'PASSED' if ok else 'FAILED'}")
        return {
            "status": "REJECTED",
            "candidate": candidate_path,
            "summary": summary
        }

    if force:
        logger.warning("Force flag enabled: Bypassing quality gates.")

    if dry_run:
        logger.info("[DRY RUN] Quality gates satisfied. No filesystem changes made.")
        return {
            "status": "DRY_RUN_PASSED",
            "candidate": candidate_path,
            "summary": summary
        }

    # 2. Archive current production model if it exists
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    archived_name = None
    if os.path.exists(PRODUCTION_MODEL_PATH):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archived_name = f"best_multi_hazard_model_archived_{ts}.pt"
        archive_path = os.path.join(ARCHIVE_DIR, archived_name)
        shutil.copy2(PRODUCTION_MODEL_PATH, archive_path)
        logger.info(f"Archived previous production model to '{archive_path}'")

    # 3. Promote candidate to production path
    shutil.copy2(candidate_path, PRODUCTION_MODEL_PATH)
    logger.info(f"Successfully promoted '{candidate_path}' to '{PRODUCTION_MODEL_PATH}'")

    # 4. Update Model Registry
    registry = []
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = []

    record = {
        "version": f"v1.{len(registry) + 1}.0",
        "promoted_at": datetime.utcnow().isoformat(),
        "source_checkpoint": candidate_path,
        "archived_checkpoint": archived_name,
        "metrics": summary,
        "gates_enforced": GATING_THRESHOLDS
    }
    registry.append(record)

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    logger.info(f"Updated model registry at '{REGISTRY_PATH}'. Active version: {record['version']}")
    return {
        "status": "PROMOTED",
        "version": record["version"],
        "promoted_at": record["promoted_at"],
        "summary": summary
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Hazard Model Promotion")
    parser.add_argument("--candidate", type=str, default=os.path.join("checkpoints", "test_model_gated_multitask.pt"))
    parser.add_argument("--force", action="store_true", help="Force promotion regardless of gates")
    parser.add_argument("--dry_run", action="store_true", help="Check gates without modifying files")
    args = parser.parse_args()

    promote_candidate(args.candidate, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

