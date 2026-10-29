# ============================================================================
# INSTANT RESUME FILE — AI Multi-Hazard Early Warning Platform (India)
# ============================================================================
# READ THIS FIRST. This file contains EVERYTHING needed to continue building
# this project from a fresh session. Paste this file's contents (or point the
# agent to it) and say "read RESUME_INSTRUCTIONS.md and continue from where
# it left off".
# ============================================================================

## 1. PROJECT IDENTITY & LOCATION

- **Name**: AI-Driven Multi-Hazard Early Warning & GeoIntelligence Platform (India)
- **Project Root**: `C:\Users\DELL\.gemini\antigravity-ide\scratch\project`
- **Python venv**: `.\.venv\Scripts\python.exe` (Python 3.10, PyTorch, GDAL, FastAPI)
- **Node/Frontend**: `dashboard/` (Next.js 14 App Router, React 18, TypeScript)
- **Scale**: Continental India, multi-hazard (Flood inundation +6h–+72h, Drought SPEI 1–6 months)
- **Constraint**: 100% free/open-source, zero paid APIs, zero credit cards

## 2. WHAT IS ALREADY BUILT & VERIFIED ✅

### Phase 0: Audit (DONE)
- Full codebase diagnostic against specification. No code changes.

### Phase 1: PostGIS Migrations & Spatial Engine (DONE — 6/6 tests)
- `migrations/001_init_postgis.sql` + `001_down.sql`: 10 tables with GIST spatial indices
- `scripts/migrate.py`: `--up`, `--down`, `--status`, `--reset` (PostGIS + SQLite)
- `api/spatial_ingest.py`: Multi-format loader, geometry repair, dialect-safe SQL
- `api/db.py`: Connection pooling, decoupled exposure queries
- `api/main.py`: `GET /health/db` endpoint
- `test_phase1.py`: 6/6 PASSED

### Phase 2: Scheduler Daemon, Locks, Retries & Admin (DONE — 6/6 tests)
- `scheduler/base.py`: `BaseScheduler`, `JobRecord`, `DeadLetterEntry`, `JobStatus`
- `scheduler/jobs.py`: 8 operational tasks + `run_full_cycle()`
  - Job 1: Open-Meteo weather pull (exponential backoff, graceful degradation)
  - Job 2: GEE satellite ingestion (ZeroLeakageValidator enforcement)
  - Job 3: Hydrometeorological feature rebuild
  - Job 4: MultiHazardModel deep learning inference
  - Job 5: Isotonic calibration + dual-threshold hysteresis
  - Job 6: PostGIS persistence (decoupled hazard/exposure)
  - Job 7: Alert dispatch trigger (wired to AlertDispatcher)
  - Job 8: RainViewer radar metadata refresh
- `scheduler/daemon.py`: APSchedulerBackend, JobLock (non-overlapping), dead-letter queue, staleness detection, `_job_funcs` registry
- `scheduler/runner.py`: CLI entrypoint with signal handling
- Admin endpoints in `api/main.py`: `GET /admin/jobs`, `POST /admin/jobs/trigger`, `GET /api/v1/system/staleness`
- `preprocessing/gee_download.py`: Added `validate_observation_timestamp()` to `ZeroLeakageValidator`
- `docker-compose.yml`: Added `scheduler` service
- `test_phase2_scheduler.py`: 6/6 PASSED

### Phase 3: Multi-Channel Alert Dispatch (DONE — 7/7 tests)
- `api/alerts/telegram.py`: Telegram Bot API notifier (free bot API, HTML formatting, NDMA disclaimer, mock fallback)
- `api/alerts/email.py`: SMTP email notifier (responsive HTML dossier email, threat color badges, multi-recipient)
- `api/alerts/sms.py`: TRAI DLT compliant SMS notifier (PE ID, Header `NDMAEW`, template IDs, 160-char GSM-7 limits, zero-cost MockSMSProvider)
- `api/alerts/cap_feed.py`: OASIS CAP v1.2 XML feed generator (urgency, severity, certainty, HydroSHEDS geocodes, Atom feed)
- `api/alerts/approval.py`: Human-in-the-Loop operator approval queue for CRITICAL alerts (`approve_alert`, `reject_alert`, auto-release safety timer)
- `api/alerts/dispatcher.py`: Central alert dispatcher with hysteresis state-transition triggers (`NEW`, `ESCALATED`, `STABLE_SUPPRESSED`, `DE_ESCALATED`), rate-limiting, and multi-channel routing
- Endpoints in `api/main.py`:
  - `GET /api/v1/alerts/cap`: OASIS CAP v1.2 XML feed
  - `GET /api/v1/alerts/pending`: Operator approval queue
  - `POST /api/v1/alerts/approve/{alert_id}`: Operator sign-off
  - `POST /api/v1/alerts/reject/{alert_id}`: Operator rejection of false alarms
  - `POST /api/v1/alerts/dispatch-test`: Multi-channel test broadcast
  - `GET /api/v1/alerts/history`: Dispatched alert audit trail
- `test_phase3_dispatch.py`: 7/7 PASSED

### Phase 4: Kaggle 2x T4 DDP Training, 2016-2023 Manifest & Model Promotion (DONE — 5/5 tests)
- `data/manifest_2016_2023.json`: 8-year continental dataset catalog (CHIRPS, Sentinel-1 SAR, Copernicus DEM, MODIS NDVI, ERA5-Land, CWC discharge) with 3 non-overlapping chronological splits: Train (2016–2021), Validation (2022), Test (2023) enforcing Zero Data Leakage.
- `scripts/download_manifest.py`: Verification utility enforcing zero data leakage causal bounds.
- `training/train_kaggle.py`: Kaggle 2x Tesla T4 DistributedDataParallel (DDP) multi-GPU training script with PyTorch AMP FP16, AdamW, Cosine Annealing, MultiTaskLoss with learned homoscedastic uncertainty weighting, and single-GPU/CPU fallback.
- `evaluation/benchmarks.py`: 6-competitor baseline benchmarking suite (Climatology, Persistence, Logistic/Ridge, Random Forest, XGBoost, MultiHazardModel) generating publication-grade comparison scorecards.
- `scripts/promote_model.py`: Automated model promotion gating candidate checkpoints against CSI >= 0.50, Brier <= 0.18, Drought RMSE <= 0.85, archiving old models, and updating `checkpoints/model_registry.json`.
- `test_phase4_kaggle.py`: 5/5 PASSED

### Pre-existing (Tasks 1–4, verified by test_phase6.py 8/8):
- **PDF Export**: `api/pdf_generator.py`, `GET /api/v1/export/briefing-pdf`, `IncidentBriefingModal.tsx`
- **Open-Meteo**: `api/weather_service.py`, 12 Indian basins, 6h cache, `LiveWeatherForecastWidget.tsx`
- **RainViewer Radar**: Live Doppler loop in `HazardMap.tsx`, play/pause, opacity, dBZ legend
- **Docker**: `Dockerfile.api`, `dashboard/Dockerfile`, `docker-compose.yml`
- **Deep Learning**: `models/encoders.py` (SpatialCNN, TemporalGRU), `models/fusion.py` (GatedFusion), `models/multitask.py` (MultiHazardModel dual heads), `training/` (dataset, loss, trainer)
- **Risk Engine**: `risk_engine/calibration.py`, `thresholding.py`, `hysteresis.py`
- **XAI**: `xai/gradcam.py`, `xai/shap_tabular.py`

## 3. WHAT REMAINS TO BUILD ❌

### Phase 5: Production Hardening (NEXT)
- `api/auth.py`: JWT / API key authentication middleware & RBAC (`viewer`, `operator`, `admin`)
- CORS lock-down configuration in `api/main.py`
- Prometheus metrics endpoint (`/metrics`) exposing request latencies, inference timing, alert dispatch metrics
- Automated database backup script (`scripts/backup_db.py`)
- Real-time model drift monitor (`evaluation/drift_monitor.py`: Kolmogorov-Smirnov / PSI feature distribution shift detection)
- Verification test suite: `test_phase5_hardening.py`

## 4. CRITICAL RULES (NEVER VIOLATE)

1. **Zero Data Leakage**: Every satellite/reanalysis input must be strictly BEFORE forecast issue time `t`. Enforced by `ZeroLeakageValidator` in `preprocessing/gee_download.py`.
2. **Decoupling**: Physical hazard probability (P_flood, SPEI) is NEVER passed population, schools, or cropland. Exposure data lives in separate `exposure` table, queried independently.
3. **Phase-by-Phase**: Complete one phase, run ALL tests, output report, then STOP and wait for user `CONTINUE`.
4. **Test Gates**: After every phase:
   - New phase test must pass (e.g., `test_phase3_dispatch.py`)
   - `test_phase6.py` must still pass 8/8
   - `cd dashboard && npx tsc --noEmit` must have 0 errors
5. **Free Stack Only**: No paid APIs, no credit cards. Telegram Bot API is free. SMTP can use Gmail app passwords (free). SMS is stub-only unless user provides gateway.

## 5. WINDOWS-SPECIFIC GOTCHAS

1. **PyTorch DLL Order**: On Windows Python 3.10, `import torch` MUST come BEFORE `import shapely`, `import geos`, or any geospatial C-extensions. Otherwise: `OSError: [WinError 1114] c10.dll`.
2. **PowerShell `&&`**: Use `Set-Location dir; command` instead of `cd dir && command`.
3. **Sandbox**: Most commands need `BypassSandbox: true` on this Windows setup.
4. **APScheduler 3.x**: `Job.next_run_time` only exists after scheduler starts. Use `getattr(job, "next_run_time", None)`.

## 6. KEY FILE MAP

```
project/
├── configs/config.yaml                    # Master hyperparameters, CRS, bounding boxes
├── preprocessing/
│   ├── gee_download.py                    # GEE connector + ZeroLeakageValidator
│   ├── harmonizer.py                      # EPSG:32644 reprojection, 1km grid
│   └── gap_fill.py                        # SAR interpolation + confidence decay
├── features/
│   ├── rainfall_windows.py                # 3/7/15/30d sums, peak burst, API
│   ├── topography.py                      # Slope, aspect, TWI, curvature
│   ├── hydrology.py                       # Flow-accumulation upstream routing
│   └── spei.py                            # SPEI-30, SPEI-90 standardized anomalies
├── models/
│   ├── baselines.py                       # Climatology, Persistence, LR, RF, XGBoost
│   ├── encoders.py                        # SpatialCNN (3-layer residual) + TemporalGRU (2-layer)
│   ├── fusion.py                          # PlainConcatenation + GatedFusion
│   ├── multitask.py                       # MultiHazardModel (shared trunk + flood/drought heads)
│   └── deep_baseline.py                   # Standalone deep benchmark
├── training/
│   ├── dataset.py                         # PatchDataset, 3-way splits
│   ├── loss.py                            # MultiTaskLoss (Focal + Huber + Kendall)
│   └── trainer.py / train.py             # PyTorch AMP, AdamW, Cosine Annealing
├── risk_engine/
│   ├── calibration.py                     # Isotonic regression calibrator
│   ├── thresholding.py                    # CWC/NDMA tiers
│   └── hysteresis.py                      # Dual-threshold hysteresis
├── xai/
│   ├── gradcam.py                         # Grad-CAM 2D saliency
│   └── shap_tabular.py                    # Tree SHAP attributions
├── scheduler/
│   ├── base.py                            # Abstract interfaces, JobRecord, DeadLetterEntry
│   ├── jobs.py                            # 8 operational tasks + run_full_cycle()
│   ├── daemon.py                          # APSchedulerBackend + JobLock + staleness
│   └── runner.py                          # CLI entrypoint
├── api/
│   ├── main.py                            # FastAPI app, all endpoints
│   ├── weather_service.py                 # Open-Meteo 12-basin integration
│   ├── pdf_generator.py                   # CWC/NDRF incident dossier PDF
│   ├── db.py                              # DB connection pooling
│   └── spatial_ingest.py                  # Multi-format vector loader
├── scripts/
│   ├── migrate.py                         # Migration runner
│   ├── ingest_spatial_data.py             # Basin/station seeder
│   └── gee_india_multihazard.js           # GEE Code Editor script
├── migrations/
│   ├── 001_init_postgis.sql               # UP migration
│   └── 001_down.sql                       # DOWN migration
├── dashboard/
│   ├── src/app/page.tsx                   # Main page (Overview, Map, Warnings, Analytics tabs)
│   └── src/components/
│       ├── HazardMap.tsx                  # Leaflet map, wave scrubber, radar overlay
│       ├── LiveWeatherForecastWidget.tsx  # 7-day weather charts
│       ├── ActiveWarningsPanel.tsx        # Live warning feed
│       ├── LocationDetailsPanel.tsx       # Catchment inspector
│       └── IncidentBriefingModal.tsx      # PDF preview/download modal
├── checkpoints/
│   ├── best_multi_hazard_model.pt         # 8.1 MB
│   └── test_model_gated_multitask.pt      # 11.8 MB
├── Dockerfile.api                         # Python API container
├── docker-compose.yml                     # PostGIS + API + Dashboard + Scheduler
├── .env.example                           # Environment template
├── requirements.txt                       # Python dependencies
├── test_phase1.py                         # Phase 1 tests (6/6)
├── test_phase2_scheduler.py               # Phase 2 tests (6/6)
├── test_phase2.py                         # Deep baseline tests
├── test_phase3.py                         # Encoder/fusion tests
├── test_phase4.py                         # Multitask/gated tests
├── test_phase5.py                         # XAI/calibration tests
├── test_phase6.py                         # Full-stack API tests (8/8) — BASELINE
└── PROJECT_STATUS_AND_HANDOVER.md         # Detailed status doc
```

## 7. MODEL ARCHITECTURE QUICK REFERENCE

```
MultiHazardModel forward() signature:
  model(
    static_features=Tensor[B, 8, 32, 32],     # terrain rasters
    dynamic_images=Tensor[B, 2, 32, 32],       # SAR/rainfall
    climate_sequence=Tensor[B, T, 3],          # temporal meteo (T=12 or 30)
    return_dict=True
  )

Output dict keys:
  flood_logit, flood_prob, spei_continuous, drought_logits, drought_probs,
  z_spatial, z_temporal, z_fused, representation,
  spatial_feature_map, temporal_attention, gate_weights (if gated)
```

## 8. HOW TO VERIFY

```powershell
# Phase 4 Kaggle & benchmarking tests
.\.venv\Scripts\python test_phase4_kaggle.py       # Expect 5/5

# Phase 3 alert dispatch tests
.\.venv\Scripts\python test_phase3_dispatch.py     # Expect 7/7

# Phase 2 scheduler tests
.\.venv\Scripts\python test_phase2_scheduler.py    # Expect 6/6

# Phase 1 PostGIS migration tests
.\.venv\Scripts\python test_phase1.py              # Expect 6/6

# Baseline regression (MUST always pass)
.\.venv\Scripts\python test_phase6.py              # Expect 8/8

# TypeScript check (MUST always pass)
Set-Location dashboard; npx tsc --noEmit           # Expect 0 errors
```

## 9. GEE AUTHENTICATION

- GCP Project: `multi-hazard-eews` (Project Number: 515726658231)
- Free academic/research tier, zero billing
- Credentials: `~/.config/earthengine/credentials`
- Datasets: CHIRPS, Sentinel-1, Copernicus DEM 30m, MODIS NDVI, ERA5-Land

## 10. RESUME COMMAND

To resume, say exactly:
> "Read RESUME_INSTRUCTIONS.md at C:\Users\DELL\.gemini\antigravity-ide\scratch\project\RESUME_INSTRUCTIONS.md and continue building from Phase 5. Follow all rules in that file. Run tests after each phase and STOP for CONTINUE."

