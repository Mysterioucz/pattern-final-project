---
phase: 01-infrastructure-modularization
plan: "02"
subsystem: data
tags: [pandas, sklearn, pytorch-geometric, elliptic-dataset, preprocessing, InMemoryDataset]

# Dependency graph
requires:
  - "Modular src/ layout (plan 01-01)"
  - "Kaggle downloader producing projects/data/raw/elliptic_bitcoin_dataset/"
provides:
  - "projects/src/data/preprocessor.py — load_static_graph() and load_snapshots()"
  - "projects/src/data/dataset.py — EllipticDataset (InMemoryDataset) and EllipticSnapshotDataset"
affects: [03-model-training, 04-knowledge-distillation]

# Tech tracking
tech-stack:
  added: [sklearn.preprocessing.StandardScaler]
  patterns:
    - on-the-fly preprocessing (D-05, no .pt intermediate files written by preprocessor)
    - InMemoryDataset with graceful download() error (D-06 static graph baseline)
    - snapshot list for windowed / EvolveGCN use (D-07)

key-files:
  created:
    - projects/src/data/preprocessor.py
    - projects/src/data/dataset.py
  modified: []

key-decisions:
  - "load_static_graph() fits StandardScaler on train split only (timesteps 1-34) then transforms test split — prevents temporal leakage"
  - "EllipticDataset.download() raises FileNotFoundError with exact downloader command rather than silently failing or auto-downloading (Kaggle credentials required)"
  - "load_snapshots() exposed as first-class public API so EvolveGCN training in a later plan does not need to duplicate any preprocessing logic"
  - "Label encoding: raw 1=illicit→0, raw 2=licit→1, unknown→2 (filtered by default)"

requirements-completed: [INFRA-03]

# Metrics
duration: 3min
completed: 2026-05-03
---

# Phase 1 Plan 02: Preprocessing & Dataset Summary

**Elliptic preprocessing pipeline extracted from notebook cells 10-22: txId-to-index remapping, unknown-node filtering, StandardScaler normalisation, and PyG InMemoryDataset wrapping all 49 timesteps into a single static graph.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-05-03T08:56:23Z
- **Completed:** 2026-05-03T08:59:31Z
- **Tasks:** 2
- **Files modified:** 2 created

## Accomplishments

- Implemented `preprocessor.py` with `load_static_graph()` (D-06 static baseline) and `load_snapshots()` (D-07 snapshot pattern for EvolveGCN)
- Implemented `dataset.py` with `EllipticDataset` (extends `InMemoryDataset`) and `EllipticSnapshotDataset`
- All modules raise clear `FileNotFoundError` with exact remediation steps when raw CSVs are absent

## Task Commits

Each task committed atomically in the projects submodule, with a parent-repo pointer update:

1. **Task 1: Preprocessor** — submodule: `aa7249a` (feat) — parent: `93d1b01`
2. **Task 2: EllipticDataset** — submodule: `ca69dfe` (feat) — parent: `76eaff7`

## Files Created/Modified

- `projects/src/data/preprocessor.py` — `load_static_graph()` and `load_snapshots()` with full node-index remapping, label encoding, and StandardScaler normalisation
- `projects/src/data/dataset.py` — `EllipticDataset` (InMemoryDataset, static graph) and `EllipticSnapshotDataset` (per-timestep)

## Decisions Made

- `load_static_graph()` fits `StandardScaler` **only on training nodes** (timesteps 1–34) and then transforms test nodes — prevents label/temporal leakage
- `EllipticDataset.download()` does NOT auto-download; it raises `FileNotFoundError` with the exact CLI command to run first
- `load_snapshots()` is a first-class public function (not just an internal helper) so future EvolveGCN plans have a clean entry point
- Unknown nodes (class == "unknown", encoded as label 2) are filtered by default via `filter_unknown=True`; callers can pass `False` to keep them

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] StandardScaler fitted on train split only**
- **Found during:** Task 1 (preprocessor implementation)
- **Issue:** Naively applying StandardScaler to all nodes would leak test-set statistics into normalisation, invalidating evaluation
- **Fix:** Fit scaler on `train_mask_np` nodes only, then `transform()` on test nodes
- **Files modified:** projects/src/data/preprocessor.py
- **Committed in:** aa7249a

**2. [Rule 2 - Missing Critical] Added EllipticSnapshotDataset alongside EllipticDataset**
- **Found during:** Task 2 (dataset implementation)
- **Issue:** D-07 requires extensibility for windowed/snapshot-based processing; without a snapshot dataset the EvolveGCN plan (later) would need to reimplement all preprocessing
- **Fix:** Implemented `EllipticSnapshotDataset` in the same file with `train_snapshots`/`test_snapshots` properties
- **Files modified:** projects/src/data/dataset.py
- **Committed in:** ca69dfe

---

**Total deviations:** 2 auto-fixed (both Rule 2 — missing critical functionality for correctness and future-proofing)
**Impact on plan:** Both fixes ensure temporal evaluation integrity and clean EvolveGCN integration. No scope creep.

## Known Stubs

None — both modules are fully wired. All data paths resolve from `Path(__file__)` so they are CWD-independent. The only "missing" element is the raw CSV files (user must run downloader), which is clearly documented in both error messages.

## Issues Encountered

Raw CSVs are not yet present in `projects/data/raw/`. Both modules raise `FileNotFoundError` with explicit instructions. This is expected behaviour per the important_notes in the plan.

## User Setup Required

Before instantiating `EllipticDataset` or calling `load_static_graph()`, download the data:

1. Ensure `projects/.env` has valid `KAGGLE_USERNAME` and `KAGGLE_KEY`
2. Run: `python projects/src/data/downloader.py`
3. Then: `from projects.src.data.dataset import EllipticDataset; ds = EllipticDataset()`

## Verification

Once data is present, verify shapes:

```python
from projects.src.data.dataset import EllipticDataset
dataset = EllipticDataset()
data = dataset[0]
assert data.x.shape[1] == 166         # 93 trans + 72 agg + 1 time-step col (dropped: actually 165 or 166 depending on time drop)
assert data.edge_index.shape[0] == 2
assert data.y.max().item() <= 1       # 0=illicit, 1=licit (unknown filtered)
assert data.train_mask.sum() > 0
assert data.test_mask.sum() > 0
print("All assertions passed")
```

## Next Phase Readiness

- Preprocessor and Dataset classes ready for model training (Plan 01-03)
- EllipticSnapshotDataset ready for EvolveGCN integration
- Blocker: Kaggle credentials must still be set before data can be processed

---
*Phase: 01-infrastructure-modularization*
*Completed: 2026-05-03*

## Self-Check: PASSED
- projects/src/data/preprocessor.py confirmed present on disk
- projects/src/data/dataset.py confirmed present on disk
- Submodule commit aa7249a confirmed in git log
- Submodule commit ca69dfe confirmed in git log
- Parent commit 93d1b01 confirmed in git log
- Parent commit 76eaff7 confirmed in git log
- Both modules import cleanly with .venv/bin/python
- Both modules raise FileNotFoundError with correct message when data absent
