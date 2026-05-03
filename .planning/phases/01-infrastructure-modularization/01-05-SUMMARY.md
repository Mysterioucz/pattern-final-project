---
phase: 01-infrastructure-modularization
plan: "05"
subsystem: data-pipeline
tags: [data-leakage, preprocessing, scaler, dataloader, gap-closure]
dependency_graph:
  requires: [01-02, 01-03]
  provides: [INFRA-03-closed, INFRA-04-resolved]
  affects: [preprocessor.py, loader.py]
tech_stack:
  added: []
  patterns: [train-only-scaler-fit, transform-all-snapshots]
key_files:
  created: []
  modified:
    - projects/src/data/preprocessor.py
    - projects/src/data/loader.py
decisions:
  - "StandardScaler in load_snapshots() now fitted once on pooled training rows (t < 35) before the per-snapshot loop, mirroring the existing load_static_graph() pattern"
  - "scaler.transform() used for every snapshot (train and test); no per-snapshot fit"
  - "DataLoader imported from torch_geometric.loader (canonical PyG 2.x path)"
metrics:
  duration: "2 min"
  completed: "2026-05-03"
  tasks_completed: 2
  files_modified: 2
---

# Phase 1 Plan 05: Gap Closure — Data Leakage & Deprecated Import Summary

**One-liner:** Eliminated StandardScaler data leakage in `load_snapshots()` by fitting once on train rows (t < 35) then calling `transform()` per snapshot; updated `DataLoader` import to `torch_geometric.loader`; corrected docstring feature count from 166 to 165.

## What Was Built

Three surgical fixes to close the remaining Phase 1 gaps identified in `01-VERIFICATION.md`:

1. **CR-03 (BLOCKER) fixed** — `load_snapshots()` in `preprocessor.py` now fits a single `StandardScaler` on all pooled training rows (timestep < 35) before the per-snapshot loop. Every snapshot (train and test) is then normalized using `scaler.transform()`. Previously, a new scaler was fit inside the loop for each timestep, meaning test snapshots were normalized using their own statistics — invalidating temporal evaluation.

2. **WR-04 fixed** — `loader.py` now imports `DataLoader` from `torch_geometric.loader` instead of the deprecated `torch_geometric.data`. `Data` remains imported from `torch_geometric.data` (correct).

3. **IN-03 fixed** — `load_snapshots()` docstring corrected: `x` shape is `(N_t, 165)` not `(N_t, 166)`.

## Commits

| Task | Description | Submodule Commit | Parent Commit |
|------|-------------|-----------------|---------------|
| 1 | Fix data leakage in load_snapshots() + docstring | 29183fa | 06c352c |
| 2 | Update DataLoader import to torch_geometric.loader | 413b91c | 609d2c2 |

## Verification Results

All three plan-level checks passed:
- `PASS: load_snapshots() scaler leakage fixed`
- `PASS: loader.py uses non-deprecated DataLoader import`
- `PASS: both modules import without errors`

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — no stub patterns introduced or remaining in modified files.

## Threat Flags

The T-01-05-01 threat (Information Disclosure via StandardScaler trained on test data) is now fully mitigated by this plan. No new security-relevant surface introduced.

## Self-Check: PASSED

- [FOUND] projects/src/data/preprocessor.py — modified as intended
- [FOUND] projects/src/data/loader.py — modified as intended
- [FOUND] commit 06c352c — Task 1 parent repo commit
- [FOUND] commit 609d2c2 — Task 2 parent repo commit
- INFRA-03 gap CR-03 fully closed
- INFRA-04 partial status resolved (WR-04 fixed)
