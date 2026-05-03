---
phase: "01-infrastructure-modularization"
plan: "04"
subsystem: "data-pipeline"
tags: [gap-closure, bug-fix, dependencies, cli]
dependency_graph:
  requires: []
  provides: [working-download-cli-path, complete-pyproject-deps]
  affects: [projects/src/main.py, pyproject.toml]
tech_stack:
  added: [python-dotenv>=1.0.0, kaggle>=1.6.0]
  patterns: [surgical-minimal-edit, submodule-commit-first]
key_files:
  created: []
  modified:
    - projects/src/main.py
    - pyproject.toml
decisions:
  - "python-dotenv 1.0.0 lower bound: stable load_dotenv() API matching existing downloader.py usage"
  - "kaggle 1.6.0 lower bound: supports Kaggle API v1 used in downloader.py"
metrics:
  duration: "1 min"
  completed: "2026-05-03"
  tasks_completed: 2
  tasks_total: 2
---

# Phase 1 Plan 04: Fix _download() CLI path and declare missing dependencies

**One-liner:** Fixed two bugs in main.py `_download()` (wrong import name CR-01, wrong kwarg CR-02) and added python-dotenv/kaggle to pyproject.toml to close INFRA-02 gaps.

## What Was Built

Two surgical edits closed the three issues identified in `01-VERIFICATION.md`:

1. **CR-01 (main.py line 38):** Changed `from data.downloader import download_dataset` to `from data.downloader import download_elliptic_dataset`. The function `download_dataset` never existed in `downloader.py`; the correct name is `download_elliptic_dataset`.

2. **CR-02 (main.py line 41):** Changed `download_dataset(dest=...)` to `download_elliptic_dataset(dest_dir=...)`. The function signature is `download_elliptic_dataset(dest_dir, force)` — passing `dest=` would have caused a `TypeError` even after fixing CR-01.

3. **Missing deps:** Added `python-dotenv>=1.0.0` and `kaggle>=1.6.0` to `[project].dependencies` in `pyproject.toml`. Both are imported at module level in `downloader.py` and were absent from the project's declared dependency list.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Fix _download() in main.py (CR-01 + CR-02) | 038fbc6 | projects/src/main.py |
| 2 | Add python-dotenv and kaggle to pyproject.toml | 09cfdd1 | pyproject.toml |

## Verification Results

All plan verification checks passed:

```
main.py: import name and kwarg are correct
pyproject.toml: dependencies present
```

Acceptance criteria confirmed:
- `grep -n "download_elliptic_dataset" projects/src/main.py` returns lines 38 and 41
- No bare `download_dataset` (without `_elliptic`) remains in the file
- `dest_dir=` appears on line 41; no bare `dest=` exists
- `tomllib` parse of `pyproject.toml` lists both `python-dotenv` and `kaggle` without error

## Deviations from Plan

None — plan executed exactly as written. Both edits were minimal and surgical. No unrelated code was touched.

## Known Stubs

None introduced by this plan. Model stubs in `gcn.py` and `evolve_gcn.py` are pre-existing and intentional (documented in 01-03-SUMMARY.md).

## Threat Flags

None. The two changes fix import wiring and declare dependencies — no new network endpoints, auth paths, or schema changes were introduced.

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| projects/src/main.py exists | FOUND |
| pyproject.toml exists | FOUND |
| Commit 038fbc6 (Task 1) | FOUND |
| Commit 09cfdd1 (Task 2) | FOUND |
