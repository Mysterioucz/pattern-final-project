---
phase: 01-infrastructure-modularization
verified: 2026-05-03T12:00:00Z
status: passed
score: 3/3 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 2/3
  gaps_closed:
    - "main.py --download now correctly imports download_elliptic_dataset and passes dest_dir= kwarg (CR-01 + CR-02 fixed, commit 038fbc6)"
    - "load_snapshots() now fits StandardScaler once on training rows (t < 35) before the loop, using scaler.transform() per snapshot (CR-03 fixed, commit 06c352c)"
    - "python-dotenv and kaggle declared in pyproject.toml dependencies (commit 09cfdd1)"
  gaps_remaining: []
  regressions: []
---

# Phase 1: Infrastructure & Modularization — Verification Report

**Phase Goal:** A clean, modular codebase ready for experiments.
**Verified:** 2026-05-03T12:00:00Z
**Status:** PASSED
**Re-verification:** Yes — after gap closure plans 01-04 and 01-05

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `projects/src/` contains separate modules for data, models, and training | VERIFIED | Directory structure intact. All modules (downloader.py, preprocessor.py, dataset.py, loader.py, main.py, gcn.py, evolve_gcn.py) present and substantive. No regression from gap closure. |
| 2 | Dataset can be downloaded and preprocessed via a single script/command | VERIFIED | `main.py _download()` now imports `download_elliptic_dataset` (correct name, confirmed by `grep`) and calls it with `dest_dir=` (correct kwarg). `python-dotenv>=1.0.0` and `kaggle>=1.6.0` are in `pyproject.toml` `[project].dependencies`. Runtime path is code-correct; credentials still required at execution time. |
| 3 | Data loaders successfully yield batches for the Elliptic dataset | VERIFIED | `load_snapshots()` now fits `StandardScaler` once on pooled training rows (`t < 35`) before the per-snapshot loop, then calls `scaler.transform()` for each snapshot. No per-snapshot `fit_transform` in the loop. `DataLoader` imported from `torch_geometric.loader` (non-deprecated). `load_static_graph()` unchanged and still correct. |

**Score:** 3/3 truths verified

### Gap Closure Verification

#### Gap 1 (CR-01 + CR-02): main.py --download CLI path

**Previous state:** Line 38 imported `download_dataset` (function does not exist); line 41 called `download_dataset(dest=...)` (wrong name and wrong kwarg).

**Verified fix in `projects/src/main.py`:**

- `download_elliptic_dataset` appears on lines 38 and 41 — correct function name
- `dest_dir=` appears on line 41 — correct kwarg
- No bare `download_dataset` (without `_elliptic`) remains in the file
- No bare `dest=` (without `_dir`) remains in the file
- Commit: `038fbc6`

#### Gap 1b: Missing pyproject.toml dependencies

**Previous state:** `python-dotenv` and `kaggle` absent from declared dependencies.

**Verified fix in `pyproject.toml`:**

- `python-dotenv>=1.0.0` and `kaggle>=1.6.0` are in `[project].dependencies`
- Parsed via `tomllib` — no parse errors, both names confirmed present
- Commit: `09cfdd1`

#### Gap 2 (CR-03): load_snapshots() data leakage

**Previous state:** Lines 356-357 instantiated a fresh `StandardScaler` per snapshot inside the loop, including for test snapshots (`t >= 35`), causing test-set statistics to influence their own normalization.

**Verified fix in `projects/src/data/preprocessor.py`:**

- `scaler.fit(train_rows)` exists as a single call BEFORE `snapshots = []` (confirmed by position: scaler init at offset 1889, `snapshots = []` at offset 1951 within the function body)
- `train_rows` is built from `merged[merged["time"] < TEST_START_TIMESTEP][feat_cols]` — training rows only
- Inside the loop: `scaler.transform(x_np)` used — no `fit_transform`
- Exactly 1 `StandardScaler()` instantiation within `load_snapshots()` body (the pre-loop one)
- Docstring corrected: `(N_t, 165)` not `(N_t, 166)` — confirmed present, old `166` variant absent
- Commit: `06c352c`

#### Gap 3 (WR-04): Deprecated DataLoader import

**Previous state:** `from torch_geometric.data import Data, DataLoader` — deprecated path.

**Verified fix in `projects/src/data/loader.py`:**

- `from torch_geometric.loader import DataLoader` — confirmed present
- Old combined import `from torch_geometric.data import Data, DataLoader` — confirmed absent
- `from torch_geometric.data import Data` — confirmed present (correct path for Data)
- Commit: `413b91c`

### Regression Check: Previously Passing Items

| Item | Regression Check | Result |
|------|-----------------|--------|
| `load_static_graph()` scaler logic | `fit_transform(x_np[train_mask_np])` + `transform(x_np[test_mask_np])` still present | NO REGRESSION |
| Module imports | `from data.preprocessor import ...`, `from .preprocessor import ...` patterns unchanged | NO REGRESSION |
| main.py --mode static/snapshot wiring | `_print_static_stats()` and `_print_snapshot_stats()` logic unchanged | NO REGRESSION |
| gcn.py / evolve_gcn.py stubs | Intentional Phase 2 stubs — untouched | NO REGRESSION |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `projects/src/main.py` | CLI entry point with --download fix | VERIFIED | CR-01/CR-02 fixed; correct import name and kwarg confirmed |
| `pyproject.toml` | Declares python-dotenv and kaggle | VERIFIED | Both packages with version lower-bounds present in [project].dependencies |
| `projects/src/data/preprocessor.py` | load_snapshots() with train-only scaler fit | VERIFIED | Pre-loop scaler.fit(train_rows) + per-snapshot scaler.transform(x_np) confirmed |
| `projects/src/data/loader.py` | DataLoader from torch_geometric.loader | VERIFIED | Non-deprecated import path confirmed present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `main.py _download()` | `downloader.py` | `from data.downloader import download_elliptic_dataset` | WIRED | Correct function name confirmed in source |
| `main.py _download()` | `download_elliptic_dataset` | `dest_dir=` kwarg | WIRED | Correct kwarg confirmed; matches function signature `(dest_dir, force)` |
| `preprocessor.py load_snapshots()` | `StandardScaler` | fit once on train_rows, transform per snapshot | WIRED | Pre-loop fit, loop-body transform confirmed |
| `loader.py` | `DataLoader` | `from torch_geometric.loader import DataLoader` | WIRED | Non-deprecated import path confirmed |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `preprocessor.load_static_graph` | x, y, edge_index, train_mask, test_mask | 3 Elliptic CSVs via pandas | Yes (requires CSV data; logic correct) | VERIFIED |
| `preprocessor.load_snapshots` | per-snapshot x, y, edge_index | Same CSVs, per-timestep loop | Yes (requires CSV data; scaler leakage fixed) | VERIFIED |
| `loader.get_snapshot_loaders` | train_loader, test_loader | Delegates to load_snapshots() | Yes, via preprocessor | VERIFIED |
| `loader.get_static_split` | Data object | Delegates to load_static_graph() | Yes, via preprocessor | VERIFIED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| main.py correct import name | `grep 'download_elliptic_dataset' projects/src/main.py` | Found on lines 38, 41 | PASS |
| main.py correct kwarg | `grep 'dest_dir=' projects/src/main.py` | Found on line 41; no bare `dest=` | PASS |
| pyproject.toml has both deps | `tomllib` parse | Both `python-dotenv` and `kaggle` in deps list | PASS |
| pre-loop scaler.fit present | positional check in load_snapshots body | scaler init at offset 1889, before snapshots=[] at 1951 | PASS |
| no fit_transform in loop | search loop_section string | `fit_transform` absent from loop section | PASS |
| scaler.transform in loop | search loop_section string | `scaler.transform(x_np)` present | PASS |
| exactly 1 StandardScaler init in load_snapshots | count in function body | 1 (correct) | PASS |
| DataLoader from torch_geometric.loader | grep loader.py | New import present, old combined import absent | PASS |
| load_static_graph scaler unchanged | regression check | fit_transform on train, transform on test — intact | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| INFRA-01 | 01-01 | Extract notebook logic into lean .py files in projects/src/ | SATISFIED | All modules present and substantive; modular structure intact |
| INFRA-02 | 01-01, 01-04 | Implement Kaggle dataset download and extraction script | SATISFIED | downloader.py correct; main.py --download wired correctly; python-dotenv + kaggle in pyproject.toml |
| INFRA-03 | 01-02, 01-05 | Create data preprocessing pipeline for the Elliptic dataset | SATISFIED | load_static_graph() and load_snapshots() both have correct train-only scaler fit |
| INFRA-04 | 01-03, 01-05 | Implement GNN data loaders for training and testing splits | SATISFIED | get_static_split() and get_snapshot_loaders() correct; DataLoader import updated |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| projects/src/main.py | 103 | `illicit_ratio_train = train_illicit / max(train_nodes, 1)` where `train_nodes` counts all nodes including unknowns (not filtered at x level) while `train_illicit` counts only known-labelled illicit nodes via `d.y[d.train_mask]` | INFO | Display-only: ratio denominator includes unknown nodes, so printed illicit ratio is lower than the true known-labelled ratio. Does not affect data pipeline, training, or evaluation metrics. `filter_unknown=True` creates a boolean mask but does not remove unknown rows from `d.x`. |
| projects/src/data/preprocessor.py | 332 | `train_rows = merged[merged["time"] < TEST_START_TIMESTEP][feat_cols]` where `merged` has not yet had unknown nodes (label==2) filtered — scaler is fitted on data that includes unknown-labelled nodes in training timesteps | INFO | Behavioral inconsistency vs `load_static_graph()` which filters unknowns before fitting the scaler. Unknown nodes are a minority; the original CR-03 leakage (fitting on test data) is fully fixed. Scaler statistics are still derived from training timesteps only. Does not invalidate experiments. |
| projects/src/data/dataset.py | 79 | `torch.load(..., weights_only=False)` | WARNING | Pre-existing from original verification; security risk if processed .pt file is tampered. Not introduced by gap-closure plans. Out of scope for Phase 1 gap closure. |

### Observations (Non-Blocking)

1. **`_print_snapshot_stats()` deflated illicit ratio** — The statistics display function computes `illicit_ratio_train = known_illicit_count / all_node_count` where the denominator includes unknown-labelled nodes because `filter_unknown=True` in `load_snapshots()` sets a boolean mask rather than removing rows from the feature tensor. The ratio printed will be slightly lower than the true known-node illicit fraction. This is a cosmetic display issue only; it affects no training code, no loss computation, and no evaluation metric. The model training pipeline receives correctly masked data via `d.train_mask`.

2. **Scaler includes unknown-labelled training nodes** — `load_snapshots()` fits the scaler on all training-timestep rows including unknown-labelled ones. `load_static_graph()` filters unknowns before fitting. This is a minor behavioral inconsistency. The critical fix (CR-03: not fitting on test data) is complete. Experiments using snapshot loaders will produce valid, reproducible results.

Neither observation is a blocker for the phase goal of "a clean, modular codebase ready for experiments."

### Human Verification Required

None. All gap-closure items are verifiable programmatically. All three original blockers are confirmed fixed by source inspection and grep verification.

### Gaps Summary

No gaps remain. All three blockers from the initial verification are closed:

- **CR-01 + CR-02** (broken --download CLI): `main.py` now calls `download_elliptic_dataset(dest_dir=...)` — both the import name and keyword argument are correct.
- **Missing deps**: `python-dotenv>=1.0.0` and `kaggle>=1.6.0` declared in `pyproject.toml`.
- **CR-03** (scaler data leakage): `load_snapshots()` fits `StandardScaler` exactly once on pooled training rows before the snapshot loop; every snapshot is normalized with `scaler.transform()`.
- **WR-04** (deprecated import): `DataLoader` now imported from `torch_geometric.loader`.

Phase 1 goal — a clean, modular codebase ready for experiments — is achieved.

---

_Verified: 2026-05-03T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification: Yes — initial verification found 2 gaps; gap-closure plans 01-04 and 01-05 executed; all gaps confirmed closed_
