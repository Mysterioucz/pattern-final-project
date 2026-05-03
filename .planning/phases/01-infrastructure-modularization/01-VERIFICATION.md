---
phase: 01-infrastructure-modularization
verified: 2026-05-03T00:00:00Z
status: gaps_found
score: 2/3 must-haves verified
overrides_applied: 0
re_verification: false
gaps:
  - truth: "Dataset can be downloaded and preprocessed via a single script/command"
    status: failed
    reason: "main.py _download() calls non-existent function 'download_dataset' (CR-01) and passes wrong kwarg 'dest' instead of 'dest_dir' (CR-02), causing guaranteed ImportError/TypeError when --download flag is used. Additionally, python-dotenv and kaggle packages are absent from pyproject.toml and the project venv, so downloader.py cannot even be imported."
    artifacts:
      - path: "projects/src/main.py"
        issue: "Line 38: imports 'download_dataset' which does not exist in downloader.py (correct name is 'download_elliptic_dataset'). Line 41: passes kwarg 'dest=' which is not a parameter of download_elliptic_dataset (correct kwarg is 'dest_dir='). Both bugs confirmed by runtime ImportError."
      - path: "projects/src/data/downloader.py"
        issue: "Depends on 'python-dotenv' (import dotenv) which is not installed in the project venv and is absent from pyproject.toml. 'kaggle' package is also missing from project dependencies."
    missing:
      - "Fix main.py line 38: change 'download_dataset' to 'download_elliptic_dataset'"
      - "Fix main.py line 41: change 'dest=' to 'dest_dir='"
      - "Add 'python-dotenv' and 'kaggle' to pyproject.toml dependencies"

  - truth: "Data loaders successfully yield batches for the Elliptic dataset"
    status: partial
    reason: "Loader code is correctly structured and imports work. However, load_snapshots() (used by get_snapshot_loaders) fits a fresh StandardScaler per snapshot including test-split snapshots (CR-03 data leakage). This means snapshot loaders will use test-set statistics for normalizing test data rather than train-set statistics, violating temporal evaluation integrity. load_static_graph() correctly fits only on train nodes, so get_static_split() is unaffected."
    artifacts:
      - path: "projects/src/data/preprocessor.py"
        issue: "Lines 356-357 in load_snapshots(): 'scaler = StandardScaler(); x_np = scaler.fit_transform(x_np)' executes for every timestep including test (t >= 35), leaking test-set feature statistics into normalization. load_static_graph() (lines 275-277) does this correctly by contrast."
    missing:
      - "Fix load_snapshots() to fit scaler once on training snapshots and apply it to test snapshots (mirrors the correct approach in load_static_graph)"
---

# Phase 1: Infrastructure & Modularization — Verification Report

**Phase Goal:** A clean, modular codebase ready for experiments.
**Verified:** 2026-05-03T00:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `projects/src/` contains separate modules for data, models, and training | VERIFIED | Directory structure confirmed: src/data/, src/models/, src/utils/ all exist with __init__.py. preprocessor.py, dataset.py, loader.py, downloader.py, gcn.py, evolve_gcn.py all present and substantive. |
| 2 | Dataset can be downloaded and preprocessed via a single script/command | FAILED | main.py --download crashes at runtime: imports non-existent 'download_dataset' (CR-01), passes wrong kwarg 'dest' (CR-02). python-dotenv/kaggle missing from project deps. |
| 3 | Data loaders successfully yield batches for the Elliptic dataset | PARTIAL | get_static_split() is correctly implemented. get_snapshot_loaders() works structurally but load_snapshots() has CR-03 data leakage: fits StandardScaler per snapshot including test snapshots. |

**Score:** 1/3 truths fully verified (1 partial, 1 failed)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `projects/src/__init__.py` | Package init | VERIFIED | Present, empty (correct) |
| `projects/src/data/__init__.py` | Package init | VERIFIED | Present, empty (correct) |
| `projects/src/models/__init__.py` | Package init | VERIFIED | Present, empty (correct) |
| `projects/src/utils/__init__.py` | Package init | VERIFIED | Present, empty (correct) |
| `projects/data/raw/.gitkeep` | Track empty dir | VERIFIED | Present |
| `projects/data/processed/.gitkeep` | Track empty dir | VERIFIED | Present |
| `projects/.env.example` | Kaggle credential template | VERIFIED | Contains KAGGLE_USERNAME and KAGGLE_KEY placeholders |
| `projects/src/data/downloader.py` | Kaggle download script | VERIFIED (code only) | File is substantive; credentials/dep injection correct. python-dotenv dep not in venv. |
| `projects/src/data/preprocessor.py` | Preprocessing pipeline | PARTIAL | load_static_graph() correct; load_snapshots() has data leakage bug (CR-03) |
| `projects/src/data/dataset.py` | PyG InMemoryDataset | VERIFIED | EllipticDataset extends InMemoryDataset; process() calls load_static_graph(); EllipticSnapshotDataset also implemented |
| `projects/src/data/loader.py` | Train/test DataLoaders | PARTIAL | Structure correct; inherits CR-03 from load_snapshots |
| `projects/src/main.py` | CLI entry point | FAILED | --download flag broken (CR-01 + CR-02); --mode static/snapshot wiring correct |
| `projects/src/models/gcn.py` | GCN stub (Phase 2) | VERIFIED | Intentional stub per plan; raises NotImplementedError with Phase 2 reference |
| `projects/src/models/evolve_gcn.py` | EvolveGCN stub (Phase 2) | VERIFIED | Intentional stub per plan; raises NotImplementedError with Phase 2 reference |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| main.py | downloader.py | `from data.downloader import download_dataset` | NOT_WIRED | Function name wrong: should be `download_elliptic_dataset`. Confirmed by runtime ImportError. |
| main.py | downloader.py | `download_dataset(dest=...)` | NOT_WIRED | Kwarg name wrong: should be `dest_dir=`. Would be TypeError even if import succeeded. |
| main.py | preprocessor.py | `from data.preprocessor import load_static_graph` | WIRED | Correct import, correct call in `_print_static_stats()` |
| main.py | loader.py | `from data.loader import get_snapshot_loaders` | WIRED | Correct import, correct call in `_print_snapshot_stats()` |
| dataset.py | preprocessor.py | `from .preprocessor import load_static_graph, load_snapshots` | WIRED | Relative imports confirmed working |
| loader.py | preprocessor.py | `from .preprocessor import load_snapshots, load_static_graph, TEST_START_TIMESTEP` | WIRED | Relative imports confirmed working |
| get_snapshot_loaders | load_snapshots | per-snapshot scaler | PARTIAL | Calls load_snapshots() correctly; load_snapshots itself fits scaler per-snapshot including test (CR-03) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `preprocessor.load_static_graph` | x, y, edge_index, train_mask, test_mask | Reads 3 Elliptic CSVs via pandas | Yes (requires CSV data) | VERIFIED (logic correct, data absent by design) |
| `preprocessor.load_snapshots` | per-snapshot x, y, edge_index | Same CSVs, per-timestep loop | Yes (requires CSV data) | PARTIAL — scaler leaks test stats (CR-03) |
| `dataset.EllipticDataset.process` | Data object | Delegates to load_static_graph | Yes, via preprocessor | VERIFIED |
| `loader.get_snapshot_loaders` | train_loader, test_loader | Delegates to load_snapshots | Yes, via preprocessor | PARTIAL — inherits CR-03 |
| `loader.get_static_split` | Data object | Delegates to load_static_graph | Yes, via preprocessor | VERIFIED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All modules import without errors | `venv/python -c "import sys; sys.path.insert(0,'projects/src'); from data.preprocessor import load_static_graph; ..."` | All imports OK | PASS |
| main.py --help works | `venv/python projects/src/main.py --help` | Correct argparse output with --download and --mode flags | PASS |
| main.py --download crashes | `_download()` call: `from data.downloader import download_dataset` | ImportError: cannot import name 'download_dataset' | FAIL (confirms CR-01) |
| GCN stub raises NotImplementedError | `gcn.forward(None)` | NotImplementedError with Phase 2 reference | PASS |
| EvolveGCN stub raises NotImplementedError | `evolve.forward([])` | NotImplementedError with Phase 2 reference | PASS |
| download_elliptic_dataset signature | inspect.signature | `(dest_dir, force)` — correct params defined | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| INFRA-01 | 01-01 | Extract notebook logic into lean .py files in projects/src/ | SATISFIED | All modules exist: downloader.py, preprocessor.py, dataset.py, loader.py, main.py, gcn.py, evolve_gcn.py — all substantive, not stubs |
| INFRA-02 | 01-01 | Implement Kaggle dataset download and extraction script | PARTIAL | downloader.py exists and logic is correct; broken at invocation level (main.py wires to wrong function name/kwarg; python-dotenv/kaggle missing from deps) |
| INFRA-03 | 01-02 | Create data preprocessing pipeline for the Elliptic dataset | PARTIAL | load_static_graph() is correctly implemented with train-only scaler fit. load_snapshots() has CR-03 data leakage. |
| INFRA-04 | 01-03 | Implement GNN data loaders for training and testing splits | PARTIAL | get_static_split() is correct. get_snapshot_loaders() structurally correct but inherits CR-03 from load_snapshots. 34/15 split logic verified correct. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| projects/src/main.py | 38 | `import download_dataset` — function does not exist | BLOCKER | --download flag always crashes with ImportError |
| projects/src/main.py | 41 | `download_dataset(dest=...)` — wrong kwarg name | BLOCKER | Would TypeError even after fixing CR-01 |
| projects/src/data/preprocessor.py | 356-357 | `scaler.fit_transform(x_np)` inside per-snapshot loop including test snapshots | BLOCKER | Test-set feature statistics leak into normalization; invalidates evaluation |
| pyproject.toml | — | python-dotenv and kaggle absent from dependencies | WARNING | downloader.py imports dotenv at module level; cannot be installed/run without these |
| projects/src/data/loader.py | 19 | `from torch_geometric.data import DataLoader` (deprecated) | WARNING | Currently still works in PyG 2.7.0 but will break in future versions |
| projects/src/data/dataset.py | 79 | `torch.load(..., weights_only=False)` | WARNING | Security: arbitrary code execution if processed .pt file is tampered |
| projects/src/data/preprocessor.py | 108 | `.rename(columns={"index": "index"})` | INFO | No-op dead code |

### Human Verification Required

No human verification items. All gaps are verifiable programmatically.

### Gaps Summary

Three blockers prevent full goal achievement:

**1. Broken --download CLI path (CR-01 + CR-02)** — Success Criterion 2 ("Dataset can be downloaded and preprocessed via a single script/command") fails because main.py's `_download()` function contains two compounding bugs: it imports a function by the wrong name (`download_dataset` instead of `download_elliptic_dataset`) and passes a wrong keyword argument (`dest=` instead of `dest_dir=`). Runtime testing confirmed `ImportError` on the import line. Additionally, `python-dotenv` and `kaggle` are absent from `pyproject.toml` and the project venv, so `downloader.py` itself cannot be imported in a clean environment.

**2. Data leakage in snapshot normalization (CR-03)** — `load_snapshots()` in `preprocessor.py` fits a new `StandardScaler` independently per snapshot, which means test-set snapshots are normalized using statistics derived from test data, not the training distribution. This is the opposite of what `load_static_graph()` correctly does (fit on train, transform test). Any experiment using `get_snapshot_loaders()` will report inflated/invalid metrics.

**Partially mitigated:** Success Criterion 1 (modular src/ layout) is fully verified. Success Criterion 3 (data loaders) is partially met — the static-graph path (`get_static_split`) works correctly; only the snapshot path is affected by CR-03.

The two gaps share no common root cause. CR-01/CR-02 are copy errors in main.py. CR-03 is an algorithmic error in preprocessor.py.

---

_Verified: 2026-05-03T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
