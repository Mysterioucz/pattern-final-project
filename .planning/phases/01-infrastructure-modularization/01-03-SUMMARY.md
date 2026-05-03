---
phase: 01-infrastructure-modularization
plan: "03"
subsystem: data
tags: [pytorch-geometric, dataloader, snapshot-split, static-graph, gcn-stub, evolve-gcn-stub]

# Dependency graph
requires:
  - "projects/src/data/preprocessor.py — load_static_graph() and load_snapshots() (plan 01-02)"
  - "projects/src/data/dataset.py — EllipticDataset (plan 01-02)"
provides:
  - "projects/src/data/loader.py — get_snapshot_loaders() and get_static_split()"
  - "projects/src/main.py — CLI entry point with --download, --mode static/snapshot"
  - "projects/src/models/gcn.py — GCN stub with forward() and embed() interface"
  - "projects/src/models/evolve_gcn.py — EvolveGCN stub with forward() interface"
affects: [02-model-training, 04-knowledge-distillation]

# Tech tracking
tech-stack:
  added: [torch_geometric.data.DataLoader, torch_geometric.utils.to_undirected, argparse]
  patterns:
    - snapshot DataLoader with per-timestep Data objects (34 train / 15 test)
    - train_mask / test_mask fields on Data objects per PyG convention
    - argparse CLI with --download flag and --mode selector
    - NotImplementedError stubs for Phase 2 model implementations

key-files:
  created:
    - projects/src/data/loader.py
    - projects/src/main.py
    - projects/src/models/gcn.py
    - projects/src/models/evolve_gcn.py
  modified: []

key-decisions:
  - "get_snapshot_loaders() applies to_undirected() to match notebook cell 23 exactly"
  - "Data.train_mask / Data.test_mask used (not a separate 'mask' field) so training loops need no special-casing"
  - "main.py uses sys.path.insert to be runnable as 'python projects/src/main.py' from repo root without install"
  - "Model stubs raise NotImplementedError with explicit Phase 2 plan reference — avoids silent wrong-behaviour"

patterns-established:
  - "Pattern 4: DataLoader wraps snapshot list with batch_size=1 and shuffle=False for test"
  - "Pattern 5: CLI scripts use argparse --mode flag for selecting static vs. snapshot pipeline"
  - "Pattern 6: Model stubs define __init__ fully but raise NotImplementedError in forward/embed with clear message"

requirements-completed: [INFRA-04]

# Metrics
duration: 4min
completed: 2026-05-03
---

# Phase 1 Plan 03: Data Loaders & Split Summary

**Snapshot DataLoaders (34/15 timestep split) and static-graph Data object wired to preprocessor, plus main.py CLI entry point and GCN/EvolveGCN placeholder stubs for Phase 2.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-03T09:03:24Z
- **Completed:** 2026-05-03T09:06:53Z
- **Tasks:** 3
- **Files modified:** 4 created

## Accomplishments

- Implemented `loader.py` with `get_snapshot_loaders()` (per-timestep DataLoaders, mirrors notebook cell 23) and `get_static_split()` (full-batch Data object for baseline GCN training)
- Created `main.py` entry point that orchestrates download, loading, and dataset statistics print-out with `--download` and `--mode` flags
- Created `gcn.py` and `evolve_gcn.py` placeholder stubs that define the full Phase 2 interface and raise `NotImplementedError` with explicit plan references

## Task Commits

Each task was committed atomically in the projects submodule, with a parent-repo pointer update after each:

1. **Task 1: Data split logic (loader.py)** — submodule: `b23b828` (feat) — parent: `3479a82`
2. **Task 2: Main entry point (main.py)** — submodule: `d566336` (feat) — parent: `38563c9`
3. **Task 3: Model placeholder stubs** — submodule: `fa09c90` (chore) — parent: `6a62929`

## Files Created/Modified

- `projects/src/data/loader.py` — `get_snapshot_loaders()` returns train/test DataLoaders (batch_size=1, to_undirected edges); `get_static_split()` returns full-batch Data with train/test masks
- `projects/src/main.py` — CLI entry point; prints nodes, edges, illicit ratios; --download triggers Kaggle downloader; --mode selects static vs. snapshot pipeline
- `projects/src/models/gcn.py` — GCN stub: `__init__`, `forward()`, `embed()` raising NotImplementedError
- `projects/src/models/evolve_gcn.py` — EvolveGCN stub: `__init__`, `forward(snapshots)` raising NotImplementedError

## Decisions Made

- `get_snapshot_loaders()` calls `to_undirected()` on each snapshot's edge_index to match the notebook cell 23 exactly — ensures edge count parity with original training runs
- Used `Data.train_mask` / `Data.test_mask` (not a generic `mask` field) so that training loops can use the same attribute name for both train and test phases without extra branching
- `main.py` uses `sys.path.insert(0, str(_SRC_DIR))` to be runnable as `python projects/src/main.py` from the repo root without requiring a package install
- Model stubs store all constructor arguments (`num_node_features`, `hidden_channels`, `rnn_type`) so Phase 2 can extend them without changing the `__init__` signature

## Deviations from Plan

None — plan executed exactly as written. All three tasks completed within acceptance criteria.

## Known Stubs

The following stubs are intentional placeholders for Phase 2:

| File | Stub | Reason |
|------|------|--------|
| `projects/src/models/gcn.py` | `GCN.forward()` → NotImplementedError | Phase 2 Plan 02-01 |
| `projects/src/models/gcn.py` | `GCN.embed()` → NotImplementedError | Phase 2 Plan 02-01 |
| `projects/src/models/evolve_gcn.py` | `EvolveGCN.forward()` → NotImplementedError | Phase 2 Plan 02-02 |

These stubs do NOT prevent the plan's goal (data loading infrastructure) from being achieved. The model implementations are explicitly scoped to Phase 2.

## Issues Encountered

Raw CSVs are not present in `projects/data/raw/` (expected — user must run downloader). Both `main.py --mode static` and `main.py --mode snapshot` would raise `FileNotFoundError` with clear instructions. The `--help` flag and import chain work correctly without data.

## User Setup Required

Before running `main.py` to print statistics:

1. Ensure `projects/.env` has valid `KAGGLE_USERNAME` and `KAGGLE_KEY`
2. Run: `python projects/src/main.py --download`
3. Then: `python projects/src/main.py` (static stats) or `python projects/src/main.py --mode snapshot`

## Next Phase Readiness

- Full data pipeline complete: download → preprocess → split → load
- `get_snapshot_loaders()` and `get_static_split()` ready for Phase 2 model training
- GCN and EvolveGCN stubs define the Phase 2 interface; no import changes needed
- Blocker: Kaggle credentials must still be set in `projects/.env` before data flows through the pipeline

---
*Phase: 01-infrastructure-modularization*
*Completed: 2026-05-03*
