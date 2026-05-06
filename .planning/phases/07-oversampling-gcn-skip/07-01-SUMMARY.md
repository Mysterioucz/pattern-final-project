---
phase: 07-oversampling-gcn-skip
plan: "01"
subsystem: data
tags: [sampling, imbalanced-learn, torch-geometric]
requires:
  - phase: 07-00
    provides: "RED test contracts for sampler behavior"
provides:
  - "Feature-space per-timestep sampling transform with induced-subgraph reconstruction"
  - "Factory support for ENN, edge_pruning, cluster_centroid, nearmiss1, nearmiss2"
affects: [07-02, training, evaluation]
tech-stack:
  added: []
  patterns: ["labeled-only sampling + unknown passthrough + relabeled induced subgraph"]
key-files:
  created: [projects/src/data/sampler.py]
  modified: []
key-decisions:
  - "Use edge_pruning as custom path sentinel via get_sampler('edge_pruning') -> None."
  - "Clamp neighbor settings for ENN/NearMiss using minority count to avoid sparse-timestep failures."
patterns-established:
  - "Always map sample_indices_ back to global indices before subgraph rebuild."
requirements-completed: [D-03, D-04, D-05, D-06, D-07]
duration: 12min
completed: 2026-05-06
---

# Phase 07 Plan 01: Sampler Implementation Summary

**Sampler module now provides five undersampling modes with unknown-node preservation and PyG-safe induced-subgraph reconstruction, turning the full sampler contract suite GREEN.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-05-06T12:50:00Z
- **Completed:** 2026-05-06T13:02:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Implemented `get_sampler(name)` and `apply_sampling(data, sampler_name)`.
- Added custom edge-pruning rule plus ClusterCentroids row-match recovery helper.
- Passed all sampler contract tests (`10 passed`) and import/type smoke checks.

## Task Commits

1. **Task 1: Implement projects/src/data/sampler.py** - `adfd3e1` (feat, submodule `projects`)
2. **Parent pointer sync** - `8380f39` (chore, parent repo)

## Files Created/Modified
- `projects/src/data/sampler.py` - Sampler factory and transform implementation.

## Decisions Made
- Retained explicit unknown-node pass-through (`y == 2`) across all methods.
- Used `subgraph(..., relabel_nodes=True)` to guarantee valid compact node indexing.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Committed inside `projects` submodule**
- **Found during:** Task 1 commit stage
- **Issue:** Parent repository cannot directly stage files under `projects/` because it is a git submodule.
- **Fix:** Committed code within submodule, then committed parent submodule pointer.
- **Files modified:** `projects/src/data/sampler.py`, `projects` (submodule pointer)
- **Committed in:** `adfd3e1`, `8380f39`

---

**Total deviations:** 1 auto-fixed (Rule 3)
**Impact on plan:** No functional scope change; required for repository correctness.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Training script can now import `apply_sampling`.
- Wave 2 ablation execution is unblocked.

## Self-Check: PASSED
- FOUND: projects/src/data/sampler.py
- FOUND: adfd3e1
- FOUND: 8380f39
