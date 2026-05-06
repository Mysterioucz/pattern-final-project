---
phase: 7
slug: oversampling-gcn-skip
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-06
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (NOT yet installed — Wave 0 installs) |
| **Config file** | none — Wave 0 task adds pytest to pyproject.toml dev deps |
| **Quick run command** | `uv run pytest tests/test_sampler.py -x -q` |
| **Full suite command** | `uv run pytest tests/ -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_sampler.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 7-W0-01 | W0 | 0 | D-06, D-07 | — | N/A | import smoke | `uv run python -c "from projects.src.data.sampler import get_sampler"` | ❌ W0 | ⬜ pending |
| 7-W0-02 | W0 | 0 | D-03, D-05 | — | N/A | unit | `uv run pytest tests/test_sampler.py::test_enn_keeps_unknowns -x` | ❌ W0 | ⬜ pending |
| 7-W0-03 | W0 | 0 | D-04 | — | N/A | unit | `uv run pytest tests/test_sampler.py::test_edge_consistency -x` | ❌ W0 | ⬜ pending |
| 7-W0-04 | W0 | 0 | D-06 | — | N/A | unit | `uv run pytest tests/test_sampler.py::test_get_sampler_factory -x` | ❌ W0 | ⬜ pending |
| 7-W0-05 | W0 | 0 | D-08 | — | N/A | unit | `uv run pytest tests/test_sampler.py::test_metrics -x` | ❌ W0 | ⬜ pending |
| 7-01-01 | 01 | 1 | D-07 | — | N/A | unit | `uv run pytest tests/test_sampler.py -x -q` | ❌ W0 | ⬜ pending |
| 7-02-01 | 02 | 2 | D-01, D-02 | — | N/A | integration | `uv run python projects/src/train_gcn_skip.py --epochs 2 --sampler enn` | ❌ impl | ⬜ pending |
| 7-02-02 | 02 | 2 | D-09 | — | N/A | integration | `ls projects/results/sampling_comparison.csv` | ❌ impl | ⬜ pending |
| 7-03-01 | 03 | 3 | D-06 | — | N/A | integration | `uv run python projects/src/train_gcn_skip.py --epochs 2 --sampler cluster_centroid` | ❌ impl | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `uv add --dev pytest` — pytest not installed; must be added before any tests can run
- [ ] `tests/__init__.py` — test package init file
- [ ] `tests/test_sampler.py` — unit test stubs covering D-03, D-04, D-05, D-06, D-07, D-08
- [ ] `projects/results/` directory — create if it doesn't exist

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Illicit F1 sanity vs TF reference | D-08/D-09 | Requires full 500-epoch training run | Compare CSV Illicit F1 against reference/EvolveGCNFai/results_skip.txt — expect similar order of magnitude (0.3–0.5 range) |
| CSV contains one row per sampler method run | D-09 | Integration output check | Open projects/results/sampling_comparison.csv and verify columns: method, test_illicit_f1, test_precision, test_recall |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
