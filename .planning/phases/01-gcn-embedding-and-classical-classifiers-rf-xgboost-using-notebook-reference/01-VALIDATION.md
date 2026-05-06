---
phase: 01
slug: gcn-embedding-and-classical-classifiers-rf-xgboost-using-notebook-reference
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-06
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 |
| **Config file** | none |
| **Quick run command** | `pytest -q tests/test_phase1_classical.py -x` |
| **Full suite command** | `pytest -q` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest -q tests/test_phase1_classical.py -x`
- **After every plan wave:** Run `pytest -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | REQ-TBD | unit | `pytest -q tests/test_phase1_classical.py::test_embedding_shape_and_determinism -x` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | REQ-TBD | unit/smoke | `pytest -q tests/test_phase1_classical.py::test_rf_pipeline_smoke -x` | ❌ W0 | ⬜ pending |
| 01-01-03 | 02 | 2 | REQ-TBD | unit/smoke | `pytest -q tests/test_phase1_classical.py::test_xgb_pipeline_smoke -x` | ❌ W0 | ⬜ pending |
| 01-01-04 | 02 | 2 | REQ-TBD | unit | `pytest -q tests/test_phase1_classical.py::test_illicit_f1_contract -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_phase1_classical.py` — stubs for REQ-TBD
- [ ] `projects/src/train_gcn_classical.py` — implement deterministic embedding extraction + RF/XGB training path
- [ ] `uv add xgboost` — add missing dependency and update lockfile

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
