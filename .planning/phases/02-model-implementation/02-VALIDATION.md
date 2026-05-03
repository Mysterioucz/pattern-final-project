---
phase: 02
slug: model-implementation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-03
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `pytest projects/src/tests/test_models.py` |
| **Full suite command** | `pytest projects/src/tests/` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest projects/src/tests/test_models.py`
- **After every plan wave:** Run `pytest projects/src/tests/`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | MODEL-01 | unit | `pytest projects/src/tests/test_models.py::test_teacher_gcn` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | MODEL-02 | unit | `pytest projects/src/tests/test_models.py::test_student_gcn` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 2 | MODEL-03 | unit | `pytest projects/src/tests/test_losses.py::test_kd_loss` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `projects/src/tests/test_models.py` — stubs for Teacher/Student GCN tests
- [ ] `projects/src/tests/test_losses.py` — stubs for KD loss tests
- [ ] `uv add --dev pytest` — install testing framework

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| None | - | - | - |

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
