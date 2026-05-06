# Roadmap: Bitcoin Fraud Detection KD

## Overview
Modularizing an existing GCN-based Bitcoin fraud detection notebook into a maintainable codebase and implementing Knowledge Distillation to compress the model into an efficient student version.

## Milestones
- [x] **v1.0: Baseline & Knowledge Distillation** — [[Archive]](file:///home/chatrin/Documents/Chat/CU/Year-3/2110573-Pattern-Recognition/final-project/.planning/milestones/v1.0-ROADMAP.md) (Shipped 2026-05-06)

## Future Phases
- [ ] **Phase 5: Temporal GNNs** — Integrate EvolveGCN as a teacher or comparison model.
- [ ] **Phase 6: Advanced KD** — Implement feature-based and relation-based distillation.
- [ ] **Phase 7: Oversampling for GCN-Skip** — Evaluate class imbalance sampling methods for GCN with skip connections: ENN, Edge Pruning (ENN at edge node), Cluster Centroid, NearMiss-1, NearMiss-2.
  - **Goal:** Working per-timestep sampling ablation study with GCN(use_skip=True) producing `projects/results/sampling_comparison.csv` with Illicit F1 per method.
  - **Plans:** 3 plans
  - Plans:
    - [x] 07-00-PLAN.md — pytest infrastructure, tests/test_sampler.py stubs (Wave 0)
    - [ ] 07-01-PLAN.md — Implement projects/src/data/sampler.py with all 5 methods (Wave 1)
    - [ ] 07-02-PLAN.md — Implement projects/src/train_gcn_skip.py, CSV output, full ablation (Wave 2)

## Progress
| Milestone | Status | Completed |
|-----------|--------|-----------|
| v1.0      | Shipped| 2026-05-06|
| v1.1      | Pending| -         |
