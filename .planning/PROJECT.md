# Bitcoin Fraud Detection - GCN & Knowledge Distillation

## What This Is
A graph-based fraud detection system for the Bitcoin network (Elliptic Dataset) using Graph Convolutional Networks (GCN). The project aims to improve efficiency by distilling knowledge from a larger Teacher GCN into a smaller, shallower Student GCN while maintaining high detection performance (specifically Illicit F1 score).

## Core Value
By implementing Knowledge Distillation (KD), we can deploy more efficient models (Student) that are faster and consume fewer resources, which is critical for real-time fraud detection in large-scale blockchain networks.

## Context
- **Dataset**: [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) (203,769 nodes, 234,355 edges).
- **Domain**: Blockchain forensics / Pattern Recognition.
- **Tech Stack**: Python, PyTorch, PyTorch Geometric, `uv`, Kaggle API.

## Requirements

### Active
- [ ] Modularize existing notebook implementation into `projects/src/`
- [ ] Implement data downloading and preprocessing pipeline (Kaggle integration)
- [ ] Train a "Teacher" GCN model (baseline performance)
- [ ] Implement Knowledge Distillation framework (KL-Divergence / Soft Targets)
- [ ] Train and evaluate a "Student" GCN model (smaller/shallower)
- [ ] Compare results: Teacher vs. Student (Vanilla) vs. Student (Distilled)

### Out of Scope
- [ ] Deploying as a real-time API (initially focus on training and evaluation)
- [ ] Support for other GNN architectures (e.g., GAT, GraphSAGE) unless needed for distillation comparison

## Current State
- **v1.0 (Shipped 2026-05-06)**: Modularized GCN framework with Knowledge Distillation (Teacher -> Student). 
- **Baseline Performance**: Achieved illicit detection parity with notebook implementation.
- **Model Compression**: Student GCN (8 hidden units) successfully distilled from Teacher (64 units).

## Next Milestone Goals (v1.1/v2.0)
- Support for temporal GNNs (EvolveGCN) as potential teachers.
- Implementation of feature-based KD (beyond just logit distillation).
- Interactive evaluation dashboard for metric visualization.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Modular Structure | Lean files in `projects/src/` for better maintainability | Validated (v1.0) |
| Knowledge Distillation | Use KD to compress the GCN model for efficiency | Validated (v1.0) |
| Dataset Handling | Store raw data in `projects/data/` | Validated (v1.0) |

## Evolution
This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone**:
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-06 after v1.0 completion*
