# Requirements: Bitcoin Fraud Detection KD

**Defined:** 2026-05-03
**Core Value:** Compressing GCN models via Knowledge Distillation for efficient blockchain fraud detection.

## v1 Requirements

### Infrastructure & Modularization
- [x] **INFRA-01**: Extract notebook logic into lean `.py` files in `projects/src/`
- [x] **INFRA-02**: Implement Kaggle dataset download and extraction script
- [x] **INFRA-03**: Create data preprocessing pipeline for the Elliptic dataset
- [ ] **INFRA-04**: Implement GNN data loaders for training and testing splits

### Model Development
- [ ] **MODEL-01**: Implement Teacher GCN model (baseline architecture)
- [ ] **MODEL-02**: Implement Student GCN model (shallow/narrow architecture)
- [ ] **MODEL-03**: Implement Knowledge Distillation loss (Soft Targets / KL-Divergence)

### Training & Distillation
- [ ] **TRAIN-01**: Train Teacher model and save best weights
- [ ] **TRAIN-02**: Train Student model from scratch (Vanilla baseline)
- [ ] **TRAIN-03**: Train Student model using Knowledge Distillation from Teacher

### Evaluation
- [ ] **EVAL-01**: Compare performance (F1, Precision, Recall, Illicit F1) across all models
- [ ] **EVAL-02**: Benchmark inference time and model size (parameters count)

## v2 Requirements
- **ENH-01**: Support for other GNN architectures (GAT, GraphSAGE)
- **ENH-02**: Implement Feature-based Knowledge Distillation
- **ENH-03**: Real-time inference dashboard

## Out of Scope
| Feature | Reason |
|---------|--------|
| Production Deployment | Focus on research and compression evaluation |
| Multiple Datasets | Initially focus on Elliptic dataset |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Complete (01-01) |
| INFRA-02 | Phase 1 | Complete (01-01) |
| INFRA-03 | Phase 1 | Complete (01-02) |
| INFRA-04 | Phase 1 | Pending |
| MODEL-01 | Phase 2 | Pending |
| MODEL-02 | Phase 2 | Pending |
| MODEL-03 | Phase 2 | Pending |
| TRAIN-01 | Phase 3 | Pending |
| TRAIN-02 | Phase 3 | Pending |
| TRAIN-03 | Phase 3 | Pending |
| EVAL-01 | Phase 4 | Pending |
| EVAL-02 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-03*
*Last updated: 2026-05-03 after initial definition*
