# Phase 7: Oversampling for GCN-Skip — Discussion Log

**Date:** 2026-05-06
**Phase:** 07-oversampling-gcn-skip
**Areas discussed:** Codebase target, Sampling application point, What to implement, Results reporting

---

## Area 1: Codebase Target

**Q1:** Which codebase should Phase 7 target?
- Options: PyTorch/PyG projects/src/ | TensorFlow reference (EvolveGCNFai) | Both
- **Selected:** PyTorch/PyG projects/src/ (Recommended)

**Q2:** How to structure the GCN-skip model?
- Options: Extend existing GCN (use_skip=True) | New dedicated GCNSkip class
- **Selected:** Extend existing GCN (use_skip=True) — add train_gcn_skip.py script

---

## Area 2: Sampling Application Point

**Q1:** Where should sampling be applied in the pipeline?
- Options: Node feature space per timestep | Full dataset across timesteps | You decide
- **Selected:** Node feature space per timestep (Recommended)

**Q2:** What happens to edges when a node is removed?
- Options: Remove incident edges | Keep all original edges | You decide
- **Selected:** Remove incident edges (induced subgraph)

**Q3:** How should unknown-class nodes be handled?
- Options: Exclude from sampling, keep all | Filter out before sampling | You decide
- **Selected:** Exclude from sampling decisions — always kept

---

## Area 3: What to Implement

**Q1:** What is the goal — new experiments or reproduce known results?
- Options: Reproduce all 5 as ablation | Document known + implement ENN only | Implement all 5, skip worse in CI
- **Selected:** Implement all 5 but skip worse ones in CI (opt-in via --sampler flag)

**Q2:** How to structure sampling methods in codebase?
- Options: Single sampler module with registry | Separate script per method | You decide
- **Selected:** Single sampler module with registry (projects/src/data/sampler.py, get_sampler(name) factory)

---

## Area 4: Results Reporting

**Q1:** Primary metric for comparing methods?
- Options: Illicit F1 score | Macro F1 | Precision + Recall separately
- **Selected:** Illicit F1 score (Recommended)

**Q2:** How to store and present results?
- Options: CSV + printed table | Notebook visualization | You decide
- **Selected:** CSV (projects/results/sampling_comparison.csv) + printed comparison table

---

## Claude's Discretion

- Epoch count and hyperparameters (match reference: 500 epochs, lr=1e-3)
- ENN/NearMiss `n_neighbors` hyperparameters (imbalanced-learn defaults)
- Edge Pruning implementation detail (graph-neighborhood as k-NN set)
- CSV column schema and table formatting

## Deferred Ideas

None.
