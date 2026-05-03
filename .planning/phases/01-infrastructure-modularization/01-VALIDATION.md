# Phase 01: infrastructure-modularization - Validation Strategy

**Date:** 2026-05-03
**Status:** Active

## 1. Domain Validation
Verify that the modularized structure correctly represents the Elliptic Bitcoin dataset and follows PyTorch Geometric (PyG) best practices.

### Infrastructure Checks
- [ ] `projects/src/data/downloader.py` successfully downloads and extracts the dataset from Kaggle.
- [ ] `projects/src/data/dataset.py` correctly filters "unknown" labels and normalizes features.
- [ ] `projects/src/models/` contains at least `gcn.py` and `evolve_gcn.py` with valid PyG layer definitions.

## 2. Data Integrity
- [ ] Check node counts: Total nodes after filtering "unknown" labels should match expected counts (~46k labeled nodes).
- [ ] Check feature shapes: Node features should have 166 dimensions (or as specified in the notebook).
- [ ] Verify temporal consistency: Data loader can correctly split data by the 49 time steps.

## 3. Performance & Resource Benchmarks
- [ ] On-the-fly preprocessing time: Merging CSVs and building the graph should complete in a reasonable time (e.g., < 2 minutes).
- [ ] Memory usage: InMemoryDataset should fit comfortably in available RAM (~2-4GB expected).

## 4. Integration Verification
- [ ] Run a baseline test: Training a simple GCN for 1 epoch using the new `src/main.py` entry point.
- [ ] Metrics: Verify that `src/utils/metrics.py` correctly calculates F1-score for the "Illicit" class (Class 1).
