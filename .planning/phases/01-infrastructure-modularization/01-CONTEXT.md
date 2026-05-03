# Phase 1: Infrastructure & Modularization - Context

**Gathered:** 2026-05-03
**Status:** Ready for planning

<domain>
## Phase Boundary
Extract logic from the `Pattern_Project_Elliptic_dataset_GCN.ipynb` notebook into a modular Python project structure. This includes setting up the Kaggle data pipeline, preprocessing, and GNN data loaders.

</domain>

<decisions>
## Implementation Decisions

### Source Code Organization
- **D-01:** Use a **Domain-Driven (Nested)** structure in `projects/src/`.
- **D-02:** Directory breakdown:
  - `src/data/`: Scripts for Kaggle downloads and Elliptic dataset preprocessing.
  - `src/models/`: GCN model definitions (Teacher/Student) and EvolveGCN for temporal analysis.
  - `src/utils/`: Evaluation metrics and helper functions.
  - `src/main.py`: Main entry point for training and evaluation.

### Kaggle Integration
- **D-03:** Authenticate using **Environment Variables (.env)**.
- **D-04:** Use `KAGGLE_USERNAME` and `KAGGLE_KEY` from the local `.env` file to ensure project-level isolation.

### Data Handling
- **D-05:** Perform **on-the-fly preprocessing**. Do not save intermediate `.pt` files to disk for now to allow for easier experimentation with feature engineering.
- **D-06:** Implement a **Static Graph** approach for the baseline, merging all 49 time steps into a single graph representation.
- **D-07:** Design the `DataLoader` to be extensible for **windowed** or **snapshot-based** processing (future-proofing for EvolveGCN integration).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Notebook
- `projects/Pattern_Project_Elliptic_dataset_GCN.ipynb` — Primary source for logic, imports, and preprocessing steps.

### Reference Implementations
- `reference/GCN-on-EllipticDataSet/` — Existing Python-based GCN implementation for the dataset.
- `reference/GCN_Elliptic_dataset/` — Secondary reference with `main.py` and `pyproject.toml`.
- `reference/EvolveGCN/` — Reference for future temporal graph handling improvements.

### Documentation
- [Elliptic Data Set (Kaggle)](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) — Dataset schema and documentation.

</canonical_refs>
