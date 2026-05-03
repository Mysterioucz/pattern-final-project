# Roadmap: Bitcoin Fraud Detection KD

## Overview

Modularizing an existing GCN-based Bitcoin fraud detection notebook into a maintainable codebase and implementing Knowledge Distillation to compress the model into an efficient student version.

## Phases

- [ ] **Phase 1: Infrastructure & Modularization** - Extract logic into lean `.py` files and set up data pipeline.
- [ ] **Phase 2: Model Implementation** - Define Teacher and Student GCN architectures and KD loss.
- [ ] **Phase 3: Training & Distillation** - Train Teacher baseline, Vanilla Student, and Distilled Student.
- [ ] **Phase 4: Evaluation & Benchmarking** - Compare performance and efficiency across models.

## Phase Details

### Phase 1: Infrastructure & Modularization

**Goal**: A clean, modular codebase ready for experiments.
**Depends on**: Nothing
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04
**Success Criteria**:

1. `projects/src/` contains separate modules for data, models, and training.
2. Dataset can be downloaded and preprocessed via a single script/command.
3. Data loaders successfully yield batches for the Elliptic dataset.
   **Plans**: 3 plans

Plans:

- [x] 01-01: Set up folder structure and implement data downloader.
- [x] 01-02: Implement data preprocessing and graph construction modules.
- [ ] 01-03: Implement PyTorch Geometric data loaders and split logic.

### Phase 2: Model Implementation

**Goal**: Definition of all model components and distillation logic.
**Depends on**: Phase 1
**Requirements**: MODEL-01, MODEL-02, MODEL-03
**Success Criteria**:

1. Teacher GCN class is defined and configurable.
2. Student GCN class is defined (shallow/narrow).
3. Knowledge Distillation loss function is implemented and tested with dummy data.
   **Plans**: 2 plans

Plans:

- [ ] 02-01: Implement GCN models (Teacher/Student) in `models.py`.
- [ ] 02-02: Implement KD loss and training utility functions.

### Phase 3: Training & Distillation

**Goal**: Trained models for comparison.
**Depends on**: Phase 2
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03
**Success Criteria**:

1. Teacher model reaches baseline performance (Illicit F1 ~0.55-0.63).
2. Student model (Vanilla) is trained and saved.
3. Student model (Distilled) is trained and saved.
   **Plans**: 3 plans

Plans:

- [ ] 03-01: Train and validate Teacher GCN.
- [ ] 03-02: Train and validate Vanilla Student GCN.
- [ ] 03-03: Train and validate Distilled Student GCN using Teacher's knowledge.

### Phase 4: Evaluation & Benchmarking

**Goal**: Final report and performance comparison.
**Depends on**: Phase 3
**Requirements**: EVAL-01, EVAL-02
**Success Criteria**:

1. Metrics (F1, Precision, Recall) are reported for all three models.
2. Model size and inference speed comparison is documented.
3. Final results confirm if KD improved Student performance over Vanilla.
   **Plans**: 1 plan

Plans:

- [ ] 04-01: Run evaluation scripts and generate comparison plots/tables.

## Progress

| Phase                      | Plans Complete | Status      | Completed |
| -------------------------- | -------------- | ----------- | --------- |
| 1. Infrastructure          | 2/3            | In Progress | -         |
| 2. Model Implementation    | 0/2            | Not started | -         |
| 3. Training & Distillation | 0/3            | Not started | -         |
| 4. Evaluation              | 0/1            | Not started | -         |
