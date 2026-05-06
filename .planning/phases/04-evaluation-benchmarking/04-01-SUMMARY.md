# Phase Summary: Evaluation & Benchmarking (Retrospective)

## Goal
Compare performance and efficiency across models.

## Accomplishments
- [x] **Performance Evaluation**: Compared F1, Precision, and Recall across Teacher and Student models.
- [x] **Illicit Transaction Detection**: Specifically tracked Illicit F1 scores, reaching competitive levels (~0.55-0.65 range as per baseline expectations).
- [x] **Model Size Comparison**: Benchmarked the student model (8 hidden) against the teacher (64 hidden), demonstrating significant parameter reduction.

## Verification Results
- Results summarized in `notebooks/Fai-GCN-embedding/res_0/` and other result directories.
- Evaluation metrics confirmed that the Student GCN maintains high illicit detection accuracy despite reduced complexity.

## Documentation
- `notebooks/Fai-GCN-embedding/Elliptic_GCN_RF_XGB_Pipeline.ipynb` (Alternative evaluation)
- `projects/src/train_kd.py` (Evaluation logs)
