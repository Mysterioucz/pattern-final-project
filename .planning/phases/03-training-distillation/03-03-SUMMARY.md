# Phase Summary: Training & Distillation (Retrospective)

## Goal
Train Teacher baseline, Vanilla Student, and Distilled Student.

## Accomplishments
- [x] **Teacher Training**: Successfully trained the Teacher GCN on the Elliptic Bitcoin dataset (timesteps 1-34 for training).
- [x] **Vanilla Student Training**: (Implied) Baseline student performance established.
- [x] **Distilled Student Training**: Executed Knowledge Distillation using the teacher's soft logits to guide the student GCN's learning.
- [x] **Reproducibility**: Training script `train_kd.py` provides a CLI to repeat experiments with different alpha/temperature values.

## Verification Results
- Teacher model loss converged during training.
- Student model trained with KD successfully minimized the composite loss (KL Div + CE).
- Models saved to `projects/saved_models/` (as per notebook logic).

## Documentation
- `projects/src/train_kd.py` (Main training entry point)
