# Phase 2: Model Implementation - Context

**Gathered:** 2026-05-03
**Status:** Ready for planning

<domain>
## Phase Boundary
Implementation of the Teacher GCN and Student GCN architectures, along with the Knowledge Distillation (KD) loss framework. This phase focuses on model definitions and the mathematical implementation of the distillation objective.

</domain>

<decisions>
## Implementation Decisions

### Model Architectures
- **D-01:** **Teacher GCN**: 2-layer GCN with 64 hidden units, using Dropout (baseline rate) and no residual connections.
- **D-02:** **Student GCN**: Implement a flexible architecture that supports multiple compression variants:
  - **Narrower**: 2 layers, reduced hidden units (e.g., 16 or 32).
  - **Shallower**: 1 layer, 64 hidden units.
  - **Both**: 1 layer with reduced hidden units.
- **D-03:** Both models will inherit from a common base class in `src/models/base.py` to ensure consistent data handling and interface.

### Knowledge Distillation Strategy
- **D-04:** **Loss Function**: Weighted average of two components:
  1. **Distillation Loss ($L_D$)**: KL-Divergence between student and teacher soft targets (logits scaled by Temperature $T$).
  2. **Student Loss ($L_S$)**: Standard Cross-Entropy between student soft logits (at $T=1$) and ground truth labels.
- **D-05:** **Soft Targets Formula**: $p(z_i, T) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$.
- **D-06:** **T^2 Scaling**: The distillation loss component will be multiplied by $T^2$ to maintain gradient magnitude consistency across different temperatures.

### Hyper-parameters & Configuration
- **D-07:** Default parameters: **Temperature (T) = 2.0**, **Alpha ($\alpha$) = 0.5**.
- **D-08:** Store all model and KD parameters in a central `projects/src/config.py` file.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Documentation
- [PyTorch Geometric GCN Documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html) — Reference for the standard GCN layer.

### Source Notebook
- `projects/Pattern_Project_Elliptic_dataset_GCN.ipynb` — Baseline model architecture and training hyperparameters.

</canonical_refs>
