# RESEARCH.md - Phase 02: Model Implementation

## DOMAIN OVERVIEW
The Elliptic Data Set is a graph representation of Bitcoin transactions where nodes are transactions and edges are BTC flows.
- **Task**: Classify nodes as "illicit" (class 1) or "licit" (class 2).
- **Challenge**: Extreme class imbalance (~3.4% illicit) and large number of "unknown" nodes.
- **KD Goal**: Transfer detection capability from a robust Teacher GCN to a lighter Student GCN for faster inference.

## PATTERNS & REUSABLE ASSETS
- **PyG GCNConv**: The fundamental building block for the architecture.
- **Temperature Scaling**: $T=2.0$ as a standard for extracting soft targets.
- **Flexible Architecture**: A single class `GCN(nn.Module)` that accepts `num_layers` and `hidden_channels` to support both Teacher and Student variants.

## TECHNICAL APPROACH
### 1. Model Architecture
A modular `GCN` class using `torch_geometric.nn.GCNConv`. By parameterizing `num_layers` and `hidden_channels`, we can instantiate the Teacher (2 layers, 64 hidden units) and various Student configurations (Narrower, Shallower, Both) from the same base logic.

### 2. Distillation Logic
The distillation loss is implemented as a weighted sum of:
1. **KL Divergence**: Between student and teacher soft targets, scaled by $T^2$.
2. **Cross Entropy**: Between student predictions and ground truth labels.

Formula: $L = \alpha \cdot L_{CE} + (1-\alpha) \cdot T^2 \cdot L_{KL}$

### 3. Implementation Details
- **Teacher Mode**: Must be in `eval()` mode with `torch.no_grad()` during student training.
- **Softmax**: Temperature-scaled softmax $p(z_i, T) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$ for both models.
- **Masking**: Only calculate loss for nodes with known labels (class 1 or 2).

## DEPENDENCIES & CONSTRAINTS
- **PyG**: Requires `torch-geometric`.
- **Memory**: Teacher and Student must both fit in memory during distillation training.

## VALIDATION ARCHITECTURE
- **Shape Checks**: Ensure models output `(N, 2)` for any input graph.
- **Logit Distribution**: Verify that higher $T$ indeed "softens" the output distribution.
- **Loss Value**: Ensure $L_{KL}$ decreases as the student learns from the teacher.
