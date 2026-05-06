# Phase 7: Oversampling for GCN-Skip - Pattern Map

**Mapped:** 2026-05-06
**Files analyzed:** 5 (new/modified files)
**Analogs found:** 5 / 5

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `projects/src/data/sampler.py` | utility | transform | `projects/src/utils/kd_loss.py` | role-match (utility/transform module) |
| `projects/src/train_gcn_skip.py` | training script | request-response (per-timestep) | `projects/src/train_kd.py` | exact |
| `projects/results/sampling_comparison.csv` | output artifact | file-I/O | n/a — runtime output, no analog needed | n/a |
| `tests/__init__.py` | config | n/a | `projects/src/data/__init__.py` | exact (empty init pattern) |
| `tests/test_sampler.py` | test | request-response | `reference/GCN-on-EllipticDataSet/test.py` | partial (different framework) |

---

## Pattern Assignments

### `projects/src/data/sampler.py` (utility, transform)

**Analog:** `projects/src/utils/kd_loss.py`

**Module structure pattern** (`projects/src/utils/kd_loss.py`, lines 1-3):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

**Imports pattern to use** (based on analog + RESEARCH.md verified APIs):
```python
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from imblearn.under_sampling import EditedNearestNeighbours, NearMiss, ClusterCentroids
```

**Factory function pattern** (adapt from `kd_loss.py` single-function module structure, lines 5-43):
- `kd_loss.py` exports one public function (`kd_loss(...)`) with typed args, a docstring, and a single return value.
- `sampler.py` follows the same convention: one public factory (`get_sampler(name)`) and one public transform (`apply_sampling(data, sampler_name)`).
- No class required — plain functions, consistent with the rest of `projects/src/utils/`.

**Core pattern — factory + transform functions:**
```python
def get_sampler(name: str):
    """Return an initialised sampler for the given name.

    Supported names: 'enn', 'edge_pruning', 'cluster_centroid',
                     'nearmiss1', 'nearmiss2'.
    'edge_pruning' returns None (custom implementation, not an imblearn object).
    """
    name = name.lower()
    if name == 'enn':
        return EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=3)
    elif name == 'nearmiss1':
        return NearMiss(version=1, n_neighbors=3)
    elif name == 'nearmiss2':
        return NearMiss(version=2, n_neighbors=3)
    elif name == 'cluster_centroid':
        return ClusterCentroids(voting='hard', random_state=42)
    elif name == 'edge_pruning':
        return None  # sentinel: custom path in apply_sampling
    else:
        raise ValueError(f"Unknown sampler: {name!r}. "
                         "Choose from: enn, edge_pruning, cluster_centroid, nearmiss1, nearmiss2")


def apply_sampling(data: Data, sampler_name: str) -> Data:
    """Apply the named sampling method to a single-timestep PyG Data object.

    Unknown nodes (y == 2) are always kept and excluded from sampling decisions (D-05).
    Surviving nodes and their induced edge subgraph are returned as a new Data object.

    Args:
        data:         PyG Data with fields x (N, F), edge_index (2, E), y (N,).
        sampler_name: One of 'enn', 'edge_pruning', 'cluster_centroid',
                      'nearmiss1', 'nearmiss2'. Case-insensitive.

    Returns:
        New Data object with surviving nodes and induced edges.
    """
```

**Edge-pruning helper pattern** (custom; no imblearn analog — from RESEARCH.md verified pseudocode):
```python
def _edge_pruning(data: Data) -> torch.Tensor:
    """Return global indices of nodes surviving edge-pruning (D-03/D-04)."""
    y = data.y
    N = data.num_nodes
    src = data.edge_index[0].tolist()
    dst = data.edge_index[1].tolist()
    adj: dict[int, set[int]] = {i: set() for i in range(N)}
    for s, d in zip(src, dst):
        adj[s].add(d)
        adj[d].add(s)  # undirected

    surviving = []
    for n in range(N):
        label = y[n].item()
        if label != 1:          # unknown (2) and illicit (0): always keep
            surviving.append(n)
        else:                   # licit: apply ENN-on-graph rule
            labeled_nbr_labels = [
                y[j].item() for j in adj[n] if y[j].item() != 2
            ]
            if len(labeled_nbr_labels) == 0:
                surviving.append(n)
            elif sum(lbl != label for lbl in labeled_nbr_labels) > len(labeled_nbr_labels) / 2:
                pass  # majority disagree -> remove
            else:
                surviving.append(n)
    return torch.tensor(surviving, dtype=torch.long)
```

**Graph reconstruction pattern** (from RESEARCH.md Section 2, VERIFIED against PyG 2.7.0):
```python
# Inside apply_sampling(), after computing surviving_global_labeled (LongTensor):
all_surviving = torch.cat([surviving_global_labeled, unknown_global_idx])
all_surviving_sorted, _ = all_surviving.sort()

new_edge_index, _ = subgraph(
    all_surviving_sorted,
    data.edge_index,
    relabel_nodes=True,
    num_nodes=data.num_nodes,
)

new_x = data.x[all_surviving_sorted]
new_y = data.y[all_surviving_sorted]
return Data(x=new_x, edge_index=new_edge_index, y=new_y)
```

**NearMiss n_neighbors guard** (CRITICAL — from RESEARCH.md Pitfall 2):
```python
# Before constructing NearMiss, clamp n_neighbors to minority class size:
n_min = int((y_labeled == 0).sum())
n_neighbors = min(3, n_min)
sampler = NearMiss(version=v, n_neighbors=n_neighbors)
```

**ClusterCentroids row-match recovery** (no sample_indices_ — from RESEARCH.md Section 1):
```python
# After cc.fit_resample(X_labeled, y_labeled) with voting='hard':
surviving_local = []
for row in X_res:
    diffs = np.abs(X_labeled - row).sum(axis=1)
    surviving_local.append(int(np.argmin(diffs)))
surviving_local = np.array(surviving_local)
```

---

### `projects/src/train_gcn_skip.py` (training script, request-response)

**Analog:** `projects/src/train_kd.py` (exact structural match)

**Imports pattern** (`projects/src/train_kd.py`, lines 1-9):
```python
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import f1_score, accuracy_score

from data.preprocessor import load_static_graph
from models.gcn import GCN
from utils.kd_loss import kd_loss
```

Adapt for this script:
```python
import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from data.preprocessor import load_snapshots
from data.sampler import apply_sampling
from models.gcn import GCN
```

**sys.path setup pattern** (`projects/src/main.py`, lines 29-33):
```python
_SRC_DIR = Path(__file__).resolve().parent
_PROJECTS_DIR = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
```

**argparse pattern** (`projects/src/train_kd.py`, lines 92-97):
```python
parser = argparse.ArgumentParser(description="Knowledge Distillation: GCN -> MLP")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--alpha", type=float, default=0.5, help="KD alpha (weight of KL div)")
parser.add_argument("--temperature", type=float, default=2.0, help="KD temperature")
args = parser.parse_args()
```

Adapt for this script (reference values: NUM_EPOCH=500, LR=1e-3 from CONTEXT.md Specifics):
```python
parser = argparse.ArgumentParser(description="GCN-Skip with sampling ablation")
parser.add_argument("--sampler", type=str, default="enn",
                    choices=["enn", "edge_pruning", "cluster_centroid", "nearmiss1", "nearmiss2", "none"],
                    help="Sampling method to apply to training snapshots (default: enn)")
parser.add_argument("--epochs", type=int, default=500, help="Training epochs (default: 500)")
parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default: 1e-3)")
args = parser.parse_args()
```

**Model initialisation pattern** (`projects/src/train_kd.py`, lines 108-113):
```python
teacher = GCN(num_node_features=num_node_features, hidden_channels=[64], use_skip=False)
```

Adapt (use_skip=True, num_node_features from data at runtime per Pitfall 6):
```python
# Load snapshots first, then derive num_node_features from data:
snapshots = load_snapshots(normalize=True, filter_unknown=False)  # D-05
num_node_features = snapshots[0]['x'].shape[1]  # 165 in practice (Pitfall 6)
model = GCN(num_node_features=num_node_features, hidden_channels=[64], use_skip=True)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
```

**Training loop pattern** (`projects/src/train_kd.py`, lines 11-27):
```python
def train_teacher(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(data, return_logits=True)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Teacher Epoch {epoch:03d}, Loss: {loss.item():.4f}")
    return model
```

Adapt for per-timestep snapshot loop with sampler hook (from RESEARCH.md Section 4):
```python
def train(model, train_snapshots, optimizer, sampler_name, epochs=500):
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for snap in train_snapshots:
            data = Data(
                x=snap['x'],
                edge_index=to_undirected(snap['edge_index']),
                y=snap['y'],
            )
            if sampler_name != 'none':
                data = apply_sampling(data, sampler_name)  # hook point (D-03)
            optimizer.zero_grad()
            out = model(data, return_logits=True)
            labeled_mask = data.y != 2
            loss = F.cross_entropy(out[labeled_mask], data.y[labeled_mask])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d}, Loss: {epoch_loss:.4f}")
```

**Evaluation pattern** (`projects/src/train_kd.py`, lines 69-90):
```python
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data, return_logits=True)
    pred = out.argmax(dim=1)
    test_mask = data.test_mask
    y_true = data.y[test_mask].cpu().numpy()
    y_pred = pred[test_mask].cpu().numpy()
    f1_illicit = f1_score(y_true, y_pred, pos_label=0, average='binary')
    print(f"Test F1 (Illicit): {f1_illicit:.4f}")
```

Adapt for snapshot-based evaluation (no test_mask field; use `y != 2` as labeled mask):
```python
def evaluate(model, test_snapshots):
    """Evaluate across all test snapshots; return (illicit_f1, precision, recall)."""
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for snap in test_snapshots:
            data = Data(x=snap['x'], edge_index=to_undirected(snap['edge_index']), y=snap['y'])
            out = model(data, return_logits=True)
            pred = out.argmax(dim=1)
            labeled_mask = data.y != 2
            y_true_all.extend(data.y[labeled_mask].cpu().tolist())
            y_pred_all.extend(pred[labeled_mask].cpu().tolist())
    illicit_f1 = f1_score(y_true_all, y_pred_all, pos_label=0, average='binary')
    precision  = precision_score(y_true_all, y_pred_all, pos_label=0, average='binary', zero_division=0)
    recall     = recall_score(y_true_all, y_pred_all, pos_label=0, average='binary', zero_division=0)
    return illicit_f1, precision, recall
```

**CSV output pattern** (no direct analog — use stdlib csv; `os.makedirs` guard from RESEARCH.md Pitfall 8):
```python
RESULTS_CSV = Path(__file__).resolve().parent.parent.parent / "results" / "sampling_comparison.csv"

def append_results(method, illicit_f1, precision, recall):
    os.makedirs(RESULTS_CSV.parent, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'test_illicit_f1', 'test_precision', 'test_recall'])
        if write_header:
            writer.writeheader()
        writer.writerow({
            'method': method,
            'test_illicit_f1': f"{illicit_f1:.4f}",
            'test_precision': f"{precision:.4f}",
            'test_recall': f"{recall:.4f}",
        })
```

**main() guard pattern** (`projects/src/train_kd.py`, line 132 / `projects/src/main.py`, line 154):
```python
if __name__ == "__main__":
    main()
```

---

### `projects/results/sampling_comparison.csv` (output artifact, file-I/O)

This file is created at runtime by `train_gcn_skip.py`. No source analog needed — the `append_results()` function in `train_gcn_skip.py` (see above) creates it via `csv.DictWriter`. The planner should note:
- Directory does not yet exist; must be created with `os.makedirs(..., exist_ok=True)`.
- Column schema: `method`, `test_illicit_f1`, `test_precision`, `test_recall`.

---

### `tests/__init__.py` (config, n/a)

**Analog:** `projects/src/data/__init__.py` (empty init file, 1 line)

The existing `projects/src/data/__init__.py` is an empty file (1 line). `tests/__init__.py` follows the identical pattern — an empty file that marks the directory as a Python package.

**Pattern:** Empty file (no content required).

---

### `tests/test_sampler.py` (test, request-response)

**Analog:** `reference/GCN-on-EllipticDataSet/test.py` (partial — different test framework; provides per-timestep evaluation loop style reference)

This project has no existing pytest test files. The test file follows pytest conventions (no existing project analog), drawing from RESEARCH.md Section 6 smoke test design.

**Imports pattern** (from RESEARCH.md Section 6, aligned with project imports style):
```python
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from projects.src.data.sampler import apply_sampling, get_sampler
```

**Fixture / helper pattern** (from RESEARCH.md Section 6):
```python
def make_mini_snapshot(seed: int = 0) -> Data:
    """Synthetic snapshot: 5 illicit (0), 15 licit (1), 5 unknown (2), 25 nodes total."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    N = 25
    x = torch.randn(N, 10)
    y = torch.tensor([0] * 5 + [1] * 15 + [2] * 5)
    ei = torch.tensor([[0, 1, 2, 5, 6, 20, 21], [1, 2, 3, 6, 7, 21, 22]], dtype=torch.long)
    return Data(x=x, edge_index=ei, y=y)
```

**Validation helper pattern** (from RESEARCH.md Section 6):
```python
def verify_data_object(data: Data) -> None:
    N = data.num_nodes
    assert data.x.shape[0] == N
    assert data.y.shape[0] == N
    if data.edge_index.shape[1] > 0:
        assert data.edge_index.max() < N, (
            f"edge_index out of range: max={data.edge_index.max()}, N={N}"
        )
```

**Test function naming convention** (pytest style — no class required, function prefix `test_`):
```python
def test_enn_keeps_unknowns():
    ...

def test_all_samplers_smoke():
    for name in ['enn', 'edge_pruning', 'cluster_centroid', 'nearmiss1', 'nearmiss2']:
        ...

def test_get_sampler_factory():
    for name in ['enn', 'nearmiss1', 'nearmiss2', 'cluster_centroid']:
        s = get_sampler(name)
        assert s is not None
    assert get_sampler('edge_pruning') is None  # sentinel

def test_invalid_sampler_raises():
    with pytest.raises(ValueError):
        get_sampler('unknown_method')
```

---

## Shared Patterns

### Snapshot data loading
**Source:** `projects/src/data/preprocessor.py`, `load_snapshots()` function (lines 287-387)
**Apply to:** `train_gcn_skip.py`

Key facts confirmed in the source:
- Returns a `list[dict]` with keys: `edge_index`, `x`, `y`, `mask`, `timestep`, `split`.
- Feature tensor `x` shape is `(N_t, 165)` — the "time step" column is dropped (see docstring line 303: `FloatTensor shape (N_t, 165)`).
- Use `filter_unknown=False` per D-05 so unknown nodes remain in the graph for message passing.
- Train split: `snap['split'] == 'train'` (timesteps 1–34); test split: `snap['split'] == 'test'` (timesteps 35–49).

```python
# Correct call in train_gcn_skip.py (D-05 compliance):
snapshots = load_snapshots(normalize=True, filter_unknown=False)
train_snapshots = [s for s in snapshots if s['split'] == 'train']
test_snapshots  = [s for s in snapshots if s['split'] == 'test']
```

### GCN model interface
**Source:** `projects/src/models/gcn.py`, `GCN.forward()` (lines 62-89)
**Apply to:** `train_gcn_skip.py`

```python
# Correct instantiation (use_skip=True, hidden_channels=[64] as per reference):
model = GCN(num_node_features=num_node_features, hidden_channels=[64], use_skip=True)

# Forward pass — always use return_logits=True for cross_entropy:
out = model(data, return_logits=True)  # shape (N, 2)

# Loss: mask to labeled nodes only (y != 2):
labeled_mask = data.y != 2
loss = F.cross_entropy(out[labeled_mask], data.y[labeled_mask])
```

### to_undirected edge conversion
**Source:** `projects/src/data/loader.py`, `get_snapshot_loaders()` (line 62)
**Apply to:** `train_gcn_skip.py`

```python
from torch_geometric.utils import to_undirected
edge_index = to_undirected(snap['edge_index'])
```

`loader.py` wraps every snapshot edge_index with `to_undirected()` before building `Data` objects. `train_gcn_skip.py` must do the same for consistency before both the sampler hook and the model forward pass.

### Illicit F1 metric computation
**Source:** `projects/src/train_kd.py`, `evaluate()` (line 85)
**Apply to:** `train_gcn_skip.py`, `tests/test_sampler.py`

```python
# pos_label=0 because illicit class is encoded as 0 (original label 1 → encoded 0)
f1_illicit = f1_score(y_true, y_pred, pos_label=0, average='binary')
```

### Path-based sys.path setup
**Source:** `projects/src/main.py` (lines 29-33)
**Apply to:** `train_gcn_skip.py`

```python
from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
```

This pattern allows the script to be run as `python projects/src/train_gcn_skip.py` from the repo root, matching how `train_kd.py` is invoked.

---

## No Analog Found

All files have analogs. No entries needed here.

---

## Metadata

**Analog search scope:** `projects/src/`, `projects/src/data/`, `projects/src/utils/`, `projects/src/models/`, `reference/GCN-on-EllipticDataSet/`, `projects/src/main.py`
**Files scanned:** 8 source files fully read
**Pattern extraction date:** 2026-05-06
