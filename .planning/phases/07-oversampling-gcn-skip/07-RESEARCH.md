# Phase 7: Oversampling for GCN-Skip - Research

**Researched:** 2026-05-06
**Domain:** Graph node sampling + imbalanced-learn + PyTorch Geometric data reconstruction
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Target the existing PyTorch/PyG `projects/src/` codebase.
- **D-02:** Extend existing `GCN(use_skip=True)` from `projects/src/models/gcn.py`. New script: `projects/src/train_gcn_skip.py`.
- **D-03:** Apply sampling per-timestep in node feature space.
- **D-04:** Remove incident edges of dropped nodes (induced subgraph via `edge_index` filtering).
- **D-05:** Unknown-class nodes (label=2) always kept, excluded from sampling decisions.
- **D-06:** All 5 methods implemented; ENN=default, others opt-in via `--sampler` flag.
- **D-07:** All methods in `projects/src/data/sampler.py` with `get_sampler(name)` factory. Supported names: `'enn'`, `'edge_pruning'`, `'cluster_centroid'`, `'nearmiss1'`, `'nearmiss2'`.
- **D-08:** Primary metric = Illicit F1 score.
- **D-09:** Output = `projects/results/sampling_comparison.csv` + printed comparison table.

### Claude's Discretion
- Number of training epochs and hyperparameters — match reference values where available (`NUM_EPOCH=500`, `LEARNING_RATE=1e-3`).
- ENN and NearMiss hyperparameters (`n_neighbors`, `version`) — use imbalanced-learn defaults unless reference suggests otherwise.
- Edge Pruning implementation detail — apply ENN logic on each node's graph-neighborhood (direct neighbors as the k-NN set) rather than feature-space k-NN.
- CSV column schema and table formatting.

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope.
</user_constraints>

---

## Research Summary

Phase 7 evaluates five class-imbalance methods applied per-timestep to the GCN-skip model on Elliptic. The core implementation is a `sampler.py` module that receives a PyG `Data` object (one timestep snapshot), applies the selected sampling method to labeled nodes only, and returns a rebuilt `Data` object with surviving nodes and their induced edge subgraph. Unknown nodes (label=2) pass through untouched per D-05.

Three of the five methods (ENN, NearMiss-1, NearMiss-2) use `imbalanced-learn 0.14.1` — the version already added to `pyproject.toml`. These three expose a `sample_indices_` attribute after `fit_resample()` that gives the indices of kept samples into the input array, making graph reconstruction straightforward. ClusterCentroids uses `voting='hard'` to return real samples (no synthetic points), and surviving indices are recovered via nearest-row matching. Edge Pruning is a custom implementation using direct graph neighbors instead of feature-space k-NN.

PyG's `torch_geometric.utils.subgraph(subset, edge_index, relabel_nodes=True)` is the correct tool for rebuilding the edge index after sampling — it filters edges, keeps only those with both endpoints in `subset`, and remaps indices to a contiguous 0-based range in one call. This was verified against the installed PyG 2.7.0.

**Primary recommendation:** Build `sampler.py` with a `apply_sampling(data, sampler_name)` function that (1) separates labeled from unknown nodes, (2) calls the appropriate method on the labeled subset, (3) recovers surviving global indices via `sample_indices_` or row matching, (4) combines with unknown indices, and (5) calls `subgraph(sorted_all_surviving, data.edge_index, relabel_nodes=True)`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Class-imbalance sampling | Data preprocessing (CPU) | — | Applied before any GCN forward pass; pure numpy/sklearn operation |
| Graph reconstruction after sampling | Data preprocessing (CPU) | — | PyG `subgraph()` utility, applied before feeding to GCN |
| GCN forward pass with skip | Model (GPU/CPU) | — | Existing `GCN(use_skip=True)` — no changes needed |
| Per-timestep training loop | Training script | — | New `train_gcn_skip.py` iterates over snapshots |
| Results aggregation | Training script | File I/O | CSV written at end of each sampler run |

---

## 1. imbalanced-learn API Reference

### Version and Installation

- **Version:** `0.14.1` (latest as of 2026-05-06) [VERIFIED: pip index versions]
- **Status:** Already added to `pyproject.toml` via `uv add imbalanced-learn` during research [VERIFIED: pyproject.toml]
- **Import path:** `from imblearn.under_sampling import EditedNearestNeighbours, NearMiss, ClusterCentroids`

### EditedNearestNeighbours (ENN)

**Class:** `imblearn.under_sampling.EditedNearestNeighbours` [VERIFIED: Context7 + official docs]

**Constructor:**
```python
EditedNearestNeighbours(
    *,
    sampling_strategy='auto',  # 'auto' = 'not minority' = only resample majority classes
    n_neighbors=3,
    kind_sel='all',            # 'all': remove if ALL k-NN disagree; 'mode': majority disagree
    n_jobs=None
)
```

**Behavior:**
- Removes majority-class nodes whose k nearest feature-space neighbors do not agree with their label.
- `sampling_strategy='auto'` means `'not minority'`: only removes from majority class (licit=1). Illicit nodes (minority) are NEVER removed.
- Supports multi-class via one-vs-rest scheme. [VERIFIED: official docs]
- With binary labeled data (illicit=0, licit=1 after excluding unknowns): removes noisy licit nodes.

**Post-fit attributes:**
- `sample_indices_`: `ndarray` of shape `(n_kept_samples,)` — indices into the `X` array passed to `fit_resample()` [VERIFIED: official docs + local test]

**Verified behavior on 120-sample binary dataset:**
```
Before: {0: 30, 1: 90}
After:  {0: 30, 1: 57}  # ENN removed 33 noisy licit nodes
sample_indices_.shape = (87,)  # 87 surviving indices into original array
```

### NearMiss

**Class:** `imblearn.under_sampling.NearMiss` [VERIFIED: Context7 + official docs]

**Constructor:**
```python
NearMiss(
    *,
    sampling_strategy='auto',  # 'not minority' = only downsample majority
    version=1,                 # 1 or 2 (or 3, not used here)
    n_neighbors=3,
    n_neighbors_ver3=3,        # only relevant for version=3
    n_jobs=None
)
```

**Version semantics:** [VERIFIED: official docs]
- `version=1`: Keeps majority-class samples with the SMALLEST average distance to their k nearest MINORITY neighbors. Selects licit nodes closest to illicit nodes.
- `version=2`: Keeps majority-class samples with the SMALLEST average distance to the k FARTHEST minority neighbors. More conservative — selects licit nodes that are borderline-far from illicit.

**Post-fit attributes:**
- `sample_indices_`: `ndarray` — indices of kept samples into input array [VERIFIED: official docs + local test]

**Key behavior:** `sampling_strategy='auto'` reduces majority class to match minority count (1:1 ratio).

**Verified behavior on 120-sample binary dataset:**
```
NearMiss-1: Before {0:30, 1:90} -> After {0:30, 1:30}  (aggressive 1:1 balance)
NearMiss-2: Before {0:30, 1:90} -> After {0:30, 1:30}
```

**CRITICAL PITFALL:** NearMiss raises `ValueError: Expected n_neighbors <= n_samples_fit` if `n_neighbors > n_minority_count`. The sampler.py implementation MUST clamp: [VERIFIED: local test]
```python
n_min = (y_labeled == minority_class).sum()
n_neighbors = min(3, n_min)
nm = NearMiss(version=v, n_neighbors=n_neighbors)
```
For Elliptic with ~200+ labeled nodes per timestep, the default `n_neighbors=3` is always safe [ASSUMED — based on known dataset scale; confirm if a very sparse timestep exists], but defensive clamping is required for robustness.

### ClusterCentroids

**Class:** `imblearn.under_sampling.ClusterCentroids` [VERIFIED: Context7 + official docs + local test]

**Constructor:**
```python
ClusterCentroids(
    *,
    sampling_strategy='auto',
    random_state=None,
    estimator=None,   # default: KMeans
    voting='auto'     # 'auto' -> 'soft' for dense numpy arrays
)
```

**CRITICAL: Synthetic vs. Real Samples:** [VERIFIED: local test]
- `voting='soft'` (default for dense numpy arrays): output X contains SYNTHETIC cluster centroids. These do NOT correspond to real nodes and CANNOT be mapped back to graph positions.
- `voting='hard'`: output X contains the REAL sample closest to each centroid. These can be matched back to original nodes via nearest-row search.

**Required override for graph reconstruction:** Always use `voting='hard'` in the sampler implementation.

**sample_indices_ attribute:** ClusterCentroids does NOT expose `sample_indices_`. [VERIFIED: official docs + local test] Use row-matching to recover original indices:
```python
cc = ClusterCentroids(random_state=42, voting='hard')
X_res, y_res = cc.fit_resample(X_labeled, y_labeled)
# Recover surviving indices via row-matching
surviving_local = []
for row in X_res:
    diffs = np.abs(X_labeled - row).sum(axis=1)
    surviving_local.append(int(np.argmin(diffs)))
surviving_local = np.array(surviving_local)
```
Verified: all recovered rows match originals exactly with `voting='hard'`. [VERIFIED: local test]

### fit_resample() Shape Compatibility

All three methods accept `X` as `numpy.ndarray` of shape `(N, F)` where F can be 166 (Elliptic feature count). [VERIFIED: local test with F=10, confirmed by docs for arbitrary F]

---

## 2. PyG Data Reconstruction After Sampling

### Canonical Pattern: `torch_geometric.utils.subgraph`

**Verified API:** [VERIFIED: local test against PyG 2.7.0]

```python
from torch_geometric.utils import subgraph

subgraph(
    subset,           # LongTensor or list[int] of node indices to KEEP (in old numbering)
    edge_index,       # original edge_index
    edge_attr=None,   # optional edge attributes
    relabel_nodes=True,  # remap node indices to contiguous 0-based range
    num_nodes=N_total,   # total nodes in original graph
)
# Returns: (new_edge_index, new_edge_attr)
```

With `relabel_nodes=True`: if `subset = [0, 2, 4, 5, 6, 7]` (old indices), the returned `new_edge_index` uses new indices `[0, 1, 2, 3, 4, 5]`. [VERIFIED: local test]

### Full Reconstruction Pipeline

```python
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

def apply_sampling(data: Data, sampler) -> Data:
    """Rebuild Data object after sampling. Unknown nodes (y==2) always kept."""
    y = data.y  # shape (N,)

    # 1. Separate labeled vs unknown
    labeled_global_idx = (y != 2).nonzero(as_tuple=False).squeeze(1)  # LongTensor
    unknown_global_idx = (y == 2).nonzero(as_tuple=False).squeeze(1)  # LongTensor

    X_labeled = data.x[labeled_global_idx].numpy()  # (N_labeled, 166)
    y_labeled = y[labeled_global_idx].numpy()        # (N_labeled,)

    # 2. Apply sampler -> get surviving local indices into X_labeled
    X_res, y_res = sampler.fit_resample(X_labeled, y_labeled)
    surviving_local = sampler.sample_indices_  # ndarray, indices into X_labeled
    # (For ClusterCentroids: use row-matching instead, see Section 1)

    # 3. Map surviving local indices -> global node indices
    surviving_global_labeled = labeled_global_idx[surviving_local]

    # 4. Combine with unknowns (always kept)
    all_surviving = torch.cat([surviving_global_labeled, unknown_global_idx])
    all_surviving_sorted, _ = all_surviving.sort()

    # 5. Rebuild graph via PyG subgraph utility
    new_edge_index, _ = subgraph(
        all_surviving_sorted,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )

    # 6. Rebuild feature matrix and labels
    new_x = data.x[all_surviving_sorted]
    new_y = data.y[all_surviving_sorted]

    return Data(x=new_x, edge_index=new_edge_index, y=new_y)
```

### Key Details

- `labeled_global_idx[sample_indices_]` is the critical mapping step. `sample_indices_` are LOCAL indices into `X_labeled` (the labeled-only array passed to `fit_resample`), NOT indices into the full snapshot.
- `subgraph(subset, edge_index, relabel_nodes=True)` handles both edge filtering AND node index remapping in one call. Do not implement this manually.
- Sorting `all_surviving` before passing to `subgraph` is not required by the API but is good practice for determinism.
- After reconstruction: `new_y` still contains label=2 for unknown nodes — the training loop must use `(new_y != 2)` as the loss mask.

---

## 3. Method-by-Method Notes

### ENN (Default — D-06)

- **imblearn class:** `EditedNearestNeighbours`
- **Mode:** Noise removal — removes licit nodes whose 3 nearest feature-space neighbors disagree with their label
- **Effect:** Mild undersampling; does NOT force 1:1 ratio. Typically removes 20-40% of majority class.
- **Reconstruction:** Use `sample_indices_` directly.
- **Graph semantics:** Removes feature-space-ambiguous licit nodes; edges updated via induced subgraph.
- **Known outcome:** Best performer in the reference experiments (hence the default per D-06).

### Edge Pruning (Custom — Worse than ENN)

- **imblearn class:** None — custom implementation required.
- **Algorithm:** ENN-style noise removal using GRAPH neighbors instead of feature-space k-NN.
- **For each majority-class (licit) labeled node n:**
  1. Find all nodes j adjacent to n in the snapshot edge_index (both directions for undirected).
  2. Filter to labeled neighbors only (j where y[j] != 2).
  3. If `len(labeled_neighbors) == 0`: keep n (no info to make decision).
  4. If majority of labeled neighbor labels disagree with y[n]: mark n for removal.
- **Returns:** Set of surviving labeled node global indices. Unknowns always appended.
- **Implementation note:** Minority-class nodes (illicit=0) are NEVER removed — only licit nodes checked. This matches ENN's `sampling_strategy='not minority'` behavior.
- **Known outcome:** Worse than feature-space ENN — noted in CONTEXT.md as a comparison baseline.

```python
def _edge_pruning(data: Data, y_full: torch.Tensor) -> list[int]:
    """Return global indices of nodes surviving edge-pruning."""
    N = data.num_nodes
    src = data.edge_index[0].tolist()
    dst = data.edge_index[1].tolist()

    # Build adjacency list
    adj = {i: set() for i in range(N)}
    for s, d in zip(src, dst):
        adj[s].add(d)
        adj[d].add(s)  # treat as undirected

    minority_class = 0  # illicit
    majority_class = 1  # licit

    surviving = []
    for n in range(N):
        label = y_full[n].item()
        if label == 2:       # unknown: always keep
            surviving.append(n)
        elif label == minority_class:  # illicit: always keep
            surviving.append(n)
        else:                # licit: apply ENN-on-graph rule
            labeled_nbr_labels = [
                y_full[j].item() for j in adj[n] if y_full[j].item() != 2
            ]
            if len(labeled_nbr_labels) == 0:
                surviving.append(n)  # no labeled neighbors -> keep
            elif sum(l != label for l in labeled_nbr_labels) > len(labeled_nbr_labels) / 2:
                pass  # majority disagree -> remove
            else:
                surviving.append(n)
    return surviving
```

### NearMiss-1

- **imblearn class:** `NearMiss(version=1)`
- **Mode:** Prototype selection — keeps licit nodes whose average distance to the 3 nearest illicit nodes is smallest (most boundary-proximate licit nodes).
- **Effect:** Aggressive — reduces licit to match illicit count (1:1 ratio).
- **Reconstruction:** Use `sample_indices_`.
- **Required guard:** Clamp `n_neighbors = min(3, n_illicit_labeled)`.
- **Known outcome:** Worse than ENN (noted in CONTEXT.md).

### NearMiss-2

- **imblearn class:** `NearMiss(version=2)`
- **Mode:** Prototype selection — keeps licit nodes whose average distance to the 3 FARTHEST illicit nodes is smallest.
- **Effect:** Same aggressive 1:1 balancing as NearMiss-1, different selection strategy.
- **Reconstruction:** Use `sample_indices_`.
- **Required guard:** Same n_neighbors clamp as NearMiss-1.
- **Known outcome:** Worse than ENN (noted in CONTEXT.md).

### ClusterCentroids

- **imblearn class:** `ClusterCentroids(voting='hard', random_state=42)`
- **Mode:** Prototype generation — clusters majority class and replaces with representative real samples.
- **Effect:** Reduces licit to match illicit count (1:1 ratio).
- **Reconstruction:** No `sample_indices_`. Use row-matching after `fit_resample()` to recover surviving indices. [VERIFIED: local test]
- **MUST use `voting='hard'`:** Default `voting='auto'` selects `'soft'` for dense arrays, producing SYNTHETIC centroid coordinates that cannot be mapped back to real graph nodes. [VERIFIED: local test]
- **Known outcome:** Worse than ENN (noted in CONTEXT.md).

---

## 4. Existing Pipeline Hook Points

### Snapshot Data Flow

The existing data pipeline is [VERIFIED: projects/src/data/preprocessor.py]:

```
load_snapshots(filter_unknown=False)   # D-05: keep all nodes including unknowns
    -> list of 49 dicts, each with:
        edge_index: LongTensor (2, E_t)
        x:          FloatTensor (N_t, 165)   ← preprocessor returns 165 feature cols
        y:          LongTensor (N_t,)         0=illicit, 1=licit, 2=unknown
        mask:       BoolTensor (N_t,)         all True when filter_unknown=False
        timestep:   int
        split:      "train" | "test"
```

**Important:** `load_snapshots` returns `x` with 165 feature columns (the "time step" column is dropped during preprocessing), NOT 166. Verify the actual column count when building the GCN.

**Hook point:** In `train_gcn_skip.py`, immediately after constructing each snapshot `Data` object and BEFORE the forward pass:

```python
for snap in train_snapshots:
    data = Data(x=snap['x'], edge_index=snap['edge_index'], y=snap['y'])
    if sampler_name != 'none':
        data = apply_sampling(data, sampler)  # <-- hook point
    out = model(data)
    loss = F.cross_entropy(out[data.y != 2], data.y[data.y != 2])
```

Sampling is applied ONLY to training snapshots (timesteps 1-34). Test snapshots (timesteps 35-49) are NEVER sampled — evaluate on original unmodified test graphs.

### Training Script Pattern

Based on `train_kd.py` [VERIFIED: projects/src/train_kd.py]:

```python
# train_gcn_skip.py follows the same pattern:
model = GCN(num_node_features=..., hidden_channels=[64], use_skip=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(500):
    model.train()
    for snap in train_snapshots:
        data = build_data(snap)          # construct Data object
        data = apply_sampling(data, s)   # apply sampler (no-op for 'none')
        optimizer.zero_grad()
        out = model(data, return_logits=True)
        loss = F.cross_entropy(out[data.y != 2], data.y[data.y != 2])
        loss.backward()
        optimizer.step()
    
    if epoch % 50 == 0:
        evaluate(model, test_snapshots)

# After training:
test_illicit_f1 = evaluate_all(model, test_snapshots)
# Append row to sampling_comparison.csv
```

### Evaluation Pattern

From `train_kd.py` and reference `results_skip.txt` [VERIFIED]:
- Illicit F1: `f1_score(y_true, y_pred, pos_label=0, average='binary')`
- Also capture: precision, recall for the illicit class
- Evaluate on each test snapshot independently; aggregate by macro-averaging across timesteps or report final epoch metrics.

### Reference Benchmark Numbers

From `reference/EvolveGCNFai/results_skip.txt` at epoch 499 [VERIFIED: file read]:
- TF GCN-Skip baseline (no sampling, FILTER_UNKNOWN=False):
  - Test Illicit F1: ~0.364
  - Test Precision: ~0.298
  - Test Recall: ~0.468

These serve as sanity-check targets for the PyTorch reproduction.

---

## 5. Dependencies

### Already in pyproject.toml (pre-existing) [VERIFIED: pyproject.toml]

| Package | Constraint | Installed |
|---------|-----------|-----------|
| `torch` | `>=2.10.0` | `2.11.0+cu130` |
| `torch-geometric` | `>=2.7.0` | `2.7.0` |
| `scikit-learn` | `>=1.8.0` | `1.8.0` |
| `numpy` | `>=2.4.2` | present |

### Added During Research [VERIFIED: uv add + pyproject.toml]

| Package | Constraint | Version |
|---------|-----------|---------|
| `imbalanced-learn` | `>=0.14.1` | `0.14.1` |

`imbalanced-learn 0.14.1` also pulled in `sklearn-compat==0.1.5` as a transitive dependency (compatibility shim).

### Installation Command (if needed for fresh env)

```bash
uv add imbalanced-learn
```

No additional dependencies are needed. Edge Pruning is custom Python/PyTorch — no extra packages.

### pyproject.toml Change Required

The `imbalanced-learn>=0.14.1` line was already added by `uv add` during research. The planner should verify this is committed. No other pyproject.toml changes are needed.

---

## 6. Validation Strategy

### Class Distribution Verification

After each `apply_sampling()` call, assert the distribution is as expected:

```python
from collections import Counter

def check_sampling(y_before: np.ndarray, y_after: np.ndarray, method: str):
    before = Counter(y_before.tolist())
    after = Counter(y_after.tolist())
    print(f"[{method}] Before: {dict(before)} -> After: {dict(after)}")
    # ENN: illicit count unchanged, licit count <= original
    # NearMiss: illicit == licit (1:1)
    # CC: illicit == licit (1:1)
```

### Edge Consistency Verification

After rebuilding the Data object, verify:
1. `new_edge_index.max() < new_data.num_nodes` — no out-of-range indices
2. All source and destination nodes in `new_edge_index` exist in the new node set
3. `new_data.x.shape[0] == new_data.y.shape[0]` — feature/label count matches

```python
def verify_data_object(data: Data):
    N = data.num_nodes
    assert data.x.shape[0] == N
    assert data.y.shape[0] == N
    if data.edge_index.shape[1] > 0:
        assert data.edge_index.max() < N, f"edge_index out of range: max={data.edge_index.max()}, N={N}"
```

### Sampler Smoke Test (unit test)

A minimal smoke test for `sampler.py` that does not require the full Elliptic dataset:

```python
# tests/test_sampler.py
import torch, numpy as np
from torch_geometric.data import Data

def make_mini_snapshot(seed=0):
    """5 illicit (label=0), 15 licit (label=1), 5 unknown (label=2)"""
    torch.manual_seed(seed); np.random.seed(seed)
    N = 25
    x = torch.randn(N, 10)
    y = torch.tensor([0]*5 + [1]*15 + [2]*5)
    ei = torch.tensor([[0,1,2,5,6,20,21],[1,2,3,6,7,21,22]], dtype=torch.long)
    return Data(x=x, edge_index=ei, y=y)

def test_enn_keeps_unknowns():
    from projects.src.data.sampler import apply_sampling
    data = make_mini_snapshot()
    result = apply_sampling(data, 'enn')
    assert (result.y == 2).sum() == 5, "unknowns must all survive"
    assert (result.y == 0).sum() == 5, "illicit must all survive (ENN only removes majority)"
    verify_data_object(result)

def test_all_samplers_smoke():
    for name in ['enn', 'edge_pruning', 'cluster_centroid', 'nearmiss1', 'nearmiss2']:
        data = make_mini_snapshot()
        result = apply_sampling(data, name)
        verify_data_object(result)
        assert (result.y == 2).sum() == 5, f"{name}: unknowns must survive"
```

### Results CSV Verification

After a full training run:
1. `sampling_comparison.csv` exists at `projects/results/sampling_comparison.csv`
2. Contains one row per sampler with at minimum: `method`, `test_illicit_f1`, `test_precision`, `test_recall`
3. ENN row's `test_illicit_f1` is plausibly in the 0.35-0.50 range (reference TF: ~0.36-0.45 at various epochs) [ASSUMED — will vary by PyTorch initialization and hyperparameters]

---

## 7. Implementation Risks and Pitfalls

### Pitfall 1: ClusterCentroids synthetic points break graph reconstruction

**What goes wrong:** Default `voting='auto'` on dense numpy arrays uses `'soft'` mode, producing synthetic cluster centroid coordinates. These are not real nodes and cannot be matched to graph positions, making edge reconstruction impossible.

**How to avoid:** Always instantiate as `ClusterCentroids(voting='hard', random_state=42)`. [VERIFIED: local test]

**Warning sign:** If `X_res` rows do not match any row in `X_labeled` exactly.

### Pitfall 2: NearMiss crashes on small minority classes

**What goes wrong:** `NearMiss(n_neighbors=3)` raises `ValueError: Expected n_neighbors <= n_samples_fit` when `n_illicit < n_neighbors`.

**How to avoid:** Before constructing NearMiss, compute `n_min = (y_labeled == 0).sum()` and use `n_neighbors = min(3, n_min)`. [VERIFIED: local test]

**Warning sign:** `ValueError: Expected n_neighbors <= n_samples_fit` in training loop.

### Pitfall 3: sample_indices_ is local, not global

**What goes wrong:** `enn.sample_indices_` contains indices 0..N_labeled-1 (into the `X_labeled` array), NOT 0..N_total-1. Using them directly to index `data.x` or `data.edge_index` gives wrong results.

**How to avoid:** Always map via `labeled_global_idx[sample_indices_]` to get global node indices. [VERIFIED: local test with explicit mapping verification]

### Pitfall 4: Sampling test snapshots

**What goes wrong:** Applying the sampler to test-split snapshots contaminates the evaluation — the model is never tested on real-world imbalanced data.

**How to avoid:** Apply `apply_sampling()` ONLY inside the training loop (train snapshots: timesteps 1-34). Test snapshots (35-49) are passed to the model unmodified for evaluation.

### Pitfall 5: filter_unknown=True removes unknowns before sampling can keep them

**What goes wrong:** Calling `load_snapshots(filter_unknown=True)` removes unknown nodes before the sampling hook. This is inconsistent with D-05 (unknowns should always be present in the graph for message passing).

**How to avoid:** Always use `load_snapshots(filter_unknown=False)` in `train_gcn_skip.py`. The sampling module handles the separation internally.

### Pitfall 6: Feature column count (165 vs 166)

**What goes wrong:** Documentation and the GCN model stub mention 166 features. The preprocessor drops the "time step" column, yielding 165 feature columns in `x`. Initializing `GCN(num_node_features=166, ...)` will fail.

**How to avoid:** Use `data.x.shape[1]` at runtime to set `num_node_features`, or verify with:
```python
snaps = load_snapshots(filter_unknown=False)
print(snaps[0]['x'].shape)  # expect (N, 165)
```
[ASSUMED — the preprocessor code shows "time step" column dropped, but the exact column count in the preprocessed output should be confirmed by the implementer at runtime.]

### Pitfall 7: Edge Pruning on disconnected nodes

**What goes wrong:** If a majority-class node has NO labeled graph neighbors (it connects only to unknown nodes), the edge-pruning rule has no information. Silently keeping such nodes is the correct behavior (consistent with ENN's behavior when no same-timestep neighbors exist).

**How to avoid:** Explicitly check `len(labeled_neighbors) == 0` and keep the node in that case.

### Pitfall 8: results/ directory does not exist

**What goes wrong:** `open('projects/results/sampling_comparison.csv', 'w')` raises `FileNotFoundError` if the directory does not exist.

**How to avoid:** Create the directory in `train_gcn_skip.py`:
```python
import os
os.makedirs('projects/results', exist_ok=True)
```

---

## Validation Architecture

Nyquist validation is enabled (`nyquist_validation: true` in `.planning/config.json`). [VERIFIED: config.json]

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (NOT yet installed) |
| Config file | none — Wave 0 task must add `pytest` to `pyproject.toml` dev deps |
| Quick run command | `uv run pytest tests/test_sampler.py -x -q` |
| Full suite command | `uv run pytest tests/ -q` |

pytest is not currently installed in the venv (`pytest --version` fails). [VERIFIED: local check]

### Phase Requirements to Test Map

| Req | Behavior | Test Type | Command | File Exists? |
|-----|----------|-----------|---------|-------------|
| D-03 | Sampling applied per-timestep in feature space | unit | `uv run pytest tests/test_sampler.py::test_enn_keeps_unknowns -x` | Wave 0 |
| D-04 | Incident edges removed via induced subgraph | unit | `uv run pytest tests/test_sampler.py::test_edge_consistency -x` | Wave 0 |
| D-05 | Unknown nodes always survive sampling | unit | `uv run pytest tests/test_sampler.py::test_all_samplers_smoke -x` | Wave 0 |
| D-06 | All 5 methods importable via get_sampler() | unit | `uv run pytest tests/test_sampler.py::test_get_sampler_factory -x` | Wave 0 |
| D-07 | sampler.py module exists with get_sampler factory | import smoke | `uv run python -c "from projects.src.data.sampler import get_sampler"` | Wave 0 |
| D-08 | Illicit F1 computed correctly | unit | `uv run pytest tests/test_sampler.py::test_metrics -x` | Wave 0 |
| D-09 | CSV written after training run | integration | manual / spot-check output file | manual |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/test_sampler.py -x -q`
- **Per wave merge:** `uv run pytest tests/ -q`
- **Phase gate:** All tests green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/__init__.py` — test package init
- [ ] `tests/test_sampler.py` — covers D-03, D-04, D-05, D-06, D-07
- [ ] `uv add --dev pytest` — pytest not installed; must be added before tests can run
- [ ] `projects/results/` directory — must be created (or created at runtime by training script)

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Reference TF Illicit F1 of 0.36-0.46 is a valid sanity-check target range for the PyTorch reproduction | Section 6 (Results verification) | If PyTorch GCN-skip trains differently (dropout=0.5 vs TF 0.3, static vs snapshot training), actual numbers may differ. Low impact — it's a sanity check, not a hard requirement. |
| A2 | Elliptic has ≥4 labeled nodes per timestep (ENN n_neighbors guard threshold) | Section 1 (NearMiss pitfall) | If a timestep has <4 labeled nodes, ENN also fails. Need defensive min-neighbor check for ENN too. |
| A3 | Preprocessor yields 165 feature columns (not 166) | Section 7 Pitfall 6 | If column count is actually 166, `GCN(num_node_features=165)` fails. Verify with `snaps[0]['x'].shape[1]` at implementation time. |

---

## Sources

### Primary (HIGH confidence)
- Context7 `/scikit-learn-contrib/imbalanced-learn` — ENN, NearMiss, ClusterCentroids API, sample_indices_, fit_resample behavior
- `https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.EditedNearestNeighbours.html` — constructor signature, attributes, multiclass support
- `https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.NearMiss.html` — constructor signature, version semantics, attributes
- `https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html` — voting parameter, synthetic vs real samples
- Local test execution in project venv (Python 3.11, imbalanced-learn 0.14.1) — all API behaviors verified
- PyG `subgraph()` docstring via `help()` in venv (PyG 2.7.0) — relabel_nodes behavior verified

### Secondary (MEDIUM confidence)
- `projects/src/data/preprocessor.py` — snapshot structure, column counts, unknown node handling
- `projects/src/models/gcn.py` — GCN(use_skip=True) skip connection implementation
- `projects/src/train_kd.py` — training script pattern
- `reference/EvolveGCNFai/train_gcn_skip.py` — hyperparameters (NUM_EPOCH=500, LR=1e-3)
- `reference/EvolveGCNFai/results_skip.txt` — TF baseline metrics for sanity-checking

### Tertiary (LOW confidence)
- None.

---

## RESEARCH COMPLETE
