---
phase: 01-infrastructure-modularization
reviewed: 2026-05-03T00:00:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - projects/src/__init__.py
  - projects/src/data/__init__.py
  - projects/src/models/__init__.py
  - projects/src/utils/__init__.py
  - projects/src/data/downloader.py
  - projects/src/data/preprocessor.py
  - projects/src/data/dataset.py
  - projects/src/data/loader.py
  - projects/src/main.py
  - projects/src/models/gcn.py
  - projects/src/models/evolve_gcn.py
findings:
  critical: 3
  warning: 5
  info: 4
  total: 12
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-05-03T00:00:00Z
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Reviewed the full data pipeline and model stub layer introduced in this phase. The four `__init__.py` files are empty placeholders (acceptable for this stage). The model stubs (`gcn.py`, `evolve_gcn.py`) are well-structured and correctly raise `NotImplementedError`. The core concerns concentrate in `main.py`, `preprocessor.py`, and `downloader.py`.

Three blockers were found: `main.py` imports a function that does not exist in `downloader.py` and also passes a wrong keyword argument, guaranteeing a runtime crash when `--download` is used. The `load_snapshots` normalizer fits a fresh `StandardScaler` independently per snapshot — including test-split snapshots — leaking test-time statistics into feature scaling. Five warnings cover a security-relevant use of `torch.load(weights_only=False)`, behavioral asymmetry between the two public preprocessor functions, a deprecated PyG import, non-deterministic node ordering, and a mask-attribute naming inconsistency between the two snapshot-facing APIs.

---

## Critical Issues

### CR-01: `main.py` imports non-existent function `download_dataset`

**File:** `projects/src/main.py:38`
**Issue:** `_download()` attempts `from data.downloader import download_dataset`. The only public function in `downloader.py` is `download_elliptic_dataset` (defined at line 27). This raises `ImportError` at runtime whenever `--download` is passed.

**Fix:**
```python
# main.py line 38–41 — correct the import and the call
from data.downloader import download_elliptic_dataset  # was: download_dataset

print("[main] Downloading Elliptic dataset from Kaggle …")
download_elliptic_dataset(dest_dir=Path(_PROJECTS_DIR / "data" / "raw"), force=False)
```

---

### CR-02: Wrong keyword argument passed to downloader

**File:** `projects/src/main.py:41`
**Issue:** `main.py` calls `download_dataset(dest=..., force=False)`. Even if the function name were corrected (CR-01), the keyword `dest` does not exist; the parameter is `dest_dir`. Calling with `dest=...` would either be silently ignored (passing an unexpected kwarg) or raise `TypeError`, depending on the Python version and whether `**kwargs` is accepted. `download_elliptic_dataset` does not accept `**kwargs`, so this raises `TypeError`.

**Fix:**
```python
download_elliptic_dataset(dest_dir=Path(_PROJECTS_DIR / "data" / "raw"), force=False)
```

---

### CR-03: `load_snapshots` fits `StandardScaler` on test-split snapshots

**File:** `projects/src/data/preprocessor.py:355-357`
**Issue:** Inside the per-timestep loop, when `normalize=True`, a brand-new `StandardScaler` is instantiated and `fit_transform`-ed on each snapshot independently — including snapshots where `t >= TEST_START_TIMESTEP` (the test set). This means test-snapshot features are normalized using statistics derived from test data, not from the training distribution. The correct approach mirrors what `load_static_graph` does: fit once on the training split and apply that same scaler to the test split.

```python
# current (lines 355–357) — WRONG for test snapshots
if normalize:
    scaler = StandardScaler()
    x_np = scaler.fit_transform(x_np)   # fits on every snapshot, including test
```

**Fix:** Fit the scaler once across all training snapshots and reuse it:
```python
# Compute global train statistics outside the loop
if normalize:
    train_rows = merged[merged["time"] < TEST_START_TIMESTEP]
    feat_cols_list = [c for c in merged.columns if c not in ("nid", "label", "time")]
    global_scaler = StandardScaler()
    global_scaler.fit(train_rows[feat_cols_list].values.astype(np.float32))

for t in range(1, NUM_TIMESTEPS + 1):
    ...
    if normalize:
        x_np = global_scaler.transform(x_np)
```

---

## Warnings

### WR-01: `torch.load` called with `weights_only=False`

**File:** `projects/src/data/dataset.py:79`
**Issue:** `torch.load(self.processed_paths[0], weights_only=False)` disables PyTorch's safe-deserialization guard. If the `.pt` file at `processed_paths[0]` is replaced or tampered with, arbitrary Python code can execute during deserialization. `weights_only=False` is required here only because `InMemoryDataset.collate` serializes non-tensor metadata alongside tensors. PyTorch ≥ 2.0 recommends keeping `weights_only=True` and using `torch.serialization.add_safe_globals` for any custom classes that need to survive the load.

**Fix:** Where possible, prefer `weights_only=True`. If collated slices require custom types, register them explicitly:
```python
# Option A — explicit safe globals (PyTorch ≥ 2.0)
import torch.serialization
torch.serialization.add_safe_globals([...])  # add any PyG classes needed
self.data, self.slices = torch.load(self.processed_paths[0], weights_only=True)

# Option B — document the risk and restrict file permissions
# (acceptable if processed/ is under the project's own control)
```

---

### WR-02: `load_snapshots` does not remove unknown nodes when `filter_unknown=True`

**File:** `projects/src/data/preprocessor.py:362-365`
**Issue:** `load_static_graph` (line 244) physically removes unknown-labelled nodes from the tensors when `filter_unknown=True`. `load_snapshots` does not: it only sets `mask_t = (labels != 2)` but leaves nodes with `label == 2` in `x_t` and `y_t`. Callers that iterate snapshot data objects without applying the mask will silently encounter the spurious label value `2` and may train or evaluate on unknown nodes. The parameter name `filter_unknown` implies removal; the mask-only behavior is undocumented.

**Fix:** Either filter the nodes before building tensors (mirrors `load_static_graph`):
```python
if filter_unknown:
    nodes_t = nodes_t[nodes_t["label"] != 2].reset_index(drop=True)
    nodes_t["local_idx"] = nodes_t.index  # rebuild local index
    id_to_local_t = nodes_t.set_index("nid")["local_idx"]
    # re-filter edges for updated id_to_local_t ...
    mask_t = torch.ones(len(nodes_t), dtype=torch.bool)  # all remaining are known
```
Or rename the parameter to `mask_unknown` and clearly document the mask-only semantics.

---

### WR-03: `EllipticSnapshotDataset` and `get_snapshot_loaders` expose inconsistent mask attribute names

**File:** `projects/src/data/dataset.py:185` and `projects/src/data/loader.py:68,76`
**Issue:** `EllipticSnapshotDataset` wraps each snapshot into a `Data` object with a `mask` attribute (line 185). `get_snapshot_loaders` in `loader.py` creates `Data` objects from the same snapshots but uses `train_mask` for training items (line 68) and `test_mask` for test items (line 76). Any downstream code written for `EllipticSnapshotDataset` will fail when switched to `get_snapshot_loaders`-produced batches because the attribute name differs.

**Fix:** Standardize on a single attribute name across both APIs. Using a single `mask` attribute is simpler and does not require the caller to know which split they are on:
```python
# In get_snapshot_loaders (loader.py) — use 'mask' for both branches
data = Data(x=snap["x"], edge_index=edge_index, y=snap["y"], mask=snap["mask"])
```

---

### WR-04: `DataLoader` imported from deprecated `torch_geometric.data`

**File:** `projects/src/data/loader.py:19`
**Issue:** `from torch_geometric.data import Data, DataLoader` — `torch_geometric.data.DataLoader` was deprecated in PyG 2.0 and removed in later versions. The canonical import is `from torch_geometric.loader import DataLoader`. The current code has `# type: ignore[attr-defined]` acknowledging this, but the deprecated symbol may not exist at all in the target PyG version, causing an `ImportError` at module load time.

**Fix:**
```python
from torch_geometric.loader import DataLoader
```

---

### WR-05: Non-deterministic node ordering in `_build_node_index`

**File:** `projects/src/data/preprocessor.py:99-110`
**Issue:** `all_nodes` is constructed as a Python `set` union of four sets, then converted to `list` before creating the DataFrame. In CPython, integer sets have a predictable but PYTHONHASHSEED-dependent iteration order when `PYTHONHASHSEED` is not fixed. While integers hash to themselves in CPython, relying on set ordering guarantees that the integer-to-node-ID mapping varies across Python interpreter invocations on some platforms and Python versions. This causes the node-feature matrix rows to be assigned different node IDs on different runs, breaking checkpoint reproducibility.

**Fix:**
```python
all_nodes_sorted = sorted(
    set(df_edge["txId1"])
    .union(set(df_edge["txId2"]))
    .union(set(df_class["txId"]))
    .union(set(df_features["id"]))
)
nodes_df = pd.DataFrame(all_nodes_sorted, columns=["id"]).reset_index()
```

---

## Info

### IN-01: No-op `rename` in `_build_node_index`

**File:** `projects/src/data/preprocessor.py:108`
**Issue:** `.rename(columns={"index": "index"})` renames the column `"index"` to `"index"` — a no-op. The column already has the name `"index"` after `.reset_index()`. Dead code that adds noise without effect.

**Fix:** Remove the `.rename(...)` call entirely:
```python
nodes_df = (
    pd.DataFrame(list(all_nodes), columns=["id"])
    .reset_index()
)
```

---

### IN-02: Redundant `reset_index` in `load_static_graph`

**File:** `projects/src/data/preprocessor.py:244,247`
**Issue:** When `filter_unknown=True`, line 244 already calls `.reset_index(drop=True)`. Line 247 then calls `.reset_index(drop=True)` again unconditionally. When `filter_unknown=False`, the first reset never runs, but the DataFrame's index is already a clean `RangeIndex` from `_build_node_label_df`/`_merge_features`, so the second reset is still a no-op.

**Fix:** Remove the unconditional reset on line 247 and fold it into the conditional branch:
```python
if filter_unknown:
    merged = merged[merged["label"] != 2].reset_index(drop=True)

merged["local_idx"] = merged.index  # line 248 — no second reset needed
```

---

### IN-03: Documented feature dimension is incorrect in multiple files

**File:** `projects/src/data/preprocessor.py:19`, `projects/src/data/dataset.py:43,64,154`, `projects/src/models/gcn.py:24`
**Issue:** Several docstrings state the node feature matrix has shape `(N, 166)`. The module-level comment in `preprocessor.py` (line 19) correctly notes that the time-step column is dropped, yielding 93 + 72 = **165** features. All downstream docstring examples citing `166` are wrong and will mislead users setting `num_node_features`.

**Fix:** Replace `166` with `165` in all docstring feature-shape examples and update `gcn.py`'s example argument comment from `(e.g. 166)` to `(e.g. 165)`.

---

### IN-04: `import torch` inside function body in `_print_snapshot_stats`

**File:** `projects/src/main.py:95`
**Issue:** `import torch` appears inside `_print_snapshot_stats()` at line 95, after `torch` is already implicitly available through `train_loader.dataset` objects. This import is never used; all the per-element computations are plain Python `int(...)` calls and the `==` operator on PyTorch tensors does not require an explicit `torch` reference. It is dead code.

**Fix:** Remove the `import torch` statement at line 95 of `main.py`.

---

_Reviewed: 2026-05-03T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
