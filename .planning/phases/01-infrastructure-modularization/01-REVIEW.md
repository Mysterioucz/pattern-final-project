---
phase: 01-infrastructure-modularization
reviewed: 2026-05-03T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - projects/src/main.py
  - projects/src/data/preprocessor.py
  - projects/src/data/loader.py
findings:
  critical: 1
  warning: 3
  info: 2
  total: 6
status: issues_found
---

# Phase 01: Code Review Report (Gap-Closure Re-Review)

**Reviewed:** 2026-05-03T00:00:00Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

This re-review targets the three files changed to close the gaps identified in the prior review (CR-01/CR-02/CR-03/WR-04). All three prior blockers that touched these files have been resolved:

- **CR-01 + CR-02 (main.py):** `_download()` now correctly imports `download_elliptic_dataset` (line 38) and calls it with `dest_dir=str(...)` (line 41). The `str()` wrapper is safe because `downloader.py` coerces its argument to `Path` internally (line 58 of `downloader.py`).
- **CR-03 (preprocessor.py):** `load_snapshots` now fits `StandardScaler` once on all training-split rows before the loop (lines 331–334) and calls only `scaler.transform()` inside the loop (line 364). The old per-snapshot `fit_transform()` pattern is gone.
- **WR-04 (loader.py):** `DataLoader` is now imported from `torch_geometric.loader` (line 20), not the deprecated `torch_geometric.data`.

One new blocker was introduced by the gap-closure changes. Two warnings and two info items remain unresolved from the prior pass or are newly introduced.

---

## Critical Issues

### CR-01: Illicit-ratio denominator inflated by unknown nodes in `_print_snapshot_stats`

**File:** `projects/src/main.py:90–104`
**Issue:** `train_nodes` (line 90) and `test_nodes` (line 91) sum `d.x.shape[0]` across every snapshot — which includes unknown-labelled nodes (label `2`) because `load_snapshots` with `filter_unknown=True` does **not** physically remove unknown nodes from `x_t`/`y_t`; it only sets a boolean mask. Meanwhile `train_illicit` (lines 97–99) counts `(d.y[d.train_mask] == 0).sum()` — `d.train_mask` is the mask that excludes unknowns, so the numerator covers only known illicit nodes. Dividing a known-only count by an all-nodes total produces a systematically underestimated illicit ratio. The Elliptic dataset has roughly 21 % unknown nodes, so the displayed ratio can be off by up to that fraction.

```python
# Buggy (lines 90–104)
train_nodes = sum(int(d.x.shape[0]) for d in train_loader.dataset)   # includes unknowns
...
train_illicit = sum(
    int((d.y[d.train_mask] == 0).sum()) for d in train_loader.dataset  # excludes unknowns
)
illicit_ratio_train = train_illicit / max(train_nodes, 1)  # wrong denominator
```

**Fix:** Count only masked (known) nodes in the denominator:

```python
train_nodes = sum(
    int(d.train_mask.sum()) for d in train_loader.dataset
)
test_nodes = sum(
    int(d.test_mask.sum()) for d in test_loader.dataset
)
train_edges = sum(int(d.edge_index.shape[1]) for d in train_loader.dataset)
test_edges  = sum(int(d.edge_index.shape[1]) for d in test_loader.dataset)
```

---

## Warnings

### WR-01: Scaler fitted on unknown nodes in `load_snapshots` — inconsistent with `load_static_graph`

**File:** `projects/src/data/preprocessor.py:332`
**Issue:** `load_static_graph` filters unknown nodes (label `2`) from `merged` **before** computing `train_mask_np` and fitting the scaler (lines 243–276); its scaler therefore sees only known-labelled training nodes. `load_snapshots` fits on `merged[merged["time"] < TEST_START_TIMESTEP][feat_cols]` (line 332) **before** any filtering, so the scaler mean/std are computed over all training-split rows including unknowns. Because ~21 % of training nodes are unknown, the two scalers diverge, and models trained via one pathway are subtly miscalibrated relative to the other.

**Fix:** Filter unknowns from `merged` before computing `train_rows`:

```python
if normalize:
    known_train = merged[
        (merged["time"] < TEST_START_TIMESTEP) & (merged["label"] != 2)
    ]
    train_rows = known_train[feat_cols].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(train_rows)
```

---

### WR-02: `load_snapshots` does not physically remove unknown nodes when `filter_unknown=True`

**File:** `projects/src/data/preprocessor.py:369–372`
**Issue:** When `filter_unknown=True`, only `mask_t` is set to `(label != 2)`. Unknown nodes with `label == 2` remain present in `x_t`, `y_t`, and the edge tensors. This conflicts with the parameter name (`filter_unknown` implies removal) and with the behaviour of `load_static_graph`, which physically removes unknown nodes. Any consumer that iterates `d.y` without applying `d.train_mask` / `d.test_mask` will encounter the sentinel label `2`, potentially corrupting loss computation.

**Fix:** Physically remove unknown nodes before building the per-snapshot tensors:

```python
if filter_unknown:
    nodes_t = nodes_t[nodes_t["label"] != 2].reset_index(drop=True)
    nodes_t = nodes_t.copy()
    nodes_t["local_idx"] = nodes_t.index
    id_to_local_t = nodes_t.set_index("nid")["local_idx"]
    # re-filter edges for the updated id_to_local_t ...
    mask_t = torch.ones(len(nodes_t), dtype=torch.bool)
```

---

### WR-03: Stale docstring — return type still cites deprecated `torch_geometric.data.DataLoader`

**File:** `projects/src/data/loader.py:50`
**Issue:** The docstring for `get_snapshot_loaders` reads:

```
train_loader, test_loader  —  ``torch_geometric.data.DataLoader``
```

The actual import was corrected to `torch_geometric.loader.DataLoader` (line 20), but the docstring was not updated. Developers reading the docstring will look in the wrong module.

**Fix:**

```python
Returns:
    train_loader, test_loader  —  ``torch_geometric.loader.DataLoader``
```

---

## Info

### IN-01: Unused `import torch` inside `_print_snapshot_stats`

**File:** `projects/src/main.py:95`
**Issue:** `import torch` appears at line 95 inside `_print_snapshot_stats()`. No `torch.*` symbol is referenced in the function body; all arithmetic is plain Python on native `int` values, and `==` comparisons on `Tensor` objects do not require an explicit `torch` import. Dead import.

**Fix:** Remove line 95 (`import torch`).

---

### IN-02: No-op `rename` in `_build_node_index`

**File:** `projects/src/data/preprocessor.py:108`
**Issue:** `.rename(columns={"index": "index"})` renames the column `"index"` to itself. After `.reset_index()` the column is already named `"index"`. Dead code that adds noise.

**Fix:** Remove the `.rename(...)` call:

```python
nodes_df = (
    pd.DataFrame(list(all_nodes), columns=["id"])
    .reset_index()
)
```

---

_Reviewed: 2026-05-03T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
