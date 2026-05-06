"""
Unit tests for projects/src/data/sampler.py — Phase 7.

Tests cover D-03 (per-timestep feature-space sampling), D-04 (incident edges removed),
D-05 (unknown nodes always kept), D-06 (all 5 methods accessible), D-07 (get_sampler factory),
D-08 (Illicit F1 uses pos_label=0).

These tests are written BEFORE the implementation (RED state). They will pass after
Wave 1 implements projects/src/data/sampler.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mini_snapshot(seed: int = 0) -> Data:
    """Synthetic snapshot: 5 illicit (0), 15 licit (1), 5 unknown (2), 25 nodes total.

    Edge list connects: 0-1-2-3 (illicit chain), 5-6-7 (licit chain),
    20-21-22 (unknown chain). Chosen so that after sampling, edge_index
    references are always within the surviving node set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    N = 25
    x = torch.randn(N, 10)
    y = torch.tensor([0] * 5 + [1] * 15 + [2] * 5)
    ei = torch.tensor(
        [[0, 1, 2, 5, 6, 20, 21], [1, 2, 3, 6, 7, 21, 22]],
        dtype=torch.long,
    )
    return Data(x=x, edge_index=ei, y=y)


def verify_data_object(data: Data) -> None:
    """Assert structural integrity of a Data object after sampling."""
    N = data.num_nodes
    assert data.x.shape[0] == N, f"x.shape[0]={data.x.shape[0]} != num_nodes={N}"
    assert data.y.shape[0] == N, f"y.shape[0]={data.y.shape[0]} != num_nodes={N}"
    if data.edge_index.shape[1] > 0:
        assert data.edge_index.max() < N, (
            f"edge_index out of range: max={data.edge_index.max()}, N={N}"
        )


# ---------------------------------------------------------------------------
# D-07: get_sampler factory
# ---------------------------------------------------------------------------

def test_get_sampler_factory():
    """D-07: get_sampler(name) returns non-None object for all imblearn-backed samplers."""
    from projects.src.data.sampler import get_sampler

    for name in ["enn", "nearmiss1", "nearmiss2", "cluster_centroid"]:
        s = get_sampler(name)
        assert s is not None, f"get_sampler('{name}') returned None — expected imblearn object"

    # edge_pruning uses custom implementation; factory returns None as sentinel
    s_ep = get_sampler("edge_pruning")
    assert s_ep is None, "get_sampler('edge_pruning') must return None (custom implementation sentinel)"


def test_get_sampler_invalid_raises():
    """D-07: get_sampler raises ValueError for unknown names."""
    from projects.src.data.sampler import get_sampler

    with pytest.raises(ValueError, match="Unknown sampler"):
        get_sampler("invalid_method_xyz")


# ---------------------------------------------------------------------------
# D-05: Unknown nodes always survive
# ---------------------------------------------------------------------------

def test_enn_keeps_unknowns():
    """D-05 + D-03: ENN keeps all unknown (y==2) nodes; illicit (y==0) nodes never removed."""
    from projects.src.data.sampler import apply_sampling

    data = make_mini_snapshot()
    result = apply_sampling(data, "enn")

    assert (result.y == 2).sum().item() == 5, (
        "ENN must keep all 5 unknown nodes (D-05)"
    )
    assert (result.y == 0).sum().item() == 5, (
        "ENN must keep all illicit nodes (minority class is never removed)"
    )
    verify_data_object(result)


def test_all_samplers_keep_unknowns():
    """D-05: All 5 samplers keep every unknown (y==2) node."""
    from projects.src.data.sampler import apply_sampling

    for name in ["enn", "edge_pruning", "cluster_centroid", "nearmiss1", "nearmiss2"]:
        data = make_mini_snapshot()
        result = apply_sampling(data, name)
        n_unknown = (result.y == 2).sum().item()
        assert n_unknown == 5, (
            f"{name}: expected 5 unknown nodes, got {n_unknown} (D-05)"
        )


# ---------------------------------------------------------------------------
# D-04: Induced subgraph — incident edges removed with dropped nodes
# ---------------------------------------------------------------------------

def test_edge_consistency_after_sampling():
    """D-04: After any sampler, edge_index references only surviving node indices."""
    from projects.src.data.sampler import apply_sampling

    for name in ["enn", "edge_pruning", "cluster_centroid", "nearmiss1", "nearmiss2"]:
        data = make_mini_snapshot()
        result = apply_sampling(data, name)
        verify_data_object(result)  # raises AssertionError if edge_index out of range


# ---------------------------------------------------------------------------
# D-03: Sampling applied per-timestep; returns a new Data object
# ---------------------------------------------------------------------------

def test_apply_sampling_returns_data_object():
    """D-03: apply_sampling returns a torch_geometric.data.Data instance."""
    from projects.src.data.sampler import apply_sampling

    data = make_mini_snapshot()
    result = apply_sampling(data, "enn")
    assert isinstance(result, Data), (
        f"apply_sampling must return Data, got {type(result)}"
    )


def test_apply_sampling_reduces_majority_class():
    """D-03: ENN reduces majority (licit=1) class size; minority (illicit=0) unchanged."""
    from projects.src.data.sampler import apply_sampling

    data = make_mini_snapshot()  # 5 illicit, 15 licit, 5 unknown
    result = apply_sampling(data, "enn")

    n_licit_after = (result.y == 1).sum().item()
    assert n_licit_after <= 15, (
        f"ENN must not increase majority class; got {n_licit_after} licit nodes"
    )


# ---------------------------------------------------------------------------
# D-06: All 5 methods smoke test
# ---------------------------------------------------------------------------

def test_all_samplers_smoke():
    """D-06: All 5 named methods complete without exception and return valid Data."""
    from projects.src.data.sampler import apply_sampling

    for name in ["enn", "edge_pruning", "cluster_centroid", "nearmiss1", "nearmiss2"]:
        data = make_mini_snapshot()
        result = apply_sampling(data, name)
        assert isinstance(result, Data), f"{name}: expected Data, got {type(result)}"
        verify_data_object(result)


# ---------------------------------------------------------------------------
# D-08: Illicit F1 uses pos_label=0 (illicit encoded as 0)
# ---------------------------------------------------------------------------

def test_illicit_f1_uses_pos_label_zero():
    """D-08: Illicit F1 is computed with pos_label=0 (illicit=0 encoding)."""
    from sklearn.metrics import f1_score

    # Perfect illicit predictions
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1]
    f1 = f1_score(y_true, y_pred, pos_label=0, average="binary")
    assert abs(f1 - 1.0) < 1e-6, f"Perfect illicit F1 should be 1.0, got {f1}"

    # All predicted as licit — illicit F1 should be 0
    y_pred_wrong = [1, 1, 1, 1, 1, 1]
    f1_wrong = f1_score(y_true, y_pred_wrong, pos_label=0, average="binary", zero_division=0)
    assert f1_wrong == 0.0, f"Illicit F1 with all-licit predictions should be 0, got {f1_wrong}"


def test_apply_sampling_invalid_raises():
    """Unknown sampler names must raise ValueError via factory path."""
    from projects.src.data.sampler import apply_sampling

    data = make_mini_snapshot()
    with pytest.raises(ValueError, match="Unknown sampler"):
        apply_sampling(data, "invalid_method_xyz")
