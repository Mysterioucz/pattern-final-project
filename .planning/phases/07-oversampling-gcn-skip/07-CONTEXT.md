# Phase 7: Oversampling for GCN-Skip - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Evaluate five class-imbalance sampling methods (ENN, Edge Pruning, Cluster Centroid, NearMiss-1, NearMiss-2) applied to GCN with skip connections on the Elliptic Bitcoin dataset. Implement all methods in the PyTorch/PyG `projects/src/` codebase as a systematic ablation study. ENN is the known winner and runs by default; other methods are opt-in for reproducibility. Temporal GNNs, feature-based KD, and other architectures are out of scope for this phase.

</domain>

<decisions>
## Implementation Decisions

### Codebase Target
- **D-01:** Target the existing PyTorch/PyG `projects/src/` codebase — consistent with v1.0 modular framework.
- **D-02:** Extend the existing `GCN` class with `use_skip=True` (already implemented in `projects/src/models/gcn.py`). Add a new `projects/src/train_gcn_skip.py` training script alongside `train_kd.py`.

### Sampling Application Point
- **D-03:** Apply sampling **per-timestep**, in node feature space. For each timestep subgraph: extract `(X_nodes, y_labels)` for labeled nodes, run the sampler, keep surviving node indices, rebuild the PyG `Data` object with surviving nodes and their induced edges.
- **D-04:** When a node is removed by the sampler, **remove its incident edges** as well (induced subgraph). Filter `edge_index` to keep only edges where both source and destination survive sampling.
- **D-05:** Unknown-class nodes are **excluded from sampling decisions** — they always pass through untouched. The sampler operates only on labeled nodes (illicit / licit). Consistent with the reference codebase's `FILTER_UNKNOWN=False` behavior.

### Sampling Method Implementation
- **D-06:** Implement all 5 methods, but ENN is the **default** (active without any flag). The other 4 methods (Edge Pruning, Cluster Centroid, NearMiss-1, NearMiss-2) are **opt-in** via a `--sampler` CLI flag or config key. This lets standard runs use the winner without extra overhead.
- **D-07:** All methods live in a single **`projects/src/data/sampler.py`** module with a `get_sampler(name)` factory function. Supported names: `'enn'`, `'edge_pruning'`, `'cluster_centroid'`, `'nearmiss1'`, `'nearmiss2'`. The training script selects the sampler via config/argument.

### Results Reporting
- **D-08:** Primary comparison metric is **Illicit F1 score** (consistent with v1.0 benchmark and reference `results_skip.txt`).
- **D-09:** Results are stored as `projects/results/sampling_comparison.csv` with per-method metrics (at minimum: method name, test Illicit F1, test Precision, test Recall). The training script prints a human-readable summary table at the end of each run.

### Claude's Discretion
- Number of training epochs and hyperparameters (learning rate, dropout) — match reference values where available (`NUM_EPOCH=500`, `LEARNING_RATE=1e-3`).
- ENN and NearMiss hyperparameters (`n_neighbors`, `version`) — use `imbalanced-learn` defaults unless the reference experiments suggest otherwise.
- Edge Pruning implementation detail — apply ENN logic on each node's graph-neighborhood (direct neighbors as the k-NN set) rather than feature-space k-NN.
- CSV column schema and table formatting.

</decisions>

<specifics>
## Specific Ideas

- Reference implementation at `reference/EvolveGCNFai/train_gcn_skip.py` uses `CLASS_WEIGHTS = [0.7, 0.29, 0.01]` and `GCNTwoLayersSkipConnection`. Use these as a baseline comparison point (not the target implementation).
- "Edge Pruning" = ENN applied at the graph-edge level (direct neighbors as k-NN), explicitly noted as **worse** than feature-space ENN.
- NearMiss-1 and NearMiss-2 are both noted as **worse** than ENN.
- Results files from reference TF experiments (`reference/EvolveGCNFai/results_skip.txt`, `results_skip_filtered.txt`) serve as sanity-check benchmarks for the PyTorch reproduction.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing model with skip connection
- `projects/src/models/gcn.py` — `GCN` class with `use_skip=True`; the skip branch is `nn.Linear(num_node_features, 2)` added to the final output. This is the model to use.

### Reference GCN-skip training script (TensorFlow — for reference only)
- `reference/EvolveGCNFai/train_gcn_skip.py` — Original experiment: class weights `[0.7, 0.29, 0.01]`, 500 epochs, Adam lr=1e-3. Do NOT port TF code; use as benchmark context.
- `reference/EvolveGCNFai/results_skip.txt` — TF baseline results for sanity-checking.
- `reference/EvolveGCNFai/results_skip_filtered.txt` — TF results with FILTER_UNKNOWN=True.

### Data pipeline (existing)
- `projects/src/data/dataset.py` — PyG dataset loading for Elliptic.
- `projects/src/data/loader.py` — Dataloader utilities.
- `projects/src/data/preprocessor.py` — Preprocessing pipeline.

### KD training script (structural pattern to follow)
- `projects/src/train_kd.py` — Pattern for training script structure in projects/src/.

### Prior phase decisions
- `.planning/phases/02-model-implementation/02-CONTEXT.md` — GCN architecture decisions (Teacher: 64 hidden, Student: compressed).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `projects/src/models/gcn.py` — `GCN(use_skip=True)` already implemented. No new model class needed.
- `projects/src/data/dataset.py` — Existing PyG Data loading pipeline to hook into.
- `projects/src/train_kd.py` — Training loop structure (epoch loop, metric logging) to replicate/adapt.

### Established Patterns
- Training scripts in `projects/src/` are standalone (`train_kd.py`, not as library functions).
- The Elliptic dataset processes labeled timesteps as independent PyG `Data` objects.
- Illicit F1 is the primary reported metric (consistent with v1.0 evaluations).

### Integration Points
- `projects/src/data/sampler.py` (new) connects between the data loader and the GCN-skip training script.
- `projects/results/sampling_comparison.csv` (new) is the output artifact.
- The `--sampler` flag or config key in `train_gcn_skip.py` selects which sampling method runs.

</code_context>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope.

</deferred>

---

*Phase: 07-oversampling-gcn-skip*
*Context gathered: 2026-05-06*
