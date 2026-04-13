# System Architecture

**Current Pattern:** 
- The project root is currently uninitialized.
- A functional deep learning prototype resides in `reference/EvolveGCN/`. This is essentially a script-based training application representing standard PyTorch/TensorFlow Deep Learning structures with an emphasis on recurrent Graph Convolutional Networks (EvolveGCN).

**Component Layers (Reference Area):**
1. **Data Ingestion (`data_loader.py`):**
   - Implements `EllipticDatasetLoader` to read nodes and edges.
   - Normalizes adjacency matrices and standardizes tensor conversions via SciPy.
2. **Model Definition (`models/`):**
   - Layer Definitions (`models/layers/`): Contains granular pieces of the neural network including `GCNLayer`, `GCNSkipLayer`, `HGRUCell`, and `SummarizeLayer`.
   - Network Definitions: Assembles layers together into structural classes (`EvolveGCN`, `GCNSequential`, `GCNTwoLayersSkipConnection`).
3. **Execution Scripts (`train_*.py`):**
   - E.g., `train_evolve_gcn.py`, `train_gcn.py`.
   - Act as standalone system entry points, coordinating data injection into models and executing the TensorFlow optimization loop for multiple epochs.

**Data Flow:**
- Load Edge Lists via `pandas` and translate to Graphs with `networkx`.
- Pre-process graphs to sparse adjacency matrices.
- Pass features through initialized subclass models.
- Predict output node embeddings over time to perform financial forensics (Anti-Money Laundering logic).
- Output model weights or accuracy metrics.
