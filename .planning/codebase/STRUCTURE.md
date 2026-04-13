# Structural Layout

**Root Directory Organization:**
- `.git/` - Git version control history.
- `.planning/` - (New) GSD working directory tracking the system roadmap, codebase mapping, and plans.
- `reference/` - Isolated container for third-party scripts.
  - `EvolveGCN/` - Full cloned repository containing the ML training pipeline.
    - `models/` - Neural network modules.
      - `layers/` - Reusable layers.
        - `e_gcu_h.py`, `gcn_layer.py`, `gcn_skip_layer.py`, `h_gru.py`, `summarize.py`
      - `evolve_gcn.py`, `gcn_skip.py`, `gcn.py`
    - `data_loader.py` - Manages dataset instantiation.
    - `train_evolve_gcn.py`, `train_gcn_skip.py`, `train_gcn.py` - Core scripts to trigger execution.
    - `utils.py` - Common mathematical manipulations.
    - `README.md` - Context over the original implementation targets.

**Naming Conventions:**
- Python files are strictly named with `snake_case.py`. 
- Top-level directories currently eschew prefixes or versions.
- Deep learning layers and model classes use `PascalCase`/`CamelCase` (`EvolveGCN`, `GCNLayer`), as is tradition in object-oriented Python network building.
- Entry points are explicitly prefixed with `train_`.
