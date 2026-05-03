# Research Report: Phase 1 - Infrastructure & Modularization

## 1. Domain Analysis: Modularizing GNN Projects with PyG
Modularizing GNN projects involves separating the data pipeline, model architecture, and training logic. 
- **Data Pipeline**: Extend `torch_geometric.data.InMemoryDataset` to handle local CSV files and preprocessing. This allows for easy integration with `DataLoader`.
- **Model Architecture**: Define GNN layers in separate files (e.g., `gcn.py`, `evolve_gcn.py`) and use a factory pattern or clear imports in `main.py`.
- **Training Logic**: Use a separate `Trainer` class or modular functions in `train.py` to handle the training loop, validation, and checkpointing.

## 2. Patterns for Elliptic Dataset Handling
The Elliptic dataset is naturally temporal (49 timesteps). 
- **Static Pattern**: All nodes and edges are treated as a single graph. This is good for a baseline but ignores the temporal evolution of fraud.
- **Snapshot Pattern**: Each timestep is a separate graph. This is required for EvolveGCN.
- **Filtering**: Nodes with "unknown" labels should be filtered out or treated as a separate class. Features should be normalized (StandardScaler) per timestep or globally.

## 3. Kaggle API Integration
- Authentication should be handled via `.env` variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.
- The `kaggle` Python package provides a convenient API for downloading and unzipping the dataset automatically.
- Example: `api.dataset_download_files('elliptic-co/elliptic-data-set', path='data/raw', unzip=True)`.

## 4. Proposed Project Structure
```text
projects/
├── data/
│   ├── raw/            # Downloaded CSVs
│   └── processed/      # PyG .pt files
├── src/
│   ├── data/
│   │   ├── downloader.py
│   │   ├── preprocessor.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── gcn.py
│   │   └── evolve_gcn.py
│   ├── utils/
│   │   ├── metrics.py
│   │   └── logging.py
│   └── main.py
└── .env                # Kaggle credentials
```

## 5. Technical Risks & Mitigation
- **Memory Issues**: Although Elliptic fits in memory (~200k nodes), future expansions might not. Mitigation: Design for `Dataset` (on-disk) if needed.
- **Class Imbalance**: Fraud nodes are rare (<2%). Mitigation: Use weighted loss and F1-score for evaluation.
- **Temporal Leakage**: Ensure training is done on earlier timesteps and testing on later ones.

## 6. Validation Architecture
- **Unit Tests**: Verify data loading shapes and label distributions.
- **Sanity Check**: Run a single epoch of training to ensure no crashes.
- **Integration Test**: A script that downloads, processes, and trains a tiny model in <1 minute.
