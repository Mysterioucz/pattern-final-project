# Tech Stack 

**Languages:**
- **Python 3.x**: This is the primary execution language for the entire project. Based on the dependency constraints in the reference implementation, the environment likely requires Python 3.7 or 3.8 to ensure compatibility with legacy versions of TensorBoard, NumPy, and TensorFlow.

**Frameworks & Libraries:**
- **TensorFlow (==2.4.1)**: Acts as the backbone for all neural network operations. The project utilizes the Keras API within TensorFlow to define custom layers and model architectures. It relies on the older 2.4.x branch, which may have specific requirements for CUDA and cuDNN versions.
- **NumPy (<= 1.19.5)**: Used for intensive numerical computations and array manipulations. The strict upper bound on the version suggests potential compatibility issues with newer NumPy releases and the specific TensorFlow version used.
- **Pandas**: Crucial for data manipulation and analysis, particularly for reading the structured CSV files that accompany the financial forensics datasets.
- **NetworkX**: A comprehensive library for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. It is used here to build graph objects from raw edge and feature data.
- **SciPy**: Specifically, the `scipy.sparse` module is used for efficient storage and manipulation of large, sparse adjacency matrices which are characteristic of the graph data being processed.

**Development & Runtime Environment:**
- **Local Execution**: The project is currently configured for direct execution on a local machine or server.
- **No Containerization**: Currently, there are no Dockerfiles or configuration for containerized environments, meaning developers must manually manage their Python virtual environments and system dependencies.

**Current Codebase Context:**
- The root of the project is currently a "brownfield" or empty project workspace.
- A functional reference implementation has been brought in via `reference/EvolveGCN/`. 
- Future development task will involve adapting this reference logic into a more structured, modern, and potentially more generic framework for Pattern Recognition and Graph Analysis.
- The stack will likely evolve to include more modern versions of these libraries or perhaps shift to PyTorch if the project requirements change.
