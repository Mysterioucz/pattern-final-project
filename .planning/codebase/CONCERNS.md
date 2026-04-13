# Technical Concerns

**Codebase Maturity & Structure:**
- **Project Nascent State**: The root of the repository is effectively an empty container. The actual logic resides in a `reference/` folder. This separation is good for reference but creates a "blank slate" problem where the primary project structure has yet to be defined.
- **Reference Dependency**: The current project is entirely dependent on the cloned `EvolveGCN` repository. If the upstream repository changes or if the local clone is deleted, the core "pattern recognition" logic will be lost.

**Dependency & Environment Risks:**
- **Legacy Stack**: The requirement for `tensorflow == 2.4.1` and `numpy <= 1.19.5` is a major concern. These are legacy versions that may not be easily installable on modern operating systems or with recent versions of Python (3.10+). This creates a "time-bomb" for environment reproducibility.
- **Incompatible Drivers**: Older versions of TensorFlow often require specific, older versions of CUDA and cuDNN, which can be difficult to maintain on modern GPU workstations.

**Robustness & Scalability:**
- **Fragile Data Ingestion**: The data loading logic is tightly coupled to the Elliptic Bitcoin dataset structure. Adapting the project to a different graph problem would require significant refactoring of the parsing logic.
- **Manual Processes**: The lack of automation in training, evaluation, and deployment means the project is prone to human error during manual execution cycles.
- **Missing Monitoring**: There is no live monitoring or logging of model performance, which makes it difficult to diagnose why a model might be underperforming or failing to converge during long training runs.

**Security Considerations:**
- **Unpatched Vulnerabilities**: Using legacy libraries (like TF 2.4.x) means the project is exposed to security vulnerabilities that have been patched in more recent versions but will never be backported to the legacy branch.

**Performance Bottlenecks:**
- **Sparse Matrix Efficiency**: While the code uses SciPy for sparse matrices, the interface between SciPy and TensorFlow can sometimes be a performance bottleneck if large matrices are frequently converted between formats.
- **GPU Utilization**: Without modern TensorFlow optimizations, maximizing GPU utilization for complex graph-based recurrent networks can be challenging.
