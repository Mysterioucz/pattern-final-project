# Coding Conventions

**General Language Standards:**
- **Python Style**: The codebase follows standard Python style guidelines. While there is no explicit `.flake8` or `pyproject.toml` file enforcing rules, the code generally aligns with PEP8 regarding naming (snake_case for files and functions, PascalCase for classes).

**Code Organization & Patterns:**
- **Modular Design**: The system is organized into distinct modules by functionality. For example, neural network layers are separated into `models/layers/`, while higher-level model assemblies are in the `models/` root.
- **Object-Oriented Programming (OOP)**: The implementation heavily utilizes OOP principles, specifically for defining neural network components. Most model parts are encapsulated within classes that inherit from `tf.keras.layers.Layer` or `tf.keras.Model`.
- **Inheritance & Polymorphism**: Custom layers like `GCNLayer` or `GCNSkipLayer` provide specialized implementations of the `call()` method, allowing them to be swapped or stacked interchangeably within model definitions.

**Specific Implementation Habits:**
- **Type Hinting**: Modern Python type annotations are used in some key model files (e.g., `typing.Tuple`, `typing.List`). This improves readability and provides some basic level of static analysis support for developers using IDEs like VS Code or PyCharm.
- **Parameter Handling**: Hyperparameters and configuration are often passed directly into class constructors. There is no central configuration management system (like Hydra or OmegaConf) currently in place.

**Error Handling & Logging:**
- **Native Exceptions**: The code relies on standard Python exceptions. There are no custom error classes defined for domain-specific failures (e.g., "InvalidGraphInputError").
- **Console Logging**: Most output is handled via standard `print()` statements. A more robust logging framework (using Python's `logging` module) has not yet been implemented.

**Naming Conventions Summary:**
- **Files**: `snake_case.py` (e.g., `data_loader.py`, `train_evolve_gcn.py`).
- **Classes**: `PascalCase` (e.g., `EvolveGCN`, `GCNSequential`).
- **Functions/Methods**: `snake_case` (e.g., `normalize_adjencency_mat`).
- **Variables**: Descriptive `snake_case`.
