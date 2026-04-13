# Testing Strategy

**Current Testing Posture:**
- **No Automated Test Suites**: The project currently lacks a formal suite of automated tests. There are no directories such as `tests/` or files prefixed with `test_` in the project root or the reference implementation.
- **Manual Verification**: Validation of code changes currently relies on manual execution of the training scripts (`train_evolve_gcn.py`, etc.). If the script runs without crashing and produces decreasing loss values, it is considered "working" in the current researcher-centric workflow.

**Testing Gaps & Opportunities:**
- **Unit Testing**: There is a significant opportunity to implement unit tests for the core mathematical utilities in `utils.py` and the data parsing logic in `data_loader.py`. Frameworks like `pytest` would be suitable here.
- **Integration Testing**: Testing the end-to-end flow from data ingestion to a single model training step (a "smoke test") would provide confidence that the complex graph transformations are being performed correctly.
- **Layer Validation**: Individual neural network layers in `models/layers/` could be tested against known input/output shapes to ensure that structural changes elsewhere in the code don't break the forward pass.

**Build and CI/CD:**
- **No CI Pipeline**: There are no continuous integration pipelines configured. This means there is no automated verification of code quality, type checks, or linting on push.
- **Environment Consistency**: Without a `requirements.txt` or `environment.yml` at the root, ensuring a consistent testing environment across different developer machines is a manual process.

**Future Testing Roadmap:**
- **Introduce Pytest**: Initialize a `tests/` directory and begin adding unit tests for the most fragile parts of the codebase (e.g., adjacency matrix normalization).
- **GitHub Actions/GitLab CI**: Configure a basic CI pipeline to run these tests automatically and verify that the environment can be built from scratch.
- **Data Validation**: Implement schema validation for input datasets to ensure that the `data_loader.py` handles malformed entries gracefully.
- **Model Result Tracking**: Use a system like MLflow or W&B to log and compare model performance, which serves as a form of high-level functional testing for ML projects.
