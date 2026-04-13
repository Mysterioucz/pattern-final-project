# External Integrations

**Data Sources & Storage:**
- **Static File Ingestion**: In its current state, the system does not integrate with any external database engines (like SQL or NoSQL). All data ingestion is handled through local file system calls, targeting specific datasets like the Elliptic Bitcoin dataset.
- **CSV/Data Exports**: The output of the training runs (accuracy metrics, loss logs) is typically written directly to local flat files or standard output, with no current integration to a centralized logging or artifact storage service.

**Third-Party APIs & Services:**
- **No Remote API Connections**: There are no identified integrations with external REST, GraphQL, or gRPC services. The logic is self-contained and does not require internet connectivity during the model training phase.
- **Authentication**: There is no authentication layer or integration with identity providers (like OAuth, Auth0) as the tools are designed for local researcher use.

**Infrastructure & Cloud:**
- **Local Computing**: The project is strictly local-first. There is no configuration for cloud-based training platforms like AWS SageMaker, Google Vertex AI, or Azure Machine Learning.
- **Monitoring**: No monitoring or observability tools (like Prometheus, Grafana, or Sentry) are integrated into the pipeline yet.

**Future Integration Considerations:**
- **MLflow/WandB**: As the project matures, integrating a dedicated experiment tracking tool like Weights & Biases or MLflow will be essential for managing various training runs and hyperparameter tuning sessions.
- **Database Backend**: Migrating from raw CSV files to a structured database or a specialized graph database (like Neo4j) could improve data management and query performance for larger network datasets.
- **Cloud Storage**: Future versions might benefit from S3 or GCS integrations to fetch large datasets on demand rather than storing them locally in a `data/` or `reference/` folder.
- **Web Interface**: If a visualization dashboard is required, integrations with frameworks like Streamlit or Dash may be introduced.

*Note: This integration map will be updated as the project transitions from a reference-heavy structure to a production-ready Pattern Recognition application.*
