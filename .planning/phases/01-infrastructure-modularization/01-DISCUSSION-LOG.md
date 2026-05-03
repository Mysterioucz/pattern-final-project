# Phase 1: Infrastructure & Modularization - Discussion Log

**Date:** 2026-05-03
**Participants:** Antigravity (AI) & User

## Q&A Session

### Area: Source Code Organization
- **Q:** How should we organize the modularized code in `projects/src/`?
- **Options:** 
  - (1) Flat Structure
  - (2) Domain-Driven (Nested) [Recommended]
  - (3) Reference-Aligned
- **Decision:** **(2) Domain-Driven (Nested)**
- **Rationale:** Better scalability for future Knowledge Distillation framework development.

### Area: Kaggle Credential Management
- **Q:** How should the modularized script authenticate with Kaggle?
- **Options:**
  - (1) Environment Variables (.env) [Recommended]
  - (2) Standard Kaggle Path
  - (3) Prompt on Run
- **Decision:** **(1) Environment Variables (.env)**
- **Rationale:** Project isolation and consistency with existing notebook setup.

### Area: Data Persistence Strategy
- **Q:** Preprocessing the Elliptic dataset involves mapping IDs and standardizing features. Should we save the result?
- **Options:**
  - (1) On-the-fly [Recommended for flexibility]
  - (2) Pickle/Torch Save
  - (3) Hybrid
- **Decision:** **(1) On-the-fly**
- **Rationale:** User prefers flexibility over speed for current development stage.

### Area: Temporal Graph Handling
- **Q:** The Elliptic dataset covers 49 distinct time steps. How should the GNN "see" this data?
- **Options:**
  - (1) Static Graph [Initial]
  - (2) Snapshot-based
  - (3) Windowed
- **Decision:** **(1) Static Graph for now**
- **Rationale:** Start with static baseline; consider windowed or EvolveGCN (handled temporal steps better) in future phases.
