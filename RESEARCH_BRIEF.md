# P4 — STATS19 Road Collision Severity

**Current scope:** user-authorized computational study of the 2024 and 2025 public collision tables. The former five-year multiclass plan is preserved in [docs/history/RESEARCH_BRIEF_PHASE0.md](docs/history/RESEARCH_BRIEF_PHASE0.md) and is superseded.

**Title:** Predicting Road Collision Severity Using Explainable Machine Learning: A Temporal Evaluation of Great Britain's STATS19 Data

- RQ1: How well do speed limit, road type, junction detail and road class discriminate recorded Fatal/Serious versus Slight collisions?
- RQ2: What joint incremental predictive value comes from light, weather, road surface, day of week and hour?
- RQ3: Do fixed random forest and histogram gradient boosting models improve upon logistic regression in the following year? One temporal test cannot establish stable improvement across many years.

Unit: one unique police-recorded personal-injury collision. Target: official recorded collision_severity 1/2→1 and 3→0, not adjusted estimates or casualty-level labels. Source and hashes: [data/manifest.json](data/manifest.json). Licence: OGL v3.0 with DfT attribution.

Design: 2024 stratified 80/20 training/validation; 2025 final test. Train-only imputation, scaling and one-hot encoding. Fixed models, validation-only F1 thresholds, no test tuning. See [protocol](docs/EXPERIMENT_PROTOCOL.md) for exact features, exclusions, parameters, metric denominators and execution-only reproducibility correction.

Metrics: binary positive-class P/R/F1, confusion counts, FPR, Average Precision (primary), separately labelled trapezoidal PR-AUC, ROC-AUC and Brier score. All eligible test collisions remain in each denominator. Interpretation uses validation permutation importance and regularized logistic coefficients, not causal effects.

Deliverables: executed Python code, tests, acquisition hashes, saved predictions and metrics, figures, English report and reproducible README. Results and limitations are reported in [report/RESEARCH_REPORT.md](report/RESEARCH_REPORT.md). No production, occurrence-prediction, or causal claim; no claim of independent student implementation. Historical gates remain unsigned and no remote publication has occurred.
