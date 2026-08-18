# SignalForge model card

## Purpose
SignalForge demonstrates an end-to-end fraud/risk decision architecture on synthetic data. It is designed for ML engineering, decision science, monitoring, and explicitly non-production portfolio evaluation.

## Learned components
- GradientBoostingClassifier wrapped by sigmoid probability calibration for fraud probability.
- GradientBoostingRegressor for synthetic expected-loss estimation.
- IsolationForest fit to synthetic legitimate-reference behavior for novelty detection.
- LogisticRegression over shared-entity graph features for synthetic fraud-ring risk.
- DBSCAN for descriptive emerging-cohort discovery.

## Decision boundary
The models do not directly block an event. `decision.py` calculates transparent synthetic expected costs for ALLOW, REVIEW, and BLOCK and chooses the minimum-cost action. This policy is intentionally separate from predictive model code.

## Monitoring boundary
PSI identifies population movement only. SignalForge does not infer concept drift or adversarial intent from PSI alone. Outcome evidence, calibration movement, analyst feedback, and business context would be required in production.

## Data and evaluation
All training, holdout, graph, drift and dashboard examples are deterministic synthetic data. Metrics are code-regression evidence and must not be represented as production fraud effectiveness.
