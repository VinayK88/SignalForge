from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .features import GRAPH_FEATURES, MODEL_FEATURES, feature_matrix, feature_vector
from .synthetic import RANDOM_STATE, generate_events

MODEL_VERSION = "0.1.0"
FEATURE_SCHEMA_VERSION = "2026-08-18"


@dataclass
class ModelBundle:
    fraud_model: CalibratedClassifierCV
    loss_model: GradientBoostingRegressor
    anomaly_model: IsolationForest
    anomaly_reference_scores: np.ndarray
    ring_scaler: StandardScaler
    ring_model: LogisticRegression
    metrics: dict


def _labels(events: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(event[key]) for event in events], dtype=float)


@lru_cache(maxsize=1)
def model_bundle() -> ModelBundle:
    events = generate_events(2600, RANDOM_STATE)
    indices = np.arange(len(events))
    y = _labels(events, "fraud_label").astype(int)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train = [events[int(i)] for i in train_idx]
    test = [events[int(i)] for i in test_idx]
    x_train = feature_matrix(train)
    x_test = feature_matrix(test)
    y_train = _labels(train, "fraud_label").astype(int)
    y_test = _labels(test, "fraud_label").astype(int)

    base = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.055,
        max_depth=2,
        random_state=RANDOM_STATE,
    )
    fraud_model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    fraud_model.fit(x_train, y_train)
    prob = fraud_model.predict_proba(x_test)[:, 1]

    loss_model = GradientBoostingRegressor(
        n_estimators=140,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE + 1,
    )
    loss_model.fit(x_train, _labels(train, "fraud_loss"))
    loss_pred = np.maximum(0.0, loss_model.predict(x_test))
    loss_true = _labels(test, "fraud_loss")

    legitimate_x = feature_matrix([event for event in train if not event["fraud_label"]])
    anomaly_model = IsolationForest(
        n_estimators=180,
        contamination=0.07,
        random_state=RANDOM_STATE + 2,
    )
    anomaly_model.fit(legitimate_x)
    anomaly_reference_scores = anomaly_model.decision_function(legitimate_x)

    ring_x_train = feature_matrix(train, GRAPH_FEATURES)
    ring_x_test = feature_matrix(test, GRAPH_FEATURES)
    ring_scaler = StandardScaler().fit(ring_x_train)
    ring_model = LogisticRegression(max_iter=600, random_state=RANDOM_STATE + 3)
    ring_model.fit(ring_scaler.transform(ring_x_train), _labels(train, "ring_label").astype(int))
    ring_prob = ring_model.predict_proba(ring_scaler.transform(ring_x_test))[:, 1]
    ring_true = _labels(test, "ring_label").astype(int)

    metrics = {
        "fraud_classifier": {
            "model": "GradientBoostingClassifier + sigmoid calibration",
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
            "pr_auc": round(float(average_precision_score(y_test, prob)), 4),
            "brier_score": round(float(brier_score_loss(y_test, prob)), 4),
            "train_rows": len(train),
            "holdout_rows": len(test),
        },
        "expected_loss": {
            "model": "GradientBoostingRegressor",
            "mae": round(float(mean_absolute_error(loss_true, loss_pred)), 2),
            "r2": round(float(r2_score(loss_true, loss_pred)), 4),
        },
        "graph_ring": {
            "model": "LogisticRegression over graph-derived sharing features",
            "roc_auc": round(float(roc_auc_score(ring_true, ring_prob)), 4),
        },
        "anomaly": {
            "model": "IsolationForest",
            "reference_rows": int(len(legitimate_x)),
        },
        "data_boundary": "deterministic synthetic populations only",
    }

    return ModelBundle(
        fraud_model=fraud_model,
        loss_model=loss_model,
        anomaly_model=anomaly_model,
        anomaly_reference_scores=anomaly_reference_scores,
        ring_scaler=ring_scaler,
        ring_model=ring_model,
        metrics=metrics,
    )


def score_models(event: dict) -> dict:
    bundle = model_bundle()
    row = feature_vector(event).reshape(1, -1)
    fraud_probability = float(bundle.fraud_model.predict_proba(row)[0, 1])
    loss_prediction = max(0.0, float(bundle.loss_model.predict(row)[0]))

    anomaly_score = float(bundle.anomaly_model.decision_function(row)[0])
    anomaly_percentile = float(np.mean(bundle.anomaly_reference_scores >= anomaly_score) * 100.0)

    graph_row = feature_vector(event, GRAPH_FEATURES).reshape(1, -1)
    graph_risk = float(bundle.ring_model.predict_proba(bundle.ring_scaler.transform(graph_row))[0, 1])

    combined_risk = float(np.clip(0.74 * fraud_probability + 0.16 * graph_risk + 0.10 * (anomaly_percentile / 100.0), 0.0, 1.0))
    expected_loss = max(loss_prediction, float(event.get("order_amount", 0.0)) * combined_risk * 0.52)

    return {
        "fraud_probability": round(fraud_probability, 4),
        "graph_risk": round(graph_risk, 4),
        "anomaly_percentile": round(anomaly_percentile, 1),
        "combined_risk": round(combined_risk, 4),
        "expected_loss": round(expected_loss, 2),
    }


def discover_cohorts(events: list[dict]) -> dict:
    names = ["account_age_days", "device_accounts_7d", "velocity_1h", "refund_rate_30d", "risky_neighbor_ratio"]
    x = feature_matrix(events, names)
    scaled = StandardScaler().fit_transform(x)
    labels = DBSCAN(eps=0.85, min_samples=10).fit_predict(scaled)
    cluster_ids = sorted(int(label) for label in set(labels) if label >= 0)
    clusters = []
    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        clusters.append({
            "cluster_id": cluster_id,
            "size": int(mask.sum()),
            "synthetic_fraud_rate": round(float(np.mean([events[i]["fraud_label"] for i in np.where(mask)[0]])), 4),
        })
    return {
        "model": "DBSCAN",
        "clusters": clusters,
        "noise_points": int(np.sum(labels < 0)),
        "boundary": "descriptive synthetic cohort discovery; not actor attribution",
    }
