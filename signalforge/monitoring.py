from __future__ import annotations

import numpy as np

from .features import DRIFT_FEATURES


def population_stability_index(reference: list[float], current: list[float], bins: int = 8) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = np.maximum(ref_hist / max(1, ref_hist.sum()), 1e-6)
    cur_pct = np.maximum(cur_hist / max(1, cur_hist.sum()), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(reference: list[dict], current: list[dict]) -> dict:
    features = []
    for name in DRIFT_FEATURES:
        psi = population_stability_index(
            [float(event[name]) for event in reference],
            [float(event[name]) for event in current],
        )
        status = "HIGH" if psi >= 0.25 else "WATCH" if psi >= 0.10 else "STABLE"
        features.append({"feature": name, "psi": round(psi, 4), "status": status})
    features.sort(key=lambda item: -item["psi"])
    return {
        "features": features,
        "max_psi": features[0]["psi"] if features else 0.0,
        "status": features[0]["status"] if features else "STABLE",
        "interpretation": "population drift only; outcome evidence is required before calling this concept drift or adversarial adaptation",
    }


def active_learning_priority(fraud_probability: float, graph_risk: float, anomaly_percentile: float) -> float:
    uncertainty = 1.0 - min(1.0, abs(fraud_probability - 0.5) * 2.0)
    anomaly_risk = anomaly_percentile / 100.0
    disagreement = max(
        abs(fraud_probability - graph_risk),
        abs(fraud_probability - anomaly_risk),
        abs(graph_risk - anomaly_risk),
    )
    return round(float(np.clip(0.65 * uncertainty + 0.35 * disagreement, 0.0, 1.0)), 4)
