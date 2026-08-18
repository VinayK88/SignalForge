from __future__ import annotations

import numpy as np

MODEL_FEATURES = [
    "account_age_days",
    "order_amount",
    "device_accounts_7d",
    "payment_accounts_30d",
    "shipping_accounts_30d",
    "velocity_1h",
    "refund_rate_30d",
    "distance_from_home_km",
    "new_device",
    "digital_goods",
    "graph_component_size",
    "risky_neighbor_ratio",
]

GRAPH_FEATURES = [
    "device_accounts_7d",
    "payment_accounts_30d",
    "shipping_accounts_30d",
    "graph_component_size",
    "risky_neighbor_ratio",
]

DRIFT_FEATURES = [
    "account_age_days",
    "device_accounts_7d",
    "payment_accounts_30d",
    "shipping_accounts_30d",
    "velocity_1h",
    "refund_rate_30d",
]


def feature_vector(event: dict, names: list[str] | None = None) -> np.ndarray:
    names = names or MODEL_FEATURES
    return np.asarray([float(event.get(name, 0.0)) for name in names], dtype=float)


def feature_matrix(events: list[dict], names: list[str] | None = None) -> np.ndarray:
    names = names or MODEL_FEATURES
    return np.asarray([feature_vector(event, names) for event in events], dtype=float)


def reason_codes(event: dict, calibrated_probability: float, graph_risk: float, anomaly_percentile: float) -> list[str]:
    reasons: list[tuple[str, float]] = []
    if event.get("account_age_days", 999) < 3:
        reasons.append(("very_new_account", 0.9))
    if event.get("device_accounts_7d", 0) >= 5:
        reasons.append(("shared_device_velocity", 0.85))
    if event.get("payment_accounts_30d", 0) >= 4:
        reasons.append(("shared_payment_instrument", 0.82))
    if event.get("shipping_accounts_30d", 0) >= 6:
        reasons.append(("shared_shipping_destination", 0.76))
    if event.get("velocity_1h", 0) >= 4:
        reasons.append(("high_purchase_velocity", 0.8))
    if event.get("refund_rate_30d", 0.0) >= 0.30:
        reasons.append(("elevated_refund_behavior", 0.72))
    if event.get("new_device", 0):
        reasons.append(("new_device", 0.58))
    if graph_risk >= 0.65:
        reasons.append(("graph_ring_risk", 0.88))
    if anomaly_percentile >= 90:
        reasons.append(("behavioral_anomaly", 0.84))
    if calibrated_probability >= 0.75:
        reasons.append(("high_calibrated_fraud_risk", 0.92))
    reasons.sort(key=lambda item: (-item[1], item[0]))
    return [name for name, _ in reasons[:5]] or ["no_elevated_reason_code"]
