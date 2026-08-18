from __future__ import annotations

from .decision import choose_action
from .features import reason_codes
from .graph import graph_summary
from .models import MODEL_VERSION, FEATURE_SCHEMA_VERSION, discover_cohorts, model_bundle, score_models
from .monitoring import active_learning_priority, drift_report
from .synthetic import RANDOM_STATE, generate_events


def normalize_event(event: dict) -> dict:
    normalized = dict(event)
    normalized.setdefault("event_id", "runtime-event")
    normalized.setdefault("account_age_days", 30.0)
    normalized.setdefault("order_amount", 100.0)
    normalized.setdefault("device_accounts_7d", 1)
    normalized.setdefault("payment_accounts_30d", 1)
    normalized.setdefault("shipping_accounts_30d", 1)
    normalized.setdefault("velocity_1h", 1)
    normalized.setdefault("refund_rate_30d", 0.05)
    normalized.setdefault("distance_from_home_km", 50.0)
    normalized.setdefault("new_device", 0)
    normalized.setdefault("digital_goods", 0)
    normalized.setdefault("graph_component_size", 4)
    normalized.setdefault("risky_neighbor_ratio", 0.05)
    return normalized


def score_event(event: dict) -> dict:
    event = normalize_event(event)
    model_scores = score_models(event)
    decision = choose_action(model_scores["combined_risk"], model_scores["expected_loss"])
    review_priority = active_learning_priority(
        model_scores["fraud_probability"],
        model_scores["graph_risk"],
        model_scores["anomaly_percentile"],
    )
    reasons = reason_codes(
        event,
        model_scores["fraud_probability"],
        model_scores["graph_risk"],
        model_scores["anomaly_percentile"],
    )
    return {
        "event_id": event["event_id"],
        **model_scores,
        **decision,
        "reason_codes": reasons,
        "review_priority": review_priority,
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "data_boundary": "synthetic-model demonstration",
    }


def build_report() -> dict:
    bundle = model_bundle()
    reference = generate_events(700, RANDOM_STATE + 10)
    current = generate_events(700, RANDOM_STATE + 11, adversarial_shift=True)
    sample_events = current[:80]
    decisions = [score_event(event)["decision"] for event in sample_events]
    return {
        "model": {
            "version": MODEL_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "evaluation": bundle.metrics,
        },
        "decision_sample": {
            "events": len(sample_events),
            "allow": decisions.count("ALLOW"),
            "review": decisions.count("REVIEW"),
            "block": decisions.count("BLOCK"),
        },
        "drift": drift_report(reference, current),
        "cohorts": discover_cohorts(current[:350]),
        "graph": graph_summary(current[:180]),
        "boundary": "all metrics are generated from deterministic synthetic data",
    }
