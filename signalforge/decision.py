from __future__ import annotations


def choose_action(combined_risk: float, expected_loss: float) -> dict:
    """Choose the action with the lowest modeled business cost.

    Costs are synthetic and intentionally transparent; they are not calibrated to
    any real retailer, bank, payment network, or customer-friction measurement.
    """
    legitimate_probability = 1.0 - combined_risk
    costs = {
        "ALLOW": expected_loss,
        "REVIEW": 9.0 + 0.20 * expected_loss + 3.0 * legitimate_probability,
        "BLOCK": 5.0 + 0.03 * expected_loss + 38.0 * legitimate_probability,
    }
    decision = min(costs, key=costs.get)
    return {
        "decision": decision,
        "action_costs": {name: round(value, 2) for name, value in costs.items()},
        "decision_margin": round(sorted(costs.values())[1] - sorted(costs.values())[0], 2),
        "boundary": "synthetic expected-cost policy; not a production policy threshold",
    }
