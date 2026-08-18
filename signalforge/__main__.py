from __future__ import annotations

import json

from .service import build_report, score_event


def main() -> None:
    demo = {
        "event_id": "ord-demo-1042",
        "account_age_days": 2,
        "order_amount": 1249.0,
        "device_accounts_7d": 6,
        "payment_accounts_30d": 4,
        "shipping_accounts_30d": 8,
        "velocity_1h": 5,
        "refund_rate_30d": 0.38,
        "distance_from_home_km": 1710,
        "new_device": True,
        "digital_goods": False,
        "graph_component_size": 21,
        "risky_neighbor_ratio": 0.46,
    }
    print(json.dumps({"sample_score": score_event(demo), "report": build_report()}, indent=2))


if __name__ == "__main__":
    main()
