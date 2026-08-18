from __future__ import annotations

from math import exp

import numpy as np

RANDOM_STATE = 83


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + exp(-value))


def generate_events(n: int = 2400, seed: int = RANDOM_STATE, adversarial_shift: bool = False) -> list[dict]:
    """Create deterministic synthetic commerce/risk events.

    adversarial_shift=True creates a deliberately shifted recent population for
    monitoring tests. It is not intended to represent any real fraud population.
    """
    rng = np.random.default_rng(seed)
    events: list[dict] = []

    for i in range(n):
        account_age_days = float(np.clip(rng.gamma(2.3, 24.0), 0.1, 365.0))
        order_amount = float(np.clip(rng.lognormal(5.25, 0.85), 8.0, 3500.0))
        device_accounts = int(1 + rng.poisson(1.1))
        payment_accounts = int(1 + rng.poisson(0.7))
        shipping_accounts = int(1 + rng.poisson(1.2))
        velocity = int(rng.poisson(1.25))
        refund_rate = float(np.clip(rng.beta(1.5, 8.5), 0.0, 0.95))
        distance = float(np.clip(rng.gamma(1.6, 210.0), 0.0, 5000.0))
        new_device = int(rng.random() < 0.22)
        digital_goods = int(rng.random() < 0.18)

        if adversarial_shift:
            # Synthetic low-and-slow adaptation: slightly newer accounts, more
            # shared entities, and higher refund behavior without extreme velocity.
            account_age_days = max(0.1, account_age_days * 0.58)
            device_accounts += int(rng.random() < 0.55)
            payment_accounts += int(rng.random() < 0.35)
            shipping_accounts += int(rng.random() < 0.45)
            refund_rate = float(np.clip(refund_rate + rng.uniform(0.03, 0.12), 0.0, 0.95))
            velocity = min(velocity, 3)

        component_size = int(1 + device_accounts + payment_accounts + shipping_accounts + rng.poisson(2.0))
        risky_neighbor_ratio = float(
            np.clip(
                rng.beta(1.2 + 0.25 * max(0, device_accounts - 2), 5.5),
                0.0,
                0.98,
            )
        )

        # The latent score intentionally contains learnable adversarial structure.
        # Small Gaussian perturbation preserves overlap without turning the label
        # into a nearly-random Bernoulli draw from a weak signal.
        linear = (
            -3.2
            + 1.55 * (account_age_days < 3.0)
            + 1.15 * (device_accounts >= 5)
            + 1.05 * (payment_accounts >= 4)
            + 0.78 * (shipping_accounts >= 6)
            + 1.10 * (velocity >= 4)
            + 1.25 * (refund_rate >= 0.30)
            + 0.82 * (distance >= 900.0)
            + 1.15 * new_device
            + 0.42 * digital_goods
            + 1.20 * (risky_neighbor_ratio >= 0.35)
            + 0.55 * (order_amount >= 900.0)
            + float(rng.normal(0.0, 0.28))
        )
        fraud_probability = _sigmoid(linear)
        latent_observation = fraud_probability + float(rng.normal(0.0, 0.045))
        fraud_label = int(latent_observation >= 0.40)
        fraud_loss = float(
            fraud_label
            * order_amount
            * (0.52 + 0.30 * refund_rate + 0.08 * digital_goods + 0.05 * new_device)
        )
        ring_label = int(
            (device_accounts >= 5 and component_size >= 13)
            or (payment_accounts >= 4 and risky_neighbor_ratio >= 0.28)
            or (shipping_accounts >= 7 and risky_neighbor_ratio >= 0.22)
        )

        events.append({
            "event_id": f"evt-{seed}-{i:05d}",
            "account_id": f"acct-{i:05d}",
            "device_id": f"dev-{i % max(20, n // 8):04d}",
            "payment_id": f"pay-{i % max(25, n // 7):04d}",
            "shipping_id": f"ship-{i % max(30, n // 6):04d}",
            "account_age_days": round(account_age_days, 4),
            "order_amount": round(order_amount, 2),
            "device_accounts_7d": device_accounts,
            "payment_accounts_30d": payment_accounts,
            "shipping_accounts_30d": shipping_accounts,
            "velocity_1h": velocity,
            "refund_rate_30d": round(refund_rate, 5),
            "distance_from_home_km": round(distance, 2),
            "new_device": new_device,
            "digital_goods": digital_goods,
            "graph_component_size": component_size,
            "risky_neighbor_ratio": round(risky_neighbor_ratio, 5),
            "fraud_label": fraud_label,
            "fraud_loss": round(fraud_loss, 2),
            "ring_label": ring_label,
        })

    return events
