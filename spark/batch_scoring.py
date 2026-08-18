"""Illustrative PySpark batch feature pipeline for SignalForge.

This file is not imported by the core package, so PySpark remains optional.
"""
from pyspark.sql import Window
from pyspark.sql import functions as F


def build_batch_features(events_df):
    account_hour = (
        Window.partitionBy("account_id")
        .orderBy(F.col("event_ts").cast("long"))
        .rangeBetween(-3600, 0)
    )
    return (
        events_df
        .withColumn("velocity_1h", F.count("event_id").over(account_hour))
        .withColumn("new_device", F.col("device_first_seen_ts") >= F.col("event_ts") - F.expr("INTERVAL 1 DAY"))
        .select(
            "event_id",
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
        )
    )
