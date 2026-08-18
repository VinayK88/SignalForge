-- SignalForge synthetic feature-engineering example.
-- Adapt table/column names to the governed warehouse in a real deployment.

WITH account_window AS (
  SELECT
    event_id,
    account_id,
    event_ts,
    order_amount,
    device_id,
    payment_id,
    shipping_id,
    COUNT(*) OVER (
      PARTITION BY account_id
      ORDER BY event_ts
      RANGE BETWEEN INTERVAL '1' HOUR PRECEDING AND CURRENT ROW
    ) AS velocity_1h,
    COUNT(DISTINCT account_id) OVER (
      PARTITION BY device_id
      ORDER BY event_ts
      RANGE BETWEEN INTERVAL '7' DAY PRECEDING AND CURRENT ROW
    ) AS device_accounts_7d,
    COUNT(DISTINCT account_id) OVER (
      PARTITION BY payment_id
      ORDER BY event_ts
      RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW
    ) AS payment_accounts_30d,
    COUNT(DISTINCT account_id) OVER (
      PARTITION BY shipping_id
      ORDER BY event_ts
      RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW
    ) AS shipping_accounts_30d
  FROM synthetic_order_events
)
SELECT * FROM account_window;
