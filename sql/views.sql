-- =====================================================================
-- Analysis views: one clean row per order / per customer.
-- Every KPI query, the ML notebooks, and the dashboard build on these,
-- so business logic (what counts as revenue, what counts as late)
-- lives in exactly one place.
-- =====================================================================

DROP VIEW IF EXISTS v_customer_summary;
DROP VIEW IF EXISTS v_orders_enriched;

CREATE VIEW v_orders_enriched AS
WITH items AS (
    SELECT order_id,
           COUNT(*)            AS n_items,
           SUM(price)          AS items_revenue,
           SUM(freight_value)  AS freight_value
    FROM order_items
    GROUP BY order_id
),
payments AS (
    SELECT order_id,
           SUM(payment_value)         AS payment_value,
           MAX(payment_installments)  AS max_installments,
           -- main payment method = the one carrying the most money
           (ARRAY_AGG(payment_type ORDER BY payment_value DESC))[1] AS main_payment_type
    FROM order_payments
    GROUP BY order_id
),
reviews AS (
    SELECT order_id, AVG(review_score)::NUMERIC(3,2) AS review_score
    FROM order_reviews
    GROUP BY order_id
)
SELECT
    o.order_id,
    c.customer_unique_id,
    c.customer_state,
    o.order_status,
    o.order_purchase_timestamp                                  AS purchase_ts,
    DATE_TRUNC('month', o.order_purchase_timestamp)::DATE       AS order_month,
    o.order_delivered_customer_date                             AS delivered_ts,
    o.order_estimated_delivery_date                             AS estimated_ts,
    i.n_items,
    i.items_revenue,
    i.freight_value,
    i.items_revenue + i.freight_value                           AS order_total,
    p.payment_value,
    p.main_payment_type,
    p.max_installments,
    r.review_score,
    -- delivery metrics (NULL when the order was never delivered)
    ROUND(EXTRACT(EPOCH FROM o.order_delivered_customer_date
                             - o.order_purchase_timestamp) / 86400.0, 1) AS delivery_days,
    ROUND(EXTRACT(EPOCH FROM o.order_delivered_customer_date
                             - o.order_estimated_delivery_date) / 86400.0, 1) AS days_vs_estimate,
    CASE WHEN o.order_delivered_customer_date IS NULL THEN NULL
         WHEN o.order_delivered_customer_date::DATE > o.order_estimated_delivery_date::DATE THEN 1
         ELSE 0 END                                             AS is_late
FROM orders o
JOIN customers c   ON c.customer_id = o.customer_id
LEFT JOIN items i    ON i.order_id = o.order_id
LEFT JOIN payments p ON p.order_id = o.order_id
LEFT JOIN reviews r  ON r.order_id = o.order_id;


-- One row per real customer (customer_unique_id), delivered orders only.
-- This is the starting point for RFM segmentation and churn modelling.
CREATE VIEW v_customer_summary AS
SELECT
    customer_unique_id,
    MIN(customer_state)                    AS customer_state,
    COUNT(*)                               AS frequency,
    SUM(order_total)                       AS monetary,
    MIN(purchase_ts)                       AS first_purchase_ts,
    MAX(purchase_ts)                       AS last_purchase_ts,
    AVG(review_score)::NUMERIC(3,2)        AS avg_review_score,
    AVG(delivery_days)::NUMERIC(6,1)       AS avg_delivery_days,
    AVG(is_late)::NUMERIC(4,3)             AS late_share
FROM v_orders_enriched
WHERE order_status = 'delivered'
GROUP BY customer_unique_id;
