-- =====================================================================
-- Business KPI queries (PostgreSQL)
-- Each query starts with "-- name: <name>" so src/run_kpis.py can run
-- them one by one and save each result to outputs/<name>.csv.
--
-- Conventions
--   * Revenue = item price + freight on DELIVERED orders.
--   * Analysis window = Jan 2017 to Aug 2018. Olist data before 2017
--     and after Aug 2018 is too sparse and would distort trends.
-- =====================================================================


-- name: 01_business_summary
-- Headline numbers for the dashboard's top row.
SELECT
    COUNT(*)                                        AS delivered_orders,
    COUNT(DISTINCT customer_unique_id)              AS customers,
    ROUND(SUM(order_total), 2)                      AS total_revenue,
    ROUND(AVG(order_total), 2)                      AS avg_order_value,
    ROUND(AVG(review_score), 2)                     AS avg_review_score,
    ROUND(100.0 * AVG(is_late), 1)                  AS late_delivery_pct
FROM v_orders_enriched
WHERE order_status = 'delivered'
  AND purchase_ts >= '2017-01-01' AND purchase_ts < '2018-09-01';


-- name: 02_monthly_revenue_growth
-- Monthly revenue, orders and AOV with month-over-month growth.
WITH monthly AS (
    SELECT order_month,
           COUNT(*)          AS orders,
           SUM(order_total)  AS revenue
    FROM v_orders_enriched
    WHERE order_status = 'delivered'
      AND purchase_ts >= '2017-01-01' AND purchase_ts < '2018-09-01'
    GROUP BY order_month
)
SELECT order_month,
       orders,
       ROUND(revenue, 2)                                       AS revenue,
       ROUND(revenue / orders, 2)                              AS avg_order_value,
       ROUND(100.0 * (revenue - LAG(revenue) OVER w)
                   / NULLIF(LAG(revenue) OVER w, 0), 1)        AS revenue_mom_pct,
       ROUND(SUM(revenue) OVER (ORDER BY order_month), 2)      AS cumulative_revenue
FROM monthly
WINDOW w AS (ORDER BY order_month)
ORDER BY order_month;


-- name: 03_delivery_performance_monthly
-- On-time delivery rate and average delivery time per month.
SELECT order_month,
       COUNT(*)                                     AS delivered_orders,
       ROUND(100.0 * (1 - AVG(is_late)), 1)         AS on_time_pct,
       ROUND(AVG(delivery_days), 1)                 AS avg_delivery_days,
       ROUND(AVG(days_vs_estimate), 1)              AS avg_days_vs_estimate
FROM v_orders_enriched
WHERE order_status = 'delivered'
  AND delivered_ts IS NOT NULL
  AND purchase_ts >= '2017-01-01' AND purchase_ts < '2018-09-01'
GROUP BY order_month
ORDER BY order_month;


-- name: 04_late_delivery_vs_review
-- How much do late deliveries hurt customer satisfaction?
-- (Key insight that feeds the churn model in Phase 3.)
SELECT CASE
           WHEN days_vs_estimate <= -7 THEN '1. 7+ days early'
           WHEN days_vs_estimate <=  0 THEN '2. On time (0-7 days early)'
           WHEN days_vs_estimate <=  7 THEN '3. 1-7 days late'
           ELSE                             '4. 7+ days late'
       END                                          AS delivery_bucket,
       COUNT(*)                                     AS orders,
       ROUND(AVG(review_score), 2)                  AS avg_review_score,
       ROUND(100.0 * AVG(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END), 1)
                                                    AS negative_review_pct
FROM v_orders_enriched
WHERE order_status = 'delivered'
  AND days_vs_estimate IS NOT NULL
  AND review_score IS NOT NULL
GROUP BY 1
ORDER BY 1;


-- name: 05_cohort_retention
-- Monthly cohort retention: of customers whose FIRST purchase was in
-- month X, what % bought again 1, 2, 3 ... months later?
WITH first_purchase AS (
    SELECT customer_unique_id,
           MIN(order_month) AS cohort_month
    FROM v_orders_enriched
    WHERE order_status = 'delivered'
    GROUP BY customer_unique_id
),
activity AS (
    SELECT DISTINCT
           f.cohort_month,
           o.customer_unique_id,
           (EXTRACT(YEAR  FROM AGE(o.order_month, f.cohort_month)) * 12
          + EXTRACT(MONTH FROM AGE(o.order_month, f.cohort_month)))::INT AS months_since_first
    FROM v_orders_enriched o
    JOIN first_purchase f USING (customer_unique_id)
    WHERE o.order_status = 'delivered'
),
cohort_size AS (
    SELECT cohort_month, COUNT(*) AS cohort_customers
    FROM first_purchase
    GROUP BY cohort_month
)
SELECT a.cohort_month,
       s.cohort_customers,
       a.months_since_first,
       COUNT(*)                                              AS active_customers,
       ROUND(100.0 * COUNT(*) / s.cohort_customers, 2)       AS retention_pct
FROM activity a
JOIN cohort_size s USING (cohort_month)
WHERE a.cohort_month >= '2017-01-01' AND a.cohort_month < '2018-09-01'
  AND a.months_since_first BETWEEN 0 AND 12
GROUP BY a.cohort_month, s.cohort_customers, a.months_since_first
ORDER BY a.cohort_month, a.months_since_first;


-- name: 06_revenue_by_category
-- Top 15 product categories by revenue, with share of total.
WITH cat AS (
    SELECT COALESCE(t.product_category_name_english,
                    p.product_category_name, 'unknown')     AS category,
           COUNT(DISTINCT oi.order_id)                       AS orders,
           SUM(oi.price + oi.freight_value)                  AS revenue
    FROM order_items oi
    JOIN orders o    ON o.order_id = oi.order_id
    JOIN products p  ON p.product_id = oi.product_id
    LEFT JOIN category_translation t
           ON t.product_category_name = p.product_category_name
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp >= '2017-01-01'
      AND o.order_purchase_timestamp <  '2018-09-01'
    GROUP BY 1
)
SELECT category,
       orders,
       ROUND(revenue, 2)                                       AS revenue,
       ROUND(100.0 * revenue / SUM(revenue) OVER (), 2)        AS revenue_share_pct,
       ROUND(100.0 * SUM(revenue) OVER (ORDER BY revenue DESC)
                   / SUM(revenue) OVER (), 2)                  AS cumulative_share_pct,
       RANK() OVER (ORDER BY revenue DESC)                     AS revenue_rank
FROM cat
ORDER BY revenue DESC
LIMIT 15;


-- name: 07_revenue_by_state
-- Revenue, customers, AOV and delivery experience by customer state.
SELECT customer_state,
       COUNT(DISTINCT customer_unique_id)                     AS customers,
       COUNT(*)                                               AS orders,
       ROUND(SUM(order_total), 2)                             AS revenue,
       ROUND(AVG(order_total), 2)                             AS avg_order_value,
       ROUND(AVG(delivery_days), 1)                           AS avg_delivery_days,
       ROUND(100.0 * AVG(is_late), 1)                         AS late_pct,
       ROUND(AVG(review_score), 2)                            AS avg_review_score
FROM v_orders_enriched
WHERE order_status = 'delivered'
  AND purchase_ts >= '2017-01-01' AND purchase_ts < '2018-09-01'
GROUP BY customer_state
ORDER BY revenue DESC;


-- name: 08_top_sellers
-- Top 20 sellers by revenue (min. 30 orders), with service quality.
SELECT oi.seller_id,
       s.seller_state,
       COUNT(DISTINCT oi.order_id)                            AS orders,
       ROUND(SUM(oi.price), 2)                                AS revenue,
       ROUND(AVG(e.review_score), 2)                          AS avg_review_score,
       ROUND(100.0 * AVG(e.is_late), 1)                       AS late_pct
FROM order_items oi
JOIN sellers s            ON s.seller_id = oi.seller_id
JOIN v_orders_enriched e  ON e.order_id = oi.order_id
WHERE e.order_status = 'delivered'
  AND e.purchase_ts >= '2017-01-01' AND e.purchase_ts < '2018-09-01'
GROUP BY oi.seller_id, s.seller_state
HAVING COUNT(DISTINCT oi.order_id) >= 30
ORDER BY revenue DESC
LIMIT 20;


-- name: 09_payment_mix
-- How customers pay, and whether installments raise basket size.
SELECT main_payment_type,
       COUNT(*)                                               AS orders,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)     AS order_share_pct,
       ROUND(AVG(order_total), 2)                             AS avg_order_value,
       ROUND(AVG(max_installments), 1)                        AS avg_installments
FROM v_orders_enriched
WHERE order_status = 'delivered'
  AND main_payment_type IS NOT NULL
GROUP BY main_payment_type
ORDER BY orders DESC;


-- name: 10_repeat_customer_rate
-- Share of customers who ever came back, and how much more they spend.
-- (Expect this to be low on Olist; it is the motivation for Phase 3.)
SELECT CASE WHEN frequency = 1 THEN 'one-time' ELSE 'repeat' END   AS customer_type,
       COUNT(*)                                                    AS customers,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)          AS customer_share_pct,
       ROUND(AVG(monetary), 2)                                     AS avg_lifetime_spend,
       ROUND(100.0 * SUM(monetary) / SUM(SUM(monetary)) OVER (), 2) AS revenue_share_pct
FROM v_customer_summary
GROUP BY 1
ORDER BY 1;
