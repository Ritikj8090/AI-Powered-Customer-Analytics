-- =====================================================================
-- Olist E-Commerce: relational schema (PostgreSQL)
-- Run order matters: parent tables first, then child tables.
-- Re-running this file drops and recreates everything.
-- =====================================================================

DROP TABLE IF EXISTS order_reviews, order_payments, order_items, orders,
                     products, sellers, customers, geolocation,
                     category_translation CASCADE;

-- ---------- Lookup / dimension tables ----------

CREATE TABLE category_translation (
    product_category_name          TEXT PRIMARY KEY,
    product_category_name_english  TEXT NOT NULL
);

CREATE TABLE customers (
    customer_id               TEXT PRIMARY KEY,   -- one per ORDER, not per person
    customer_unique_id        TEXT NOT NULL,      -- the real customer identifier
    customer_zip_code_prefix  TEXT,
    customer_city             TEXT,
    customer_state            CHAR(2)
);

CREATE TABLE sellers (
    seller_id               TEXT PRIMARY KEY,
    seller_zip_code_prefix  TEXT,
    seller_city             TEXT,
    seller_state            CHAR(2)
);

-- Not every category exists in the translation file, so no FK here;
-- queries use LEFT JOIN + COALESCE instead.
CREATE TABLE products (
    product_id                  TEXT PRIMARY KEY,
    product_category_name       TEXT,
    product_name_length         INTEGER,
    product_description_length  INTEGER,
    product_photos_qty          INTEGER,
    product_weight_g            NUMERIC,
    product_length_cm           NUMERIC,
    product_height_cm           NUMERIC,
    product_width_cm            NUMERIC
);

-- Many rows per zip prefix (lat/lng samples), so no primary key.
CREATE TABLE geolocation (
    geolocation_zip_code_prefix  TEXT,
    geolocation_lat              DOUBLE PRECISION,
    geolocation_lng              DOUBLE PRECISION,
    geolocation_city             TEXT,
    geolocation_state            CHAR(2)
);

-- ---------- Fact tables ----------

CREATE TABLE orders (
    order_id                       TEXT PRIMARY KEY,
    customer_id                    TEXT NOT NULL REFERENCES customers(customer_id),
    order_status                   TEXT NOT NULL,
    order_purchase_timestamp       TIMESTAMP NOT NULL,
    order_approved_at              TIMESTAMP,
    order_delivered_carrier_date   TIMESTAMP,
    order_delivered_customer_date  TIMESTAMP,
    order_estimated_delivery_date  TIMESTAMP
);

CREATE TABLE order_items (
    order_id             TEXT    NOT NULL REFERENCES orders(order_id),
    order_item_id        INTEGER NOT NULL,     -- line number within the order
    product_id           TEXT    NOT NULL REFERENCES products(product_id),
    seller_id            TEXT    NOT NULL REFERENCES sellers(seller_id),
    shipping_limit_date  TIMESTAMP,
    price                NUMERIC(10,2) NOT NULL,
    freight_value        NUMERIC(10,2) NOT NULL,
    PRIMARY KEY (order_id, order_item_id)
);

CREATE TABLE order_payments (
    order_id              TEXT    NOT NULL REFERENCES orders(order_id),
    payment_sequential    INTEGER NOT NULL,
    payment_type          TEXT,
    payment_installments  INTEGER,
    payment_value         NUMERIC(10,2),
    PRIMARY KEY (order_id, payment_sequential)
);

-- review_id is NOT unique in the raw data (one review can cover several
-- orders), so the key is the (review_id, order_id) pair.
CREATE TABLE order_reviews (
    review_id                TEXT NOT NULL,
    order_id                 TEXT NOT NULL REFERENCES orders(order_id),
    review_score             SMALLINT CHECK (review_score BETWEEN 1 AND 5),
    review_comment_title     TEXT,
    review_comment_message   TEXT,
    review_creation_date     TIMESTAMP,
    review_answer_timestamp  TIMESTAMP,
    PRIMARY KEY (review_id, order_id)
);

-- ---------- Indexes for the joins and filters used in analysis ----------

CREATE INDEX idx_customers_unique   ON customers (customer_unique_id);
CREATE INDEX idx_orders_customer    ON orders (customer_id);
CREATE INDEX idx_orders_purchase    ON orders (order_purchase_timestamp);
CREATE INDEX idx_orders_status      ON orders (order_status);
CREATE INDEX idx_items_product      ON order_items (product_id);
CREATE INDEX idx_items_seller       ON order_items (seller_id);
CREATE INDEX idx_reviews_order      ON order_reviews (order_id);
CREATE INDEX idx_geo_zip            ON geolocation (geolocation_zip_code_prefix);
