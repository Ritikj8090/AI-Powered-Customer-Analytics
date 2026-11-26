-- Read-only database user for the deployed dashboard and AI assistant.
-- Run once as the database owner (e.g. in Neon's SQL editor), then use
-- this user's credentials in the dashboard's DATABASE_URL.
-- Defence in depth: the assistant already runs queries in READ ONLY
-- transactions, but this user cannot write even if that were bypassed.

CREATE ROLE dashboard_reader WITH LOGIN PASSWORD 'change-me-to-a-long-password';
GRANT CONNECT ON DATABASE neondb TO dashboard_reader;   -- your database name
GRANT USAGE ON SCHEMA public TO dashboard_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO dashboard_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dashboard_reader;
ALTER ROLE dashboard_reader SET statement_timeout = '15s';
