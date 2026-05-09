-- Resolv.AI — PostgreSQL schema
-- Run: psql -d resolv -f scripts/create_schema.sql

BEGIN;

CREATE TABLE IF NOT EXISTS complaints (
    id                      SERIAL PRIMARY KEY,
    ticket_id               VARCHAR(50) UNIQUE NOT NULL,
    site_name               VARCHAR(100),
    zone                    VARCHAR(50),
    created_date            TIMESTAMP,
    complaint_title         TEXT NOT NULL,
    status                  VARCHAR(20),
    category                VARCHAR(100),
    sub_category            VARCHAR(200),
    issue_type              VARCHAR(50),       -- FM / Project
    created_by              VARCHAR(200),
    tower                   VARCHAR(100),
    flat                    VARCHAR(50),
    aging_days              INTEGER,
    priority                VARCHAR(10),
    response_tat_minutes    INTEGER,
    resolution_tat_minutes  INTEGER,
    response_tat_breached   BOOLEAN,
    resolution_tat_breached BOOLEAN,
    closed_date             TIMESTAMP,
    raw_data                JSONB              -- full original row
);

CREATE INDEX IF NOT EXISTS idx_flat         ON complaints(site_name, flat);
CREATE INDEX IF NOT EXISTS idx_tower        ON complaints(site_name, tower);
CREATE INDEX IF NOT EXISTS idx_category     ON complaints(category);
CREATE INDEX IF NOT EXISTS idx_created      ON complaints(created_date);
CREATE INDEX IF NOT EXISTS idx_status       ON complaints(status);
CREATE INDEX IF NOT EXISTS idx_text         ON complaints USING gin(to_tsvector('english', complaint_title));

-- Flat adjacency: above / below / lateral neighbours
CREATE TABLE IF NOT EXISTS flat_adjacency (
    site_name       VARCHAR(100)    NOT NULL,
    tower           VARCHAR(100)    NOT NULL,
    flat            VARCHAR(50)     NOT NULL,
    above_flat      VARCHAR(50),
    below_flat      VARCHAR(50),
    lateral_flats   VARCHAR(50)[],
    PRIMARY KEY (site_name, tower, flat)
);

CREATE INDEX IF NOT EXISTS idx_adj_site_tower ON flat_adjacency(site_name, tower);

COMMIT;
