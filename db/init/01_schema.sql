-- ADA Encroachment MVP — PostGIS schema.
-- Auto-run by the postgis image on first DB init (mounted at
-- /docker-entrypoint-initdb.d). Column names/types mirror gateway/app/models.py
-- 1:1. Idempotent so a re-run (or the app's create paths) never conflicts.

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(64)  NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role          VARCHAR(32)  NOT NULL DEFAULT 'analyst',
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            VARCHAR(255),
    status          VARCHAR(32)  NOT NULL DEFAULT 'pending',
    mode            VARCHAR(16)  NOT NULL DEFAULT 'ml',
    t1_path         TEXT         NOT NULL,
    t2_path         TEXT         NOT NULL,
    mask_path       TEXT,
    pixel_size_m    DOUBLE PRECISION NOT NULL DEFAULT 10.0,
    crs             VARCHAR(32),
    num_detections  INTEGER      NOT NULL DEFAULT 0,
    total_area_m2   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    inference_ms    DOUBLE PRECISION,
    error           TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS detections (
    id           SERIAL PRIMARY KEY,
    run_id       INTEGER      NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    geom         geometry(Polygon, 4326) NOT NULL,
    area_m2      DOUBLE PRECISION NOT NULL,
    confidence   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    class_label  VARCHAR(64)  NOT NULL DEFAULT 'change',
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_runs_user_id      ON runs(user_id);
CREATE INDEX IF NOT EXISTS idx_detections_run_id ON detections(run_id);
CREATE INDEX IF NOT EXISTS idx_detections_geom   ON detections USING GIST (geom);

-- Demo admin (admin/admin123) is created idempotently by the gateway on
-- startup (ensure_seed_admin), so no password hash is hardcoded here.
