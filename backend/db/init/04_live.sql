-- Live runs (Stage 5.9). Safe to run more than once.
-- New database: runs automatically. Existing database: run it once with
--   docker exec -i guardian-db psql -U guardian -d guardian < backend/db/init/04_live.sql
ALTER TABLE runs ADD COLUMN IF NOT EXISTS status   text NOT NULL DEFAULT 'done';   -- live / done / stopped
ALTER TABLE runs ADD COLUMN IF NOT EXISTS ended_at timestamptz;
CREATE INDEX IF NOT EXISTS runs_status_idx ON runs (status);
