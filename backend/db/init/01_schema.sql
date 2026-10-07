-- Guardian schema (Stage 5.3). Plain SQL, no web framework assumed.
-- Everything in the demo is SIMULATED: simulated people, simulated runs.

CREATE TYPE user_role    AS ENUM ('wearer', 'caregiver', 'admin');
CREATE TYPE alert_status AS ENUM ('open', 'acknowledged', 'resolved', 'false_alarm');

CREATE TABLE users (
    id            serial PRIMARY KEY,
    email         text UNIQUE NOT NULL,
    full_name     text NOT NULL,
    role          user_role NOT NULL,
    password_hash text,                       -- set by the backend later (Stage 8)
    simulated     boolean NOT NULL DEFAULT true,
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- who looks after whom: the basis of caregiver access
CREATE TABLE caregiver_links (
    caregiver_id  int NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    wearer_id     int NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    relation      text,
    PRIMARY KEY (caregiver_id, wearer_id)
);

CREATE TABLE devices (
    id            serial PRIMARY KEY,
    wearer_id     int NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    label         text NOT NULL,
    kind          text NOT NULL DEFAULT 'simulated',     -- later: 'band'
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- one replay session (a simulated scenario now, a real wear session later)
CREATE TABLE runs (
    id            serial PRIMARY KEY,
    wearer_id     int NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id     int REFERENCES devices(id) ON DELETE SET NULL,
    scenario      text,
    simulated     boolean NOT NULL DEFAULT true,
    truth         jsonb,                      -- ground truth, demos only
    started_at    timestamptz NOT NULL DEFAULT now(),
    notes         text
);

-- one row per stream per window; stream-specific values live in extras
CREATE TABLE stream_outputs (
    id            bigserial PRIMARY KEY,
    run_id        int NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    wearer_id     int NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    t_s           real NOT NULL,              -- seconds from run start (window end)
    stream        text NOT NULL,              -- motion / activity / physiological / ...
    score         real NOT NULL CHECK (score BETWEEN 0 AND 1),
    quality       real NOT NULL CHECK (quality BETWEEN 0 AND 1),
    extras        jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX stream_outputs_run_idx ON stream_outputs (run_id, stream, t_s);
CREATE INDEX stream_outputs_wearer_idx ON stream_outputs (wearer_id);

-- filled by fusion in Stage 7; empty until then
CREATE TABLE alerts (
    id               serial PRIMARY KEY,
    run_id           int REFERENCES runs(id) ON DELETE CASCADE,
    wearer_id        int NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    t_s              real,
    kind             text NOT NULL,           -- hard fall, long lie, faint, HR, SOS, ...
    emergency_score  real CHECK (emergency_score BETWEEN 0 AND 1),
    evidence         jsonb NOT NULL DEFAULT '{}',   -- which streams said what
    status           alert_status NOT NULL DEFAULT 'open',
    acknowledged_by  int REFERENCES users(id),
    acknowledged_at  timestamptz,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX alerts_wearer_idx ON alerts (wearer_id, status);

CREATE TABLE audit_log (
    id         bigserial PRIMARY KEY,
    user_id    int REFERENCES users(id),
    action     text NOT NULL,                 -- viewed / acknowledged / ...
    target     text,
    details    jsonb NOT NULL DEFAULT '{}',
    at         timestamptz NOT NULL DEFAULT now()
);
