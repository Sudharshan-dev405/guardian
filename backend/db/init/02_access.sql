-- Role-based access (Stage 5.3), enforced by the database itself.
--
-- Two logins:
--   guardian      owner; used by the pipeline/simulator to write. Not limited.
--   guardian_app  used by the website backend. Limited by the policies below.
-- The backend says who is asking with:   SET app.user_id = '<users.id>';
-- (inside a transaction: SET LOCAL). With no user set, it sees nothing.
--
--   admin      sees everything
--   wearer     sees only their own data
--   caregiver  sees only the wearers linked to them, and can acknowledge their alerts

CREATE ROLE guardian_app LOGIN PASSWORD 'guardian_app_dev';
GRANT CONNECT ON DATABASE guardian TO guardian_app;
GRANT USAGE ON SCHEMA public TO guardian_app;
GRANT SELECT ON users, caregiver_links, devices, runs, stream_outputs, alerts TO guardian_app;
GRANT UPDATE (status, acknowledged_by, acknowledged_at) ON alerts TO guardian_app;
GRANT INSERT ON audit_log TO guardian_app;
GRANT USAGE ON SEQUENCE audit_log_id_seq TO guardian_app;

CREATE FUNCTION app_user_id() RETURNS int
LANGUAGE sql STABLE AS $$
    SELECT NULLIF(current_setting('app.user_id', true), '')::int
$$;

-- runs as the owner, so it can read users/links without being limited itself
CREATE FUNCTION app_can_see(wid int) RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
    SELECT EXISTS (
        SELECT 1 FROM users u
        WHERE u.id = app_user_id()
          AND (u.role = 'admin'
               OR (u.role = 'wearer' AND u.id = wid)
               OR (u.role = 'caregiver' AND EXISTS (
                     SELECT 1 FROM caregiver_links l
                     WHERE l.caregiver_id = u.id AND l.wearer_id = wid)))
    )
$$;

CREATE FUNCTION app_is_admin() RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
    SELECT EXISTS (SELECT 1 FROM users WHERE id = app_user_id() AND role = 'admin')
$$;

ALTER TABLE users           ENABLE ROW LEVEL SECURITY;
ALTER TABLE caregiver_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE devices         ENABLE ROW LEVEL SECURITY;
ALTER TABLE runs            ENABLE ROW LEVEL SECURITY;
ALTER TABLE stream_outputs  ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts          ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log       ENABLE ROW LEVEL SECURITY;

-- people: yourself, the people you can see, and everyone if admin
CREATE POLICY users_read ON users FOR SELECT TO guardian_app
    USING (id = app_user_id() OR app_can_see(id)
           OR EXISTS (SELECT 1 FROM caregiver_links l      -- a wearer sees their caregivers
                      WHERE l.caregiver_id = users.id AND l.wearer_id = app_user_id()));
CREATE POLICY links_read ON caregiver_links FOR SELECT TO guardian_app
    USING (app_is_admin() OR caregiver_id = app_user_id() OR wearer_id = app_user_id());

CREATE POLICY devices_read ON devices        FOR SELECT TO guardian_app USING (app_can_see(wearer_id));
CREATE POLICY runs_read    ON runs           FOR SELECT TO guardian_app USING (app_can_see(wearer_id));
CREATE POLICY outputs_read ON stream_outputs FOR SELECT TO guardian_app USING (app_can_see(wearer_id));
CREATE POLICY alerts_read  ON alerts         FOR SELECT TO guardian_app USING (app_can_see(wearer_id));

-- only caregivers (linked) and admins acknowledge or resolve alerts
CREATE POLICY alerts_ack ON alerts FOR UPDATE TO guardian_app
    USING (app_can_see(wearer_id)
           AND EXISTS (SELECT 1 FROM users WHERE id = app_user_id() AND role IN ('caregiver', 'admin')))
    WITH CHECK (acknowledged_by = app_user_id());

CREATE POLICY audit_write ON audit_log FOR INSERT TO guardian_app WITH CHECK (user_id = app_user_id());
