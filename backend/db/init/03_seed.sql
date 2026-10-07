-- SIMULATED people for the demo. Not real persons.
INSERT INTO users (email, full_name, role) VALUES
    ('admin@guardian.demo',      'Admin (demo)',              'admin'),     -- 1
    ('wearer1@guardian.demo',    'Wearer 1 (demo, age 78)',   'wearer'),    -- 2
    ('wearer2@guardian.demo',    'Wearer 2 (demo, age 84)',   'wearer'),    -- 3
    ('wearer3@guardian.demo',    'Wearer 3 (demo, age 71)',   'wearer'),    -- 4
    ('caregiver1@guardian.demo', 'Caregiver 1 (demo, daughter)', 'caregiver'),  -- 5
    ('caregiver2@guardian.demo', 'Caregiver 2 (demo, nurse)',    'caregiver');  -- 6

INSERT INTO caregiver_links (caregiver_id, wearer_id, relation) VALUES
    (5, 2, 'daughter'),
    (6, 2, 'nurse'),
    (6, 3, 'nurse');
-- Wearer 3 has no caregiver linked: only the admin and Wearer 3 see their data.

INSERT INTO devices (wearer_id, label) VALUES
    (2, 'band-001 (simulated)'),
    (3, 'band-002 (simulated)'),
    (4, 'band-003 (simulated)');
