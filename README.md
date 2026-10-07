# Guardian

Wrist-worn band that estimates emergency risk for older adults living alone.
The band senses and sends; everything below runs on the paired laptop.

## Layout

| Folder | What is in it | Owner |
|---|---|---|
| `contract.py` | the interface every stream follows | |
| `data/` | dataset loaders (UMAFall, FallAllD, WEDA-FALL); `data/raw/` is not in git | Pranavah |
| `streams/context.py` | activity module (four states + unknown) | Pranavah |
| `streams/motion.py` | motion module (impact gate, impact score, stillness) | Pranavah |
| `streams/physiological.py` | heart-rate module, placeholder | Mithuna |
| `core/pipeline.py` | runs every stream on each window; new modules plug in here | Pranavah |
| `core/store.py` | saves outputs in Postgres, reads them back as a given user | Pranavah |
| `core/fusion.py` | fusion and decision manager, placeholder | Pranavah |
| `simulator/` | SIMULATED scenarios (real WEDA-FALL motion + synthetic heart rate) | Pranavah |
| `backend/db/init/` | database tables, role-based access rules, demo users | Pranavah |
| `frontend/` | web prototype, three roles, placeholder | Sudharshan |
| `models/` | models in use; older ones in `models/archive/` | |
| `testing/` | every experiment: model choice, tuning, cross-dataset tests | Pranavah |
| `testing/outputs/` | results, figures, `motion_module_log.txt`, Excel, `xgboost.txt` | |
| `testing/tools/` | probes and debugging tools | |
| `testing/old/` | September notes and scratch scripts, kept for reference | |
| `coursework/` | course assignment files that used the project data | |

## Database (Postgres in Docker)

    copy .env.example .env
    docker compose up -d
    pip install "psycopg[binary]"
    py -m core.store --ping

Demo users are simulated. Roles: admin sees everything, a wearer sees only
themselves, a caregiver sees only the wearers linked to them. The database
enforces this itself (row-level security in `02_access.sql`).

## Run (from the repo root)

    py -m streams.motion --selftest --model models/motion.joblib
    py -m streams.context --selftest
    py testing/test_contract.py
    py -m core.pipeline              (which streams load)
    py -m simulator.run all          (replay the simulated scenarios)
    py -m simulator.run all --db     (same, and save the runs in Postgres)
    py -m core.store --check         (what each demo user can see)
    py -m testing.stage4 e3          (any experiment: py -m testing.<script> <step>)

The full history of the motion module, with every number and decision, is in
`testing/outputs/motion_module_log.txt`.
