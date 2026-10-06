"""
tools/inspect_weda.py -- print what the WEDA-FALL download looks like, so the
loader can be written against the real files.

Run from the repo root (works in cmd or PowerShell):
    py tools\\inspect_weda.py
"""
from collections import Counter
from pathlib import Path

ROOT = Path(r"data\raw\WEDA-FALL")
DATA = ROOT / "dataset"

print("== folders (2 levels) ==")
for d in sorted(p for p in DATA.glob("*") if p.is_dir()):
    subs = sorted(p.name for p in d.iterdir() if p.is_dir())
    print(f"{d.name}/  ({len(subs)} subfolders) {subs[:12]}{' ...' if len(subs) > 12 else ''}")

files = [p for p in DATA.rglob("*") if p.is_file()]
print(f"\n== {len(files)} files ==")
print("by extension:", Counter(p.suffix for p in files).most_common())
print("by top folder:", Counter(p.relative_to(DATA).parts[0] for p in files).most_common())

print("\n== first 12 file paths ==")
for p in sorted(files)[:12]:
    print(" ", p.relative_to(ROOT))

endings = Counter("_".join(p.stem.split("_")[2:]) or p.stem for p in files)
print("\n== file name endings (sensor types) ==")
print(endings.most_common(15))


def head(pattern):
    hits = sorted(DATA.rglob(pattern))
    if not hits:
        print(f"\n(no file matching {pattern})")
        return
    p = hits[0]
    print(f"\n== first 10 lines of {p.relative_to(ROOT)} ==")
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= 10:
                break
            print("  ", line.rstrip())


head("*accel*.csv")
head("*gyro*.csv")

for name in ("README.md", "readme.md", "README.txt"):
    r = ROOT / name
    if r.exists():
        print(f"\n== {name} ==")
        print(r.read_text(encoding="utf-8", errors="replace"))
        break

for extra in sorted(ROOT.glob("*.csv")) + sorted(ROOT.glob("*.xlsx")):
    print(f"\n(also in repo root: {extra.name})")
