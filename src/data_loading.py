from pathlib import Path

raw = Path("data/raw")

files = [
    "adult.data",
    "adult.test",
    "adult.names",
    "german.data",
    "german.data-numeric",
]

for f in files:
    p = raw / f
    print(f"{f:20} exists: {p.exists()}, size: {p.stat().st_size if p.exists() else 'MISSING'}")

