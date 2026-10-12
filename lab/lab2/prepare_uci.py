#!/usr/bin/env python
"""Convert the UCI Banknote Authentication source file to lab2 CSV format."""
import argparse
import csv
import hashlib
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()

rows = []
with args.source.open(newline="", encoding="utf-8") as file:
    for line_number, line in enumerate(file, 1):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise SystemExit(f"line {line_number}: expected 4 features and one label")
        values = [float(field) for field in fields[:4]]
        label = int(fields[4])
        if label not in (0, 1):
            raise SystemExit(f"line {line_number}: label must be 0 or 1")
        rows.append([label, *values])

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["label", "x0", "x1", "x2", "x3"])
    writer.writerows(rows)

counts = {label: sum(row[0] == label for row in rows) for label in (0, 1)}
digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
print(f"samples={len(rows)} class0={counts[0]} class1={counts[1]}")
print(f"source_sha256={digest}")
print(f"output={args.output}")
