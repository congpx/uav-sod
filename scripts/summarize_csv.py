#!/usr/bin/env python3
from __future__ import annotations
import csv
import sys
from pathlib import Path

for csv_path in map(Path, sys.argv[1:]):
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        continue
    r = rows[-1]
    print(csv_path.name)
    for k in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if k in r:
            print(f"  {k}: {r[k]}")
