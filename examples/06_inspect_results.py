"""Inspect saved results: plot the normalised light curve and raw flux of one star.

Usage: python 06_inspect_results.py [data_root] [star_id]
Without a star id, the first flagged star is shown.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\obsdata")
TARGET = "M31"
out = ROOT / TARGET / "output_light_curves"

with open(out / "flagged_stars.csv", newline="") as f:
    flagged = [int(r["star_id"]) for r in csv.DictReader(f)]
star = int(sys.argv[2]) if len(sys.argv) > 2 else (flagged[0] if flagged else 0)

raw = np.load(out / "light_curves.npz")
norm = np.load(out / "normalized_light_curves.npz")
t = norm["time"] - norm["time"].min()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.errorbar(t, raw["flux"][star], raw["error"][star], fmt=".", color="gray")
ax1.set_ylabel("Raw flux (ADU)")
ax2.errorbar(t, norm["relative_flux"][star], norm["relative_error"][star], fmt="o", ms=3)
ax2.set_ylabel("Relative flux")
ax2.set_xlabel("Time since first frame (days)")
x, y = raw["positions"][star]
fig.suptitle(f"{TARGET} star {star} at ({x:.0f}, {y:.0f})  flagged: {star in flagged}")
plt.show()
