"""Plate solve the reference frame (astrometry.net) and print RA/Dec of the flagged stars.

Usage: python 05_plate_solve_flagged.py [data_root]
Needs the environment variable ASTROMETRY_API_KEY (your nova.astrometry.net API key) and network access.
Reuses saved light curves, so run one of the other examples on the target first.
"""
import csv
import logging
import os
import sys
from pathlib import Path

from variabledetection import Config, TargetPipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\obsdata")
TARGET = "M31"

if not os.environ.get("ASTROMETRY_API_KEY"):
    sys.exit("Set the ASTROMETRY_API_KEY environment variable first.")

cfg = Config()
cfg.platesolve.enabled = True
cfg.platesolve.scale_lower, cfg.platesolve.scale_upper = 0.5, 2.0   # arcsec/px hint; speeds up solving
# cfg.platesolve.ra_hint, cfg.platesolve.dec_hint = 10.68, 41.27    # deg, optional

result = TargetPipeline(ROOT / TARGET, cfg).run(reuse=True)

with open(result.output_dir / "flagged_stars.csv", newline="") as f:
    for row in csv.DictReader(f):
        coords = f"RA {row['ra_deg']}  Dec {row['dec_deg']}" if "ra_deg" in row else "(plate solve failed)"
        print(f"star {row['star_id']:>4}  x={row['x']:>8} y={row['y']:>8}  {row['reasons']:<20} {coords}")
