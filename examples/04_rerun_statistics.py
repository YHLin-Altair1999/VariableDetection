"""Redo normalisation, variability tests and plots with new thresholds, without redoing photometry.

Usage: python 04_rerun_statistics.py [data_root]
Needs an earlier run of the same target (light_curves.npz in the output folder).
"""
import logging
import sys
from pathlib import Path

from variabledetection import Config, TargetPipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\obsdata")
TARGET = "M31"

cfg = Config()
cfg.variability.f_threshold = 30.0           # be more sensitive than the default 50
cfg.variability.autocorr_threshold = 4.0
cfg.normalization.max_comparison_distance = 500.0

result = TargetPipeline(ROOT / TARGET, cfg).run(reuse=True)
print(f"{len(result.flagged)} flagged: {result.flagged}")
