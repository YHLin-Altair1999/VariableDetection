"""Process one target with settings written directly in Python.

Usage: python 01_single_target.py [data_root]
Expects <data_root>/M31/reduced/*.fits
"""
import logging
import sys
from pathlib import Path

from variabledetection import Config, TargetPipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\obsdata")
TARGET = "M31"

cfg = Config()
cfg.camera.gain = 0.28               # e-/ADU, drives the photon-noise error bars
cfg.variability.f_threshold = 50.0
cfg.variability.autocorr_threshold = 5.0

result = TargetPipeline(ROOT / TARGET, cfg).run()

print(f"{result.n_stars} stars, {result.n_frames_used} frames, {len(result.flagged)} flagged")
for star_id in result.flagged:
    print(f"  star {star_id}: {', '.join(result.reasons[star_id])}")
print(f"Frames not used: {result.skipped_frames}")
print(f"Results in {result.output_dir}")
