"""Seeing-aware photometry: per-frame PSF fit, aperture following the seeing, poor frames dropped.

Usage: python 03_seeing_aware_photometry.py [data_root]
"""
import logging
import sys
from pathlib import Path

from variabledetection import Config, TargetPipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\obsdata")
TARGET = "M31"

cfg = Config()
cfg.camera.gain = 0.28
cfg.psf.fwhm_method = "photutils"          # Gaussian PRF fit; "sep" is the faster half-light estimate
cfg.psf.max_fwhm_factor = 1.5              # drop frames with FWHM > 1.5 x median
cfg.photometry.aperture_mode = "per_frame"  # aperture radius = aperture_fwhm_factor x each frame's FWHM
cfg.photometry.aperture_fwhm_factor = 2.0

result = TargetPipeline(ROOT / TARGET, cfg).run()
print(f"{len(result.flagged)} flagged: {result.flagged}")
print("Frames dropped:")
for name, reason in result.skipped_frames.items():
    print(f"  {name}: {reason}")
print(f"Per-frame FWHM and aperture radius: {result.output_dir / 'frames.csv'}")
