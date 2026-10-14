"""Target discovery and FITS reading."""
from __future__ import annotations

import logging
from pathlib import Path

import astropy.io.fits as fits
import numpy as np
from astropy.time import Time

from .config import Config, DataConfig

log = logging.getLogger(__name__)


def find_frames(target_dir: Path, cfg: DataConfig) -> list[Path]:
    return sorted((Path(target_dir) / cfg.reduced_subdir).glob(cfg.file_pattern))


def discover_targets(root: Path, cfg: Config, force: bool = False) -> list[Path]:
    """Target folders under ``root`` holding ``<reduced_subdir>/<file_pattern>`` and, unless
    ``force``, no output folder yet. ``root`` itself is a target if it has a reduced folder."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")
    candidates = [root] if (root / cfg.data.reduced_subdir).is_dir() else sorted(
        p for p in root.iterdir() if p.is_dir())
    targets = []
    for p in candidates:
        if not find_frames(p, cfg.data):
            continue
        if not force and (p / cfg.data.output_subdir).exists():
            continue
        targets.append(p)
    return targets


def read_header(path: Path) -> fits.Header:
    """Primary header updated with the header of the first image HDU (no pixel reading)."""
    with fits.open(path) as hdul:
        header = hdul[0].header.copy()
        for hdu in hdul:
            if hdu.header.get("NAXIS", 0) >= 2 or "ZNAXIS" in hdu.header:
                header.update(hdu.header)
                break
    return header


def load_image(path: Path) -> np.ndarray:
    """First 2-D image in the file as native-endian float64."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return np.ascontiguousarray(hdu.data, dtype=np.float64)
    raise ValueError(f"No 2-D image data in {path}")


def read_time(header: fits.Header, cfg: DataConfig) -> float:
    """Mid-exposure Julian Date from the first usable keyword in ``cfg.time_keywords``."""
    for key in cfg.time_keywords:
        value = header.get(key)
        if value is None:
            continue
        try:
            k = key.upper()
            if k in ("MJD-OBS", "MJD"):
                return float(value) + 2400000.5
            if k == "JD":
                return float(value)
            exptime = float(header.get(cfg.exposure_keyword, 0.0) or 0.0)
            return float(Time(str(value)).jd) + exptime / 2.0 / 86400.0
        except (ValueError, TypeError) as exc:
            log.warning("Could not parse %s=%r: %s", key, value, exc)
    raise ValueError(f"No usable time keyword among {cfg.time_keywords}")
