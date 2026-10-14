"""Source and star detection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sep
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from .config import DetectionConfig, PsfConfig
from .psf import FWHM_TO_SIGMA, fwhm_from_half_light, fwhm_from_psf_fit  # noqa: F401


@dataclass
class FrameSources:
    xy: np.ndarray        # (n, 2) pixel positions, brightest first
    flux: np.ndarray
    fwhm: float           # PSF FWHM of this frame (nan if not requested or undetermined)
    n_detected: int


def extract_sources(data: np.ndarray, cfg: DetectionConfig, max_sources: int,
                    psf: PsfConfig | None = None) -> FrameSources:
    """Bright, round, unsaturated sources for registration; measures the frame's FWHM if ``psf`` is given."""
    bkg = sep.Background(data)
    sub = data - bkg.back()
    try:
        objs = sep.extract(sub, cfg.threshold_sigma * bkg.globalrms, minarea=5)
    except Exception:
        objs = np.empty(0, dtype=[("x", float), ("y", float), ("flux", float), ("a", float),
                                  ("b", float), ("peak", float), ("flag", int)])
    keep = (objs["flag"] == 0) & (objs["a"] / np.maximum(objs["b"], 1e-3) <= cfg.max_elongation)
    if cfg.saturation_adu is not None:
        keep &= objs["peak"] < cfg.saturation_adu
    objs = objs[keep]
    objs = objs[np.argsort(objs["flux"])[::-1]]

    fwhm = np.nan
    if psf is not None:
        fwhm = (fwhm_from_psf_fit(sub, objs, psf) if psf.fwhm_method == "photutils"
                else fwhm_from_half_light(sub, objs, psf.n_stars))
    top = objs[:max_sources]
    return FrameSources(np.column_stack([top["x"], top["y"]]), np.asarray(top["flux"]), fwhm, len(objs))


def detect_stars(data: np.ndarray, cfg: DetectionConfig, fwhm: float,
                 region: tuple[float, float, float, float]):
    """DAOStarFinder on the reference frame; keeps stars inside ``region`` (x0, y0, x1, y1).

    Returns (xy (n, 2), peak (n,)) ordered as detected.
    """
    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=max(fwhm, 2.0), threshold=cfg.threshold_sigma * std)
    table = finder(data - median)
    if table is None or len(table) == 0:
        return np.empty((0, 2)), np.empty(0)
    x, y = np.asarray(table["xcentroid"]), np.asarray(table["ycentroid"])
    peak = np.asarray(table["peak"])
    x0, y0, x1, y1 = region
    keep = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (peak >= cfg.threshold_sigma * std)  # drop noise peaks
    if cfg.saturation_adu is not None:
        keep &= (peak + median) < cfg.saturation_adu
    return np.column_stack([x[keep], y[keep]]), peak[keep]
