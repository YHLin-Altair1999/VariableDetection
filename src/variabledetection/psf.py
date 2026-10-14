"""Per-frame PSF width (FWHM) measurement."""
from __future__ import annotations

import warnings

import numpy as np
import sep
from astropy.modeling.fitting import TRFLSQFitter
from astropy.table import Table
from photutils.psf import CircularGaussianPRF, PSFPhotometry
from scipy.spatial import cKDTree

from .config import PsfConfig

FWHM_TO_SIGMA = 2.355


def fwhm_from_half_light(sub: np.ndarray, objs, n_stars: int) -> float:
    """FWHM from the median half-light radius of the brightest sources (FWHM = 2 r50 for a Gaussian)."""
    top = objs[:n_stars]
    if len(top) < 3:
        return np.nan
    r50, flag = sep.flux_radius(sub, top["x"], top["y"], np.full(len(top), 15.0), 0.5)
    r50 = r50[(flag == 0) & np.isfinite(r50) & (r50 > 0)]
    return float(2.0 * np.median(r50)) if len(r50) else np.nan


def fwhm_from_psf_fit(sub: np.ndarray, objs, cfg: PsfConfig) -> float:
    """Median FWHM of a free-width circular Gaussian PRF fitted to isolated bright stars.

    Falls back to the half-light estimate if too few fits converge.
    """
    start = fwhm_from_half_light(sub, objs, cfg.n_stars)
    if not np.isfinite(start):
        return np.nan
    xy = np.column_stack([objs["x"], objs["y"]])
    ny, nx = sub.shape
    half = cfg.fit_shape // 2 + 1
    nearest = cKDTree(xy).query(xy, k=2)[0][:, 1] if len(xy) > 1 else np.full(len(xy), np.inf)
    ok = (nearest > cfg.fit_shape) & (xy[:, 0] >= half) & (xy[:, 0] < nx - half) \
        & (xy[:, 1] >= half) & (xy[:, 1] < ny - half)
    xy = xy[ok][:cfg.n_stars]  # objs is sorted brightest first
    if len(xy) < 3:
        return start

    model = CircularGaussianPRF(fwhm=start)
    model.fwhm.fixed = False
    model.fwhm.bounds = (0.5 * start, 3.0 * start)
    phot = PSFPhotometry(model, (cfg.fit_shape, cfg.fit_shape), fitter=TRFLSQFitter(),
                         aperture_radius=max(start, 2.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = phot(sub, init_params=Table({"x": xy[:, 0], "y": xy[:, 1]}))
        except Exception:
            return start
    fit = np.asarray(res["fwhm_fit"], float)
    good = (np.asarray(res["flags"]) == 0) & np.isfinite(fit) & (fit > model.fwhm.bounds[0]) \
        & (fit < model.fwhm.bounds[1])
    return float(np.median(fit[good])) if good.sum() >= 3 else start
