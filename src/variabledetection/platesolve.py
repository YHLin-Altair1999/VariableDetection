"""Plate solving of the reference frame (astrometry.net) to get RA/Dec of stars."""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from astropy.wcs import WCS

from .config import PlateSolveConfig

log = logging.getLogger(__name__)


def solve_reference(xy: np.ndarray, flux_order: np.ndarray, shape: tuple[int, int],
                    cfg: PlateSolveConfig) -> Optional[WCS]:
    """Solve from the ``max_sources`` brightest star positions of the reference frame.

    Only the source list is uploaded, not the image. Returns a WCS for 0-based pixel
    coordinates, or None if solving is impossible or fails.
    """
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        log.warning("Plate solving skipped: environment variable %s is not set", cfg.api_key_env)
        return None
    try:
        from astroquery.astrometry_net import AstrometryNet
    except ImportError:
        log.warning("Plate solving skipped: astroquery is not installed")
        return None

    ny, nx = shape
    brightest = xy[np.argsort(flux_order)[::-1][:cfg.max_sources]]
    kwargs = {}
    if cfg.scale_lower is not None and cfg.scale_upper is not None:
        kwargs.update(scale_units="arcsecperpix", scale_type="ul",
                      scale_lower=cfg.scale_lower, scale_upper=cfg.scale_upper)
    if cfg.ra_hint is not None and cfg.dec_hint is not None:
        kwargs.update(center_ra=cfg.ra_hint, center_dec=cfg.dec_hint, radius=cfg.radius_hint)

    ast = AstrometryNet()
    ast.api_key = api_key
    try:
        # astrometry.net uses 1-based pixel coordinates
        header = ast.solve_from_source_list(brightest[:, 0] + 1, brightest[:, 1] + 1, nx, ny,
                                            solve_timeout=cfg.solve_timeout, **kwargs)
    except Exception as exc:
        log.warning("Plate solving failed: %s", exc)
        return None
    if not header:
        log.warning("Plate solving failed: astrometry.net returned no solution")
        return None
    return WCS(header)


def pixel_to_radec(wcs: WCS, xy: np.ndarray) -> np.ndarray:
    """(n, 2) 0-based pixel positions -> (n, 2) RA, Dec in degrees."""
    xy = np.atleast_2d(xy)
    ra, dec = wcs.all_pix2world(xy[:, 0] + 1, xy[:, 1] + 1, 1)
    return np.column_stack([ra, dec])
