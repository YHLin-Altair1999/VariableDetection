"""Differential photometry against an ensemble of comparison stars."""
from __future__ import annotations

import logging

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.spatial import cKDTree

from .config import NormalizationConfig

log = logging.getLogger(__name__)


def find_blended(positions: np.ndarray, radius: float) -> set[int]:
    """Indices of stars with another star within ``radius`` pixels."""
    pairs = cKDTree(positions).query_pairs(radius)
    return {i for pair in pairs for i in pair}


def normalize(flux: np.ndarray, err: np.ndarray, positions: np.ndarray, blended: set[int],
              cfg: NormalizationConfig, exclude: set[int] | None = None):
    """Each star's flux divided by the ensemble of similar-brightness, non-blended stars.

    ``exclude`` holds stars (e.g. suspected variables) that may not act as comparison
    stars unless that would leave a target with none. Returns (relative flux, error),
    both (n_stars, n_frames) with each curve scaled to a median of 1.
    """
    exclude = exclude or set()
    n = flux.shape[0]
    blended_mask = np.zeros(n, bool)
    blended_mask[list(blended)] = True
    median_flux = np.nanmedian(flux, axis=1)
    rel, rel_err = [], []
    for i in range(n):
        with np.errstate(divide="ignore", invalid="ignore"):
            mask = np.abs(np.log10(np.abs(median_flux / median_flux[i]))) < cfg.brightness_window_dex
        mask[i] = False
        mask &= ~blended_mask
        if cfg.max_comparison_distance is not None:
            mask &= np.hypot(*(positions - positions[i]).T) <= cfg.max_comparison_distance
        if exclude:
            candidate = mask.copy()
            candidate[list(exclude)] = False
            if candidate.any():
                mask = candidate
        if not mask.any():
            mask = np.ones(n, bool)
            mask[i] = False

        comp = flux[mask]
        scaled = comp / np.nanmedian(comp, axis=1)[:, None]
        ensemble = sigma_clipped_stats(scaled, sigma=cfg.ensemble_sigma_clip, axis=0)[1]

        comp_rel_err = err[mask] / np.abs(comp)
        n_comp = np.sum(np.isfinite(comp_rel_err), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ens_rel_err = np.sqrt(np.nansum(comp_rel_err ** 2, axis=0)) / n_comp
            r = flux[i] / ensemble
            r = r / np.nanmedian(r)
            rel.append(r)
            rel_err.append(np.abs(r) * np.sqrt((err[i] / flux[i]) ** 2 + ens_rel_err ** 2))
    return np.array(rel), np.array(rel_err)
