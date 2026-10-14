"""Variability tests on normalised light curves: slope outliers, F-test, autocorrelation."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .config import VariabilityConfig

try:
    from numpy.exceptions import RankWarning
except ImportError:  # NumPy < 1.25
    RankWarning = np.RankWarning

warnings.filterwarnings("ignore", category=RankWarning)


def sigma_clip_polyfit(t, y, deg, sigma=3.0, max_iters=5):
    """Iterative kappa-sigma clipped polynomial fit; returns (coefficients, inlier mask)."""
    mask = np.ones(len(y), dtype=bool)
    min_pts = deg + 1
    coeffs = np.polyfit(t, y, deg)
    for _ in range(max_iters):
        if mask.sum() < min_pts:
            break
        coeffs = np.polyfit(t[mask], y[mask], deg)
        residuals = y - np.polyval(coeffs, t)
        std = np.std(residuals[mask])
        if std == 0:
            break
        new_mask = np.abs(residuals) < sigma * std
        if new_mask.sum() < min_pts or np.array_equal(new_mask, mask):
            mask = new_mask if new_mask.sum() >= min_pts else mask
            break
        mask = new_mask
    return coeffs, mask


def compute_slopes(time, rel_lcs, blended, sigma_clip):
    """Robust linear slope per star (NaN for blended or unfittable stars) and its population mean/std."""
    t_c = time - np.median(time)
    slopes = np.full(len(rel_lcs), np.nan)
    for k, lc in enumerate(rel_lcs):
        if k in blended:
            continue
        try:
            m = np.isfinite(lc) & (lc > 0)
            slopes[k] = sigma_clip_polyfit(t_c[m], lc[m], 1, sigma=sigma_clip)[0][0]
        except Exception:
            pass
    return slopes, float(np.nanmean(slopes)), float(np.nanstd(slopes))


def autocorr_significance(resid, max_lag):
    """Minimum over lags 1..max_lag of the z-score of the residual autocorrelation.

    White noise gives r_k ~ N(0, 1/N), so z_k = r_k * sqrt(N). Real variability stays
    correlated over several frames, hence the weakest lag is the statistic.
    Returns (min z, lag of min z).
    """
    resid = np.asarray(resid, float)
    resid = resid[np.isfinite(resid)]
    n = len(resid)
    if n < max(10, 3 * max_lag):
        return 0.0, 0
    resid = resid - resid.mean()
    denom = np.sum(resid ** 2)
    if denom <= 0:
        return 0.0, 0
    zs = [np.sum(resid[:-k] * resid[k:]) / denom * np.sqrt(n) for k in range(1, max_lag + 1)]
    weakest = int(np.argmin(zs))
    return float(zs[weakest]), weakest + 1


@dataclass
class StarResult:
    flagged: bool = False
    reasons: list[str] = field(default_factory=list)
    slope: float = np.nan
    f_stat: float = 0.0
    autocorr_z: float = 0.0
    autocorr_lag: int = 0
    # fit details used for plotting
    t: np.ndarray | None = None
    y: np.ndarray | None = None
    inliers: np.ndarray | None = None
    lin: np.ndarray | None = None
    quad: np.ndarray | None = None
    sigma: float = np.nan
    chi2_lin: float = np.nan
    chi2_quad: float = np.nan
    dof_lin: int = 0
    dof_quad: int = 0


def evaluate_star(time, lc, blended: bool, slope_mean: float, slope_std: float,
              cfg: VariabilityConfig) -> StarResult:
    """Linear vs quadratic F-test, slope-outlier test and residual autocorrelation for one star."""
    res = StarResult()
    if blended:
        return res
    m = np.isfinite(lc)
    y, t = lc[m], (time - np.median(time))[m]
    if len(y) < cfg.min_points:
        return res

    lin, lin_in = sigma_clip_polyfit(t, y, 1, sigma=cfg.sigma_clip)
    quad, quad_in = sigma_clip_polyfit(t, y, 2, sigma=cfg.sigma_clip)
    inliers = lin_in & quad_in
    y_in, t_in = y[inliers], t[inliers]
    lin_res = y_in - np.polyval(lin, t_in)
    quad_res = y_in - np.polyval(quad, t_in)

    sigma = max(np.sqrt(np.mean(lin_res ** 2)),
                np.median(np.abs(lin_res - np.median(lin_res))) / 0.6745,
                np.std(y_in) / 10.0)
    chi2_l = float(np.sum((lin_res / sigma) ** 2))
    chi2_q = float(np.sum((quad_res / sigma) ** 2))
    dof_l, dof_q = len(y_in) - 2, len(y_in) - 3
    f_stat = (chi2_l - chi2_q) / (chi2_q / dof_q) if chi2_q > 0 and dof_q > 0 else 0.0

    sig_slope = bool(slope_std > 0 and abs(lin[0] - slope_mean) / slope_std > cfg.slope_sigma_threshold)
    nonlinear = bool(f_stat > cfg.f_threshold)
    z, lag = autocorr_significance(y - np.polyval(quad, t), cfg.autocorr_max_lag)
    autocorr = bool(z > cfg.autocorr_threshold)

    res.reasons = [n for n, f in (("slope", sig_slope), ("nonlinear", nonlinear), ("autocorr", autocorr)) if f]
    res.flagged = bool(res.reasons)
    res.slope, res.f_stat, res.autocorr_z, res.autocorr_lag = float(lin[0]), float(f_stat), z, lag
    res.t, res.y, res.inliers, res.lin, res.quad = t, y, inliers, lin, quad
    res.sigma, res.chi2_lin, res.chi2_quad, res.dof_lin, res.dof_quad = float(sigma), chi2_l, chi2_q, dof_l, dof_q
    return res
