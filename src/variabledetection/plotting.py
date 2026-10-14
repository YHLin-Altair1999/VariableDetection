"""Diagnostic plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from auto_stretch import apply_stretch
from matplotlib.patches import Circle, Rectangle


def _frame_axes(image):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(apply_stretch(image), cmap="gray", origin="lower")
    return fig, ax


def plot_detected_stars(path: Path, image, positions, region, title, dpi):
    fig, ax = _frame_axes(image)
    for sid, (x, y) in enumerate(positions):
        ax.add_patch(Circle((x, y), 10, color="red", fill=False, lw=1.5))
        ax.text(x, y + 15, str(sid), color="red", fontsize=6, ha="center", clip_on=True)
    if region is not None:
        x0, y0, x1, y1 = region
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1.5, ls="--"))
    ax.set_title(title)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_frame_apertures(path: Path, image, pos, r_in_out, title, dpi):
    r, r_in, r_out = r_in_out
    ny, nx = image.shape
    fig, ax = _frame_axes(image)
    for sid, (x, y) in enumerate(pos):
        if not (0 <= x < nx and 0 <= y < ny):
            continue
        ax.add_patch(Circle((x, y), r, color="red", fill=False, lw=1.2))
        ax.add_patch(Circle((x, y), r_in, color="cyan", fill=False, lw=0.8, ls="--"))
        ax.add_patch(Circle((x, y), r_out, color="cyan", fill=False, lw=0.8, ls="--"))
    ax.set_title(title)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_flagged_map(path: Path, image, positions, flagged, blended, radius, region, dpi):
    ny, nx = image.shape
    fig, ax = _frame_axes(image)
    flagged = set(flagged)
    for sid, (x, y) in enumerate(positions):
        if sid in flagged:
            ax.add_patch(Circle((x, y), radius + 4, color="orange", fill=False, lw=2))
            ax.text(x, y + radius + 8, str(sid), color="orange", fontsize=8, ha="center", clip_on=True)
        else:
            b = sid in blended
            ax.add_patch(Circle((x, y), radius + 4, color="lightblue" if b else "white", fill=False,
                                lw=0.5, alpha=0.8 if b else 0.4))
    if region is not None:
        x0, y0, x1, y1 = region
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1, ls="--"))
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_title(f"Flagged stars: {len(flagged)} of {len(positions)} "
                 f"(light blue = {len(blended)} blended, skipped)")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_star_fit(path: Path, idx, res, autocorr_max_lag, dpi):
    y_in, t_in = res.y[res.inliers], res.t[res.inliers]
    s = res.sigma
    y_lo, y_hi = np.min(y_in - s), np.max(y_in + s)
    pad = 0.05 * (y_hi - y_lo)
    t_lo, t_hi = t_in.min(), t_in.max()
    tpad = 0.03 * (t_hi - t_lo)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(t_in, y_in, yerr=s, fmt="o", alpha=0.6, capsize=3, label="Data (±empirical σ)")
    out = ~res.inliers
    if out.any():
        ax.errorbar(res.t[out], res.y[out], yerr=s, fmt="x", color="k", alpha=0.8, capsize=3,
                    label="Clipped outliers")
    tl = np.linspace(t_lo, t_hi, 100)
    r_l = res.chi2_lin / res.dof_lin if res.dof_lin > 0 else np.inf
    r_q = res.chi2_quad / res.dof_quad if res.dof_quad > 0 else np.inf
    ax.plot(tl, np.polyval(res.lin, tl), "r-", lw=2.5, label=f"Linear (χ²_red={r_l:.4f})")
    ax.plot(tl, np.polyval(res.quad, tl), "g--", lw=2.5, label=f"Quadratic (χ²_red={r_q:.4f})")
    flags = {"slope": "Significant Slope", "nonlinear": f"Nonlinear (F={res.f_stat:.2f})",
             "autocorr": f"Autocorrelation (z>{res.autocorr_z:.1f} at all lags 1-{autocorr_max_lag})"}
    ax.set_title(f"Star {idx} [{' + '.join(flags[r] for r in res.reasons)}]")
    ax.set_xlabel("Time - median (days)")
    ax.set_ylabel("Relative Flux")
    ax.set_xlim(t_lo - tpad, t_hi + tpad)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_overview(path: Path, time, rel_lcs, flagged, dpi):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in flagged:
        ok = np.isfinite(rel_lcs[i]) & (rel_lcs[i] > 0)
        ax.plot(time[ok], rel_lcs[i][ok] / np.nanmedian(rel_lcs[i]), marker="o", ms=3, alpha=0.5,
                label=f"Star {i}")
    ax.legend(fontsize=6)
    ax.set_xlabel("Time (JD)")
    ax.set_ylabel("Normalized Flux")
    ax.set_ylim(0.5, 1.5)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
