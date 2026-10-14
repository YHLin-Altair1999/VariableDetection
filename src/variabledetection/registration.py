"""Frame registration (astroalign) and the common photometry region."""
from __future__ import annotations

import logging
from typing import Optional

import astroalign as aa
import numpy as np

from .config import RegistrationConfig

log = logging.getLogger(__name__)


def register(src_xy: np.ndarray, ref_xy: np.ndarray, cfg: RegistrationConfig
             ) -> tuple[Optional[np.ndarray], int, str]:
    """Similarity transform mapping frame pixel coords to reference pixel coords.

    Returns (3x3 matrix or None, number of matched stars, failure reason or "").
    """
    n = cfg.max_control_points
    if len(src_xy) < max(3, cfg.min_matches) or len(ref_xy) < 3:
        return None, 0, f"too few sources ({len(src_xy)})"
    try:
        transform, (matched, _) = aa.find_transform(src_xy[:n], ref_xy[:n])
    except (aa.MaxIterError, ValueError, TypeError) as exc:
        return None, 0, f"astroalign failed: {exc}"
    matrix = np.asarray(transform.params, float)
    n_matched = len(matched)
    scale = float(np.sqrt(abs(np.linalg.det(matrix[:2, :2]))))
    if n_matched < cfg.min_matches:
        return None, n_matched, f"only {n_matched} matched stars"
    if abs(scale - 1.0) > cfg.max_scale_deviation:
        return None, n_matched, f"implausible scale {scale:.3f}"
    return matrix, n_matched, ""


def apply_matrix(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, float)
    return xy @ matrix[:2, :2].T + matrix[:2, 2]


def footprint(shape: tuple[int, int], margin: float, matrix: np.ndarray) -> np.ndarray:
    """Corners (4, 2) of a frame, shrunk by ``margin`` px, in reference coordinates."""
    ny, nx = shape
    corners = np.array([[margin - 0.5, margin - 0.5], [nx - margin - 0.5, margin - 0.5],
                        [nx - margin - 0.5, ny - margin - 0.5], [margin - 0.5, ny - margin - 0.5]])
    return apply_matrix(matrix, corners)


def _ccw(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return poly if area >= 0 else poly[::-1]


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping of a polygon by a convex polygon."""
    out = [tuple(p) for p in _ccw(subject)]
    clip = _ccw(clipper)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if not out:
            break
        inp, out = out, []
        edge = b - a

        def side(p):
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])

        for p, q in zip(inp, inp[1:] + inp[:1]):
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return np.array(out) if len(out) >= 3 else np.empty((0, 2))


def intersect_footprints(quads: list[np.ndarray]) -> np.ndarray:
    poly = _ccw(quads[0])
    for q in quads[1:]:
        poly = _clip(poly, q)
        if len(poly) < 3:
            return np.empty((0, 2))
    return poly


def _vertical_extent(poly: np.ndarray, xs: np.ndarray):
    lo = np.full(len(xs), np.inf)
    hi = np.full(len(xs), -np.inf)
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        if p[0] == q[0]:
            continue
        a, b = sorted((p[0], q[0]))
        inside = (xs >= a) & (xs <= b)
        y = p[1] + (xs - p[0]) * (q[1] - p[1]) / (q[0] - p[0])
        lo = np.where(inside, np.minimum(lo, y), lo)
        hi = np.where(inside, np.maximum(hi, y), hi)
    return lo, hi


def largest_rectangle(poly: np.ndarray, n_candidates: int = 400
                      ) -> Optional[tuple[float, float, float, float]]:
    """Largest axis-aligned rectangle (x0, y0, x1, y1) inside a convex polygon.

    For a convex polygon the lower boundary is convex and the upper one concave, so for
    a given x-range the best y-range is fixed by the two end columns.
    """
    if len(poly) < 3:
        return None
    xs = np.unique(np.concatenate([poly[:, 0], np.linspace(poly[:, 0].min(), poly[:, 0].max(), n_candidates)]))
    lo, hi = _vertical_extent(poly, xs)
    y0 = np.maximum(lo[:, None], lo[None, :])
    y1 = np.minimum(hi[:, None], hi[None, :])
    width = xs[None, :] - xs[:, None]
    area = np.where(np.isfinite(y0) & np.isfinite(y1) & (y1 > y0) & (width > 0), width * (y1 - y0), 0.0)
    i, j = np.unravel_index(np.argmax(area), area.shape)
    if area[i, j] <= 0:
        return None
    return float(xs[i]), float(y0[i, j]), float(xs[j]), float(y1[i, j])


def _rect_area(rect) -> float:
    return 0.0 if rect is None else (rect[2] - rect[0]) * (rect[3] - rect[1])


def common_region(quads: dict[int, np.ndarray], full_area: float, cfg: RegistrationConfig,
                  protected: int) -> tuple[Optional[tuple[float, float, float, float]], list[int]]:
    """Largest rectangle covered by every frame; greedily drops the frames whose footprint
    shrinks it most while it is smaller than ``min_region_fraction`` of the reference frame.

    ``quads`` maps frame index -> footprint. The reference frame (``protected``) is never dropped.
    Returns (rectangle or None, dropped frame indices).
    """
    active = dict(quads)
    dropped: list[int] = []
    max_drop = int(cfg.max_dropped_fraction * len(quads))

    def region_of(frames):
        return largest_rectangle(intersect_footprints([frames[k] for k in frames]))

    rect = region_of(active)
    while _rect_area(rect) < cfg.min_region_fraction * full_area and len(dropped) < max_drop:
        centres = {k: q.mean(axis=0) for k, q in active.items() if k != protected}
        if not centres:
            break
        median = np.median(np.array(list(centres.values())), axis=0)
        worst = sorted(centres, key=lambda k: -np.hypot(*(centres[k] - median)))[:10]
        best_k, best_rect = None, rect
        for k in worst:
            trial = {m: q for m, q in active.items() if m != k}
            r = region_of(trial)
            if _rect_area(r) > _rect_area(best_rect):
                best_k, best_rect = k, r
        if best_k is None:
            break
        del active[best_k]
        dropped.append(best_k)
        rect = best_rect
    return rect, dropped
