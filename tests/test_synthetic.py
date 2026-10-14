"""End-to-end test on synthetic frames (random star field, shifts, rotation, one variable star)."""
import csv

import numpy as np
import pytest
from astropy.io import fits

from variabledetection.config import Config, DetectionConfig, PhotometryConfig, PsfConfig, load_config
from variabledetection.detection import extract_sources
from variabledetection.io import discover_targets
from variabledetection.pipeline import TargetPipeline
from variabledetection.registration import common_region, footprint, largest_rectangle

NX, NY = 700, 600
N_FRAMES = 30
VARIABLE = 7
BAD_FRAME = 5
BAD_SIGMA = 4.5


def _render(xy, flux, sigma=2.0, rng=None):
    img = rng.normal(1000.0, 5.0, (NY, NX))
    yy, xx = np.mgrid[0:NY, 0:NX]
    for (x, y), f in zip(xy, flux):
        if not (-20 < x < NX + 20 and -20 < y < NY + 20):
            continue
        x0, x1, y0, y1 = int(max(x - 15, 0)), int(min(x + 16, NX)), int(max(y - 15, 0)), int(min(y + 16, NY))
        if x1 <= x0 or y1 <= y0:
            continue
        g = np.exp(-((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2) / (2 * sigma ** 2))
        img[y0:y1, x0:x1] += f * g / (2 * np.pi * sigma ** 2)
    return img


@pytest.fixture()
def target(tmp_path):
    rng = np.random.default_rng(1)
    n = 60
    base = np.column_stack([rng.uniform(60, NX - 60, n), rng.uniform(60, NY - 60, n)])
    # keep stars apart so none is flagged as blended
    for i in range(n):
        for j in range(i):
            if np.hypot(*(base[i] - base[j])) < 40:
                base[i] += 45
    base = np.clip(base, 30, [NX - 30, NY - 30])
    flux0 = 10 ** rng.uniform(4.3, 5.0, n)
    folder = tmp_path / "TargetA" / "reduced"
    folder.mkdir(parents=True)
    centre = np.array([NX / 2, NY / 2])
    for k in range(N_FRAMES):
        theta = np.radians(0.3) * np.sin(k / 4)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        shift = rng.uniform(-8, 8, 2)
        xy = (base - centre) @ rot.T + centre + shift
        flux = flux0.copy()
        flux[VARIABLE] *= 0.8 if 12 <= k < 20 else 1.0  # transit-like dip
        sigma = BAD_SIGMA if k == BAD_FRAME else 2.0 + 0.3 * np.sin(k / 5)
        img = _render(xy, flux, sigma=sigma, rng=rng)
        hdr = fits.Header()
        hdr["JD"] = 2460000.5 + k * 0.002
        fits.writeto(folder / f"frame_{k:03d}.fits", img.astype(np.float32), hdr)
    return tmp_path


@pytest.mark.parametrize("fwhm_method", ["sep", "photutils"])
def test_fwhm_measurement(fwhm_method):
    rng = np.random.default_rng(3)
    xy = np.column_stack([rng.uniform(40, NX - 40, 25), rng.uniform(40, NY - 40, 25)])
    for i in range(25):
        for j in range(i):
            if np.hypot(*(xy[i] - xy[j])) < 50:
                xy[i] = (xy[i] + 60) % [NX - 60, NY - 60] + 30
    img = _render(xy, np.full(25, 1e5), sigma=2.5, rng=rng)
    psf = PsfConfig(fwhm_method=fwhm_method)
    fwhm = extract_sources(img, DetectionConfig(), 50, psf).fwhm
    assert fwhm == pytest.approx(2.355 * 2.5, rel=0.1)


def test_per_frame_aperture_and_fwhm_rejection(target):
    cfg = Config()
    cfg.photometry.aperture_mode = "per_frame"
    cfg.psf.fwhm_method = "photutils"
    cfg.psf.max_fwhm_factor = 1.5
    cfg.detection.edge_buffer = 30
    cfg.output.frame_plots = 0
    res = TargetPipeline(target / "TargetA", cfg).run()
    assert res.n_frames_used == N_FRAMES - 1
    assert "poor seeing" in res.skipped_frames[f"frame_{BAD_FRAME:03d}.fits"]
    rows = list(csv.DictReader(open(res.output_dir / "frames.csv")))
    radii = {float(r["aperture_radius_px"]) for r in rows if r["aperture_radius_px"]}
    assert len(radii) > 3  # radius follows the changing seeing
    rel = np.load(res.output_dir / "normalized_light_curves.npz")["relative_flux"]
    assert int(np.argmax(np.nanstd(rel, axis=1))) in res.flagged


def test_per_frame_rejects_fixed_radius():
    with pytest.raises(ValueError):
        PhotometryConfig(aperture_mode="per_frame", aperture_radius=6.0)


def test_discover_targets(target):
    cfg = Config()
    assert [p.name for p in discover_targets(target, cfg)] == ["TargetA"]


@pytest.mark.parametrize("method", ["sep", "aperture"])
def test_pipeline_flags_variable(target, method):
    cfg = Config()
    cfg.photometry.method = method
    cfg.camera.gain = 1.5
    cfg.detection.edge_buffer = 30
    cfg.output.frame_plots = 1
    res = TargetPipeline(target / "TargetA", cfg).run()
    assert res.n_frames_used == N_FRAMES
    assert res.n_stars >= 30
    out = res.output_dir
    for name in ("light_curves.npz", "flagged_stars.csv", "star_statistics.csv", "frames.csv",
                 "detected_stars.png", "flagged_star_positions.png"):
        assert (out / name).exists(), name
    # the variable star has by far the largest scatter and must be among the flagged
    rel = np.load(out / "normalized_light_curves.npz")["relative_flux"]
    amp = np.nanstd(rel, axis=1)
    top = int(np.argmax(amp))
    assert amp[top] > 10 * np.median(amp)
    assert top in res.flagged
    assert len(res.flagged) <= 3

    # reuse path reproduces the same flagged set
    again = TargetPipeline(target / "TargetA", cfg).run(reuse=True)
    assert again.flagged == res.flagged


def test_common_region_rectangle():
    shape = (600, 700)
    quads = {0: footprint(shape, 30, np.eye(3))}
    m = np.eye(3)
    m[:2, 2] = [10, -6]
    quads[1] = footprint(shape, 30, m)
    rect, dropped = common_region(quads, 640 * 540.0, Config().registration, protected=0)
    assert dropped == []
    x0, y0, x1, y1 = rect
    assert x0 == pytest.approx(40 - 0.5, abs=3) and x1 == pytest.approx(669.5, abs=3)
    assert y0 == pytest.approx(29.5, abs=3) and y1 == pytest.approx(563.5, abs=3)


def test_largest_rectangle_of_square():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert largest_rectangle(square) == pytest.approx((0, 0, 10, 10))


def test_config_validation(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[camera]\ngain = 0.28\n[variability]\nautocorr_threshold = 4.0\n")
    cfg = load_config(p)
    assert cfg.camera.gain == 0.28 and cfg.variability.autocorr_threshold == 4.0
    p.write_text("[camera]\ngian = 1\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_default_config_file_loads():
    from variabledetection.config import DEFAULT_CONFIG_PATH
    assert load_config(DEFAULT_CONFIG_PATH).data.reduced_subdir == "reduced"
