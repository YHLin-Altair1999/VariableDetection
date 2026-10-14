"""End-to-end processing of one target folder."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import plotting
from .config import Config
from .detection import detect_stars, extract_sources
from .io import find_frames, load_image, read_header, read_time
from .normalize import find_blended, normalize
from .photometry import Photometer
from .platesolve import pixel_to_radec, solve_reference
from .registration import apply_matrix, common_region, footprint, register
from .variability import StarResult, compute_slopes, evaluate_star

log = logging.getLogger(__name__)

LIGHT_CURVES = "light_curves.npz"


@dataclass
class RunResult:
    target: str
    n_frames_used: int
    n_stars: int
    flagged: list[int]
    reasons: dict[int, list[str]]
    output_dir: Path
    slope_mean: float = np.nan
    slope_std: float = np.nan
    skipped_frames: dict[str, str] = field(default_factory=dict)


class TargetPipeline:
    def __init__(self, target_dir: Path, cfg: Config):
        self.target_dir = Path(target_dir)
        self.cfg = cfg
        self.output_dir = self.target_dir / cfg.data.output_subdir
        self.skipped: dict[str, str] = {}

    # ── frames ────────────────────────────────────────────────────

    def _list_frames(self) -> list[tuple[Path, float]]:
        frames = []
        for path in find_frames(self.target_dir, self.cfg.data):
            try:
                frames.append((path, read_time(read_header(path), self.cfg.data)))
            except Exception as exc:
                self.skipped[path.name] = f"unreadable header/time: {exc}"
        frames.sort(key=lambda f: f[1])
        if not frames:
            raise RuntimeError(f"No usable FITS frames in {self.target_dir / self.cfg.data.reduced_subdir}")
        return frames

    # ── registration, region, photometry ──────────────────────────

    def _choose_reference(self, n_sources: list[int], candidates: list[int]) -> int:
        setting = self.cfg.registration.reference_frame
        if setting != "auto":
            if setting not in candidates:
                raise ValueError(f"registration.reference_frame={setting} is out of range or was rejected")
            return setting
        best = max(n_sources[i] for i in candidates)
        good = [i for i in candidates if n_sources[i] >= 0.9 * best]
        return min(good, key=lambda i: abs(i - len(n_sources) // 2))

    def _extract_light_curves(self, frames):
        cfg = self.cfg
        out = self.output_dir
        p = cfg.photometry
        sources, shapes = [], []
        for path, _ in frames:
            data = load_image(path)
            sources.append(extract_sources(data, cfg.detection, cfg.registration.max_control_points, cfg.psf))
            shapes.append(data.shape)

        # per-frame PSF width; poor-seeing frames can be rejected
        measured = np.array([s.fwhm for s in sources])
        if np.isfinite(measured).any():
            median_fwhm = float(np.nanmedian(measured))
        else:
            median_fwhm = 3.0
            log.warning("Could not measure the FWHM in any frame; using %.1f px", median_fwhm)
        nominal_fwhm = cfg.detection.fwhm or median_fwhm
        frame_fwhm = np.where(np.isfinite(measured), measured, nominal_fwhm)
        candidates = list(range(len(frames)))
        if cfg.psf.max_fwhm_factor is not None:
            for i in list(candidates):
                if frame_fwhm[i] > cfg.psf.max_fwhm_factor * median_fwhm:
                    self.skipped[frames[i][0].name] = (f"poor seeing: FWHM {frame_fwhm[i]:.2f}px "
                                                       f"> {cfg.psf.max_fwhm_factor} x median {median_fwhm:.2f}px")
                    candidates.remove(i)
        if not candidates:
            raise RuntimeError("All frames were rejected by the FWHM cut")

        def radius_of(i):
            if p.aperture_mode == "per_frame":
                return float(np.clip(p.aperture_fwhm_factor * frame_fwhm[i], p.aperture_min, p.aperture_max))
            return nominal_radius

        nominal_radius = p.aperture_radius or float(
            np.clip(p.aperture_fwhm_factor * nominal_fwhm, p.aperture_min, p.aperture_max))
        photometer = Photometer(p, cfg.camera.gain)
        log.info("FWHM median %.2f px (frames %.2f-%.2f); aperture mode %s, nominal radius %.1f px",
                 median_fwhm, np.min(frame_fwhm[candidates]), np.max(frame_fwhm[candidates]),
                 p.aperture_mode, nominal_radius)

        ref = self._choose_reference([s.n_detected for s in sources], candidates)
        ref_path = frames[ref][0]
        log.info("Reference frame: #%d (%s, %d sources)", ref, ref_path.name, sources[ref].n_detected)
        ref_data = load_image(ref_path)

        # registration to the reference frame
        matrices: dict[int, np.ndarray] = {ref: np.eye(3)}
        n_matched: dict[int, int] = {ref: len(sources[ref].xy)}
        for i in candidates:
            if i == ref:
                continue
            path = frames[i][0]
            if shapes[i] != shapes[ref]:
                self.skipped[path.name] = f"image shape {shapes[i]} differs from reference {shapes[ref]}"
                continue
            matrix, n, reason = register(sources[i].xy, sources[ref].xy, cfg.registration)
            n_matched[i] = n
            if matrix is None:
                self.skipped[path.name] = f"registration failed: {reason}"
            else:
                matrices[i] = matrix
        log.info("Registered %d of %d frames", len(matrices), len(frames))

        # common region seen by every kept frame
        max_radius = max(radius_of(i) for i in matrices)
        margin = max(cfg.detection.edge_buffer, photometer.margin(max_radius) + 1)
        quads = {i: footprint(shapes[ref], margin, m) for i, m in matrices.items()}
        full_area = float(np.prod(np.ptp(quads[ref], axis=0)))
        region, dropped = common_region(quads, full_area, cfg.registration, protected=ref)
        for i in dropped:
            del matrices[i]
            self.skipped[frames[i][0].name] = "dropped to keep a larger common region"
        if region is None:
            raise RuntimeError("Registered frames share no common area")
        frac = (region[2] - region[0]) * (region[3] - region[1]) / full_area
        log.info("Common region x[%.0f, %.0f] y[%.0f, %.0f] = %.0f%% of the reference frame "
                 "(%d frames dropped for it)", region[0], region[2], region[1], region[3], 100 * frac, len(dropped))

        positions, peaks = detect_stars(ref_data, cfg.detection, frame_fwhm[ref], region)
        if len(positions) < 3:
            raise RuntimeError(f"Only {len(positions)} stars detected inside the common region")
        log.info("%d stars inside the common region", len(positions))
        plotting.plot_detected_stars(out / "detected_stars.png", ref_data, positions, region,
                                     f"Detected stars (reference frame {ref}); yellow = common region",
                                     cfg.output.dpi)

        used = sorted(matrices)
        plot_at = set(np.linspace(0, len(used) - 1, min(cfg.output.frame_plots, len(used))).astype(int)) \
            if cfg.output.frame_plots > 0 else set()
        flux = np.full((len(positions), len(used)), np.nan)
        err = np.full_like(flux, np.nan)
        for k, i in enumerate(used):
            data = ref_data if i == ref else load_image(frames[i][0])
            pos = apply_matrix(np.linalg.inv(matrices[i]), positions)
            r = radius_of(i)
            flux[:, k], err[:, k] = photometer.measure(data, pos, r, frame_fwhm[i])
            if k in plot_at:
                plotting.plot_frame_apertures(out / f"frame_{i:04d}_apertures.png", data, pos,
                                              photometer.radii(r),
                                              f"Frame {i} ({frames[i][0].name}), FWHM {frame_fwhm[i]:.2f}px",
                                              cfg.output.dpi)
            if k % max(1, len(used) // 10) == 0:
                log.info("  photometry %d/%d", k + 1, len(used))

        radii = {i: radius_of(i) for i in used}
        self._write_frame_table(frames, matrices, n_matched, ref, measured, radii)
        time = np.array([frames[i][1] for i in used])
        np.savez(out / LIGHT_CURVES, time=time, flux=flux, error=err, positions=positions, peaks=peaks,
                 ref_file=ref_path.name, region=np.array(region), aperture_radius=nominal_radius,
                 fwhm=frame_fwhm[used], frame_radius=np.array([radii[i] for i in used]),
                 files=np.array([frames[i][0].name for i in used]))
        return dict(time=time, flux=flux, error=err, positions=positions, peaks=peaks, ref_file=ref_path.name,
                    region=region, aperture_radius=nominal_radius, ref_data=ref_data)

    def _write_frame_table(self, frames, matrices, n_matched, ref, fwhm, radii):
        with open(self.output_dir / "frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "jd", "status", "dx", "dy", "rotation_deg", "scale", "n_matched",
                        "fwhm_px", "aperture_radius_px"])
            for i, (path, t) in enumerate(frames):
                m = matrices.get(i)
                fw = f"{fwhm[i]:.3f}" if np.isfinite(fwhm[i]) else ""
                if m is None:
                    w.writerow([path.name, f"{t:.6f}", self.skipped.get(path.name, "not used"), "", "", "", "",
                                n_matched.get(i, ""), fw, ""])
                else:
                    rot = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
                    scale = np.sqrt(abs(np.linalg.det(m[:2, :2])))
                    w.writerow([path.name, f"{t:.6f}", "reference" if i == ref else "used",
                                f"{m[0, 2]:.3f}", f"{m[1, 2]:.3f}", f"{rot:.4f}", f"{scale:.5f}", n_matched[i],
                                fw, f"{radii[i]:.2f}"])

    def _load_light_curves(self, frames):
        with np.load(self.output_dir / LIGHT_CURVES) as d:
            ref_file = str(d["ref_file"])
            ref_path = next(p for p, _ in frames if p.name == ref_file)
            return dict(time=d["time"], flux=d["flux"], error=d["error"], positions=d["positions"],
                        peaks=d["peaks"], ref_file=ref_file, region=tuple(d["region"]),
                        aperture_radius=float(d["aperture_radius"]), ref_data=load_image(ref_path))

    # ── analysis ──────────────────────────────────────────────────

    def _analyse(self, lc) -> RunResult:
        cfg, out = self.cfg, self.output_dir
        time, flux, err, positions = lc["time"], lc["flux"], lc["error"], lc["positions"]
        n_stars = len(positions)
        blended = find_blended(positions, cfg.normalization.blend_radius_factor * lc["aperture_radius"])
        log.info("%d blended stars excluded from detection and comparison ensemble", len(blended))

        exclude: set[int] = set()
        for it in range(cfg.normalization.max_ensemble_iterations):
            rel, rel_err = normalize(flux, err, positions, blended, cfg.normalization, exclude)
            slopes, s_mean, s_std = compute_slopes(time, rel, blended, cfg.variability.sigma_clip)
            results = [evaluate_star(time, rel[i], i in blended, s_mean, s_std, cfg.variability)
                       for i in range(n_stars)]
            flagged_set = {i for i, r in enumerate(results) if r.flagged}
            log.info("Ensemble iteration %d: %d candidate variable stars", it, len(flagged_set))
            if flagged_set == exclude:
                break
            exclude = flagged_set

        flagged = sorted(flagged_set)
        if cfg.variability.plot_flagged:
            (out / "linear_fits").mkdir(exist_ok=True)
            for i in flagged:
                plotting.plot_star_fit(out / "linear_fits" / f"curve_{i:03d}.png", i, results[i],
                                       cfg.variability.autocorr_max_lag, 150)
        plotting.plot_flagged_map(out / "flagged_star_positions.png", lc["ref_data"], positions, flagged,
                                  blended, lc["aperture_radius"], lc["region"], cfg.output.dpi)
        if flagged:
            plotting.plot_overview(out / "normalized_light_curves.png", time, rel, flagged, cfg.output.dpi)
        np.savez(out / "normalized_light_curves.npz", time=time, relative_flux=rel, relative_error=rel_err)

        radec = self._plate_solve(lc, flagged, n_stars)
        self._write_stars_csv(out / "star_statistics.csv", range(n_stars), lc, results, blended, radec)
        self._write_stars_csv(out / "flagged_stars.csv", flagged, lc, results, blended, radec)
        return RunResult(self.target_dir.name, len(time), n_stars, flagged,
                         {i: results[i].reasons for i in flagged}, out, s_mean, s_std, dict(self.skipped))

    def _plate_solve(self, lc, flagged, n_stars) -> Optional[np.ndarray]:
        if not self.cfg.platesolve.enabled or not flagged:
            return None
        wcs = solve_reference(lc["positions"], lc["peaks"], lc["ref_data"].shape, self.cfg.platesolve)
        if wcs is None:
            return None
        wcs.to_header().tofile(str(self.output_dir / "reference_wcs.hdr"), overwrite=True)
        return pixel_to_radec(wcs, lc["positions"])

    @staticmethod
    def _write_stars_csv(path, ids, lc, results: list[StarResult], blended, radec):
        header = ["star_id", "x", "y", "peak", "blended", "flagged", "reasons", "slope", "f_stat",
                  "autocorr_z", "autocorr_lag"] + (["ra_deg", "dec_deg"] if radec is not None else [])
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in ids:
                r = results[i]
                row = [i, f"{lc['positions'][i, 0]:.2f}", f"{lc['positions'][i, 1]:.2f}", f"{lc['peaks'][i]:.1f}",
                       int(i in blended), int(r.flagged), "+".join(r.reasons), f"{r.slope:.6e}",
                       f"{r.f_stat:.2f}", f"{r.autocorr_z:.2f}", r.autocorr_lag]
                if radec is not None:
                    row += [f"{radec[i, 0]:.6f}", f"{radec[i, 1]:.6f}"]
                w.writerow(row)

    # ── entry point ───────────────────────────────────────────────

    def run(self, reuse: bool = False) -> RunResult:
        frames = self._list_frames()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lc = self._load_light_curves(frames) if reuse else self._extract_light_curves(frames)
        return self._analyse(lc)
