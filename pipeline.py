"""
Full-frame photometry pipeline: star detection, light curve extraction,
differential photometry, and nonlinearity detection in a single class.
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from glob import glob
import os
import pickle
import warnings
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.psf import PSFPhotometry, IntegratedGaussianPRF
from astropy.table import Table
from astropy.modeling.fitting import TRFLSQFitter
from astropy.stats import sigma_clipped_stats
from auto_stretch import apply_stretch
import sep
from scipy.spatial import cKDTree
try:
    from numpy.exceptions import RankWarning
except ImportError:  # NumPy < 1.25
    RankWarning = np.RankWarning

warnings.filterwarnings('ignore', category=RankWarning)


class PhotometryPipeline:
    """End-to-end pipeline: detect stars, extract light curves, normalize, detect nonlinearity."""

    # ZWO ASI6200MM Pro at gain setting 100 (e-/ADU)
    CAMERA_GAIN = 0.28
    # HOPS stores the PSF as a Gaussian-sigma-like width in (binned) pixels; calibrated against
    # measured half-flux diameters, FWHM ≈ 2.355 * PSF.
    PSF_TO_FWHM = 2.355
    APERTURE_FWHM_FACTOR = 2.0

    def __init__(self, working_dir='./', output_dir='./output_light_curves', f_threshold=50.0, slope_std_threshold=5.0, comparison_star_distance_threshold=300, gain=CAMERA_GAIN, autocorr_threshold=5.0, autocorr_max_lag=5, blend_radius_factor=2.0):
        fits_path = os.path.join(working_dir, 'REDUCED_DATA', '*.fits')
        fnames = sorted(glob(fits_path))
        headers = {f: self._read_header(f) for f in fnames}
        # Drop frames rejected during the HOPS visual inspection (SKIP = True)
        self.skipped_fnames = [f for f in fnames if headers[f].get('SKIP') in (True, 'T', 'True')]
        fnames = [f for f in fnames if f not in set(self.skipped_fnames)]
        # Order frames by observation time, not by filename
        jds = [headers[f].get('HOPSJD', i) for i, f in enumerate(fnames)]
        self.fnames = [fnames[i] for i in np.argsort(jds, kind='stable')]
        self._headers = headers
        self.gain = gain
        self.output_dir = output_dir
        self.f_threshold = f_threshold
        self.slope_std_threshold = slope_std_threshold  # sigma threshold for slope outliers
        self.autocorr_threshold = autocorr_threshold  # z-score threshold for residual autocorrelation
        self.autocorr_max_lag = autocorr_max_lag
        self.blend_radius_factor = blend_radius_factor  # blended if a neighbour is within factor * aperture radius
        self.blended_sids = set()
        self.flag_reasons = {}  # star index -> list of triggered criteria
        self.comparison_star_distance_threshold = comparison_star_distance_threshold
        self._estimate_psf()
        os.makedirs(output_dir, exist_ok=True)
        print(f"Found {len(self.fnames)} FITS files ({len(self.skipped_fnames)} skipped by HOPS) | Output: {output_dir}")
        print(f"HOPS PSF: sigma={self.psf_sigma:.2f}px, FWHM={self.fwhm:.2f}px -> aperture radius {self.default_aperture_radius:.1f}px")

    def _estimate_psf(self):
        """Derive PSF width, FWHM and default aperture radius from the HOPS 'PSF' header values.

        HOPS writes 10 when a frame's PSF fit fails, so the median over frames (ignoring
        non-positive values) is used. Falls back to FWHM = 3 px if no PSF info exists.
        """
        psfs = np.array([h.get('PSF', np.nan) for h in self._headers.values()], float)
        psfs = psfs[np.isfinite(psfs) & (psfs > 0)]
        if len(psfs):
            self.psf_sigma = float(np.median(psfs))
            self.fwhm = max(self.PSF_TO_FWHM * self.psf_sigma, 1.5)
        else:
            self.fwhm = 3.0
            self.psf_sigma = self.fwhm / self.PSF_TO_FWHM
        self.default_aperture_radius = float(np.clip(self.APERTURE_FWHM_FACTOR * self.fwhm, 3.0, 20.0))

    # ── FITS helpers ──────────────────────────────────────────────

    def _load_fits(self, fname):
        """Return (image, header) from first HDU with data."""
        with fits.open(fname) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    return hdu.data.copy(), hdu.header.copy()
        return None, None

    @staticmethod
    def _read_header(fname):
        """Header only (no pixel decompression) of the HDU holding the image keywords."""
        with fits.open(fname) as hdul:
            for hdu in hdul:
                if 'HOPSJD' in hdu.header:
                    return hdu.header.copy()
            return hdul[-1].header.copy()

    @staticmethod
    def _align_values(header):
        """(x0, y0, u0) from HOPS alignment keywords, or None if missing/failed (all zero)."""
        vals = [header.get(k) for k in ('ALIGN_X0', 'ALIGN_Y0', 'ALIGN_U0')]
        if any(v is None or not np.isfinite(v) for v in vals) or not any(vals):
            return None
        return tuple(float(v) for v in vals)

    def _transform_array(self, pts, ax0, ay0, au0):
        """Predict star positions (n, 2) in a frame from its HOPS alignment (rotation + translation)."""
        dx, dy = pts[:, 0] - self._ref_x0, pts[:, 1] - self._ref_y0
        angle = au0 - self._ref_u0
        c, s = np.cos(angle), np.sin(angle)
        return np.column_stack([c * dx - s * dy + ax0, s * dx + c * dy + ay0])

    def _choose_ref_frame(self, ref_frame):
        """Keep the requested reference frame unless its alignment is missing or an outlier.

        Among frames sharing the dominant orientation (i.e. one side of a meridian
        flip), a frame is an outlier if its (x0, y0, u0) is far from the group median;
        in that case the frame closest to the median is used instead.
        """
        al = [self._align_values(self._headers[f]) for f in self.fnames]
        valid = [i for i, a in enumerate(al) if a is not None]
        if not valid:
            return ref_frame
        arr = np.array([al[i] for i in valid])
        dang = np.angle(np.exp(1j * (arr[:, 2] - arr[0, 2])))
        group = np.round(dang).astype(int)
        major = np.bincount(group - group.min()).argmax() + group.min()
        sel = group == major
        med = np.median(arr[sel], axis=0)
        ddang = np.angle(np.exp(1j * (arr[:, 2] - med[2])))
        dist = np.hypot(arr[:, 0] - med[0], arr[:, 1] - med[1])
        good = sel & (dist < 10) & (np.abs(ddang) < 0.02)
        if ref_frame in valid and good[valid.index(ref_frame)]:
            return ref_frame
        best = valid[int(np.argmin(np.where(good, dist, np.inf)))] if good.any() else valid[int(np.argmin(dist))]
        print(f"Reference frame {ref_frame} has a missing/outlying alignment; using frame {best} instead")
        return best

    # ── Star detection ────────────────────────────────────────────

    def detect_stars(self, ref_frame=20, buff_x=50, buff_y=50):
        """Detect stars in a reference frame using DAOStarFinder.

        Parameters
        ----------
        ref_frame : int
            Index of the reference frame to use for detection.
        buff_x : int
            Buffer in pixels from the left/right edges to exclude stars.
        buff_y : int
            Buffer in pixels from the top/bottom edges to exclude stars.
        """
        ref_frame = self._choose_ref_frame(ref_frame)
        self.ref_frame = ref_frame
        image, header = self._load_fits(self.fnames[ref_frame])
        self._ref_x0 = header.get('ALIGN_X0', 0)
        self._ref_y0 = header.get('ALIGN_Y0', 0)
        self._ref_u0 = header.get('ALIGN_U0', 0)

        ny, nx = image.shape
        bg_median, _, bg_std = sigma_clipped_stats(image, sigma=3.0)
        sources = DAOStarFinder(fwhm=max(self.fwhm, 2.0), threshold=5 * bg_std)(image - bg_median)

        # Filter out stars too close to the frame edges
        x = sources['xcentroid']
        y = sources['ycentroid']
        edge_mask = (x >= buff_x) & (x <= nx - buff_x) & (y >= buff_y) & (y <= ny - buff_y)
        sources = sources[edge_mask]

        self.star_positions = list(zip(sources['xcentroid'], sources['ycentroid']))
        self.star_peaks = np.array(sources['peak'])
        print(f"Detected {len(self.star_positions)} stars in frame {ref_frame}")

        # Save annotated image
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(apply_stretch(image), cmap='gray', origin='lower')
        for sid, (x, y) in enumerate(self.star_positions):
            ax.add_patch(plt.Circle((x, y), 10, color='red', fill=False, lw=2))
            ax.text(x, y + 15, str(sid), color='red', fontsize=8, ha='center')
        ax.set_title(f'Detected Stars (Frame {ref_frame})')
        fig.savefig(os.path.join(self.output_dir, 'detected_stars.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        return self.star_positions

    # ── Light curve extraction ────────────────────────────────────

    def extract_light_curves(self, ref_frame=20, aperture_radius=None, method='sep',
                              psf_sigma=None, fit_shape=11):
        """Extract light curves for all detected stars.

        Parameters
        ----------
        ref_frame : int
            Reference frame index for star detection.
        aperture_radius : float or None
            Aperture radius (used by 'aperture'/'sep' methods and frame plots).
            None: 2 x FWHM derived from the HOPS 'PSF' header.
        method : str
            Photometry method: 'sep' (SEP aperture photometry, default),
            'aperture' (photutils classical aperture), or 'psf' (PSF fitting).
        psf_sigma : float or None
            Gaussian PSF sigma in pixels (used by 'psf' method). None: HOPS value.
        fit_shape : int
            Size of the fitting region in pixels (used by 'psf' method).
        """
        if method not in ('aperture', 'psf', 'sep'):
            raise ValueError(f"Unknown method '{method}'. Use 'sep', 'aperture' or 'psf'.")
        if aperture_radius is None:
            aperture_radius = self.default_aperture_radius
        if psf_sigma is None:
            psf_sigma = max(self.psf_sigma, 0.5)

        if not hasattr(self, 'star_positions'):
            self.detect_stars(ref_frame)

        n_stars = len(self.star_positions)
        n_frames = len(self.fnames)
        lcs = {sid: {'time': [], 'flux': [], 'error': [], 'position': pos}
               for sid, pos in enumerate(self.star_positions)}

        # Set up PSF photometry objects once if needed
        if method == 'psf':
            psf_model = IntegratedGaussianPRF(sigma=psf_sigma)
            fitter = TRFLSQFitter()
            psfphot = PSFPhotometry(psf_model, fit_shape=(fit_shape, fit_shape),
                                    fitter=fitter, aperture_radius=aperture_radius)
            print(f"Using PSF photometry (sigma={psf_sigma}, fit_shape={fit_shape})")
        elif method == 'sep':
            print(f"Using SEP aperture photometry (r={aperture_radius})")
        else:
            print(f"Using aperture photometry (r={aperture_radius})")

        n_all = len(self.star_positions)
        for fi in range(n_frames):
            if fi % max(1, n_frames // 10) == 0:
                print(f"  Frame {fi}/{n_frames}")
            image, header = self._load_fits(self.fnames[fi])
            if image is None:
                header = self._read_header(self.fnames[fi])
                self._append_nan(lcs, range(n_all), header.get('HOPSJD', fi))
                continue
            hopsjd = header.get('HOPSJD', fi)

            pos = self._solve_positions(image, header)
            if pos is None:
                print(f"  Frame {fi}: alignment failed, frame set to NaN")
                self._append_nan(lcs, range(n_all), hopsjd)
                continue

            # Visualize a few frames
            if fi % max(1, n_frames // 5) == 0:
                self._plot_frame(image, fi, aperture_radius, pos, hopsjd)

            if method == 'psf':
                self._extract_psf_frame(image, lcs, pos, hopsjd, psfphot)
            elif method == 'sep':
                self._extract_sep_frame(image, lcs, pos, hopsjd, aperture_radius)
            else:
                self._extract_aperture_frame(image, lcs, pos, hopsjd, aperture_radius)

        # Convert to arrays
        for sid in lcs:
            for k in ('time', 'flux', 'error'):
                lcs[sid][k] = np.array(lcs[sid][k])

        self.light_curves = lcs
        print(f"Extracted light curves for {n_stars} stars across {n_frames} frames")
        return lcs

    @staticmethod
    def _append_nan(lcs, sids, hopsjd):
        for sid in sids:
            lcs[sid]['time'].append(hopsjd)
            lcs[sid]['flux'].append(np.nan)
            lcs[sid]['error'].append(np.nan)

    def _solve_positions(self, image, header, max_shift=6.0, vote_range=60.0, min_good=5):
        """Star positions (n, 2) in this frame, or None if alignment cannot be established.

        The HOPS header alignment only provides the initial guess (it is a single-star
        pivot, so small angle errors and meridian flips displace distant stars by many
        pixels). It is then corrected by (1) a translation vote between detected sources
        and predicted positions, and (2) a fit of a rigid transform (rotation + shift) to
        windowed centroids of the bright reference stars, iterated with outlier rejection.
        """
        align = self._align_values(header)
        if align is None:
            return None
        ref = np.asarray(self.star_positions, float)
        pred = self._transform_array(ref, *align)

        data = np.ascontiguousarray(image, dtype=np.float64)
        bkg = sep.Background(data)
        data_sub = data - bkg.back()
        ny, nx = data.shape

        try:
            objs = sep.extract(data_sub, 5.0 * bkg.globalrms, minarea=5)
        except Exception:
            objs = []
        if len(objs) >= 3:
            objs = objs[np.argsort(objs['flux'])[::-1][:80]]
            det = np.column_stack([objs['x'], objs['y']])
            d = (det[None, :, :] - pred[:, None, :]).reshape(-1, 2)
            d = d[(np.abs(d) < vote_range).all(axis=1)]
            if len(d) >= 3:
                edges = np.arange(-vote_range, vote_range + 2, 2.0)
                h, xe, ye = np.histogram2d(d[:, 0], d[:, 1], bins=[edges, edges])
                ix, iy = np.unravel_index(np.argmax(h), h.shape)
                if h[ix, iy] >= 3:
                    c = np.array([(xe[ix] + xe[ix + 1]) / 2, (ye[iy] + ye[iy + 1]) / 2])
                    near = d[np.hypot(*(d - c).T) < 3.0]
                    pred = pred + near.mean(axis=0)

        bright = self.star_peaks >= np.median(self.star_peaks)
        for _ in range(3):
            inb = (pred[:, 0] > 5) & (pred[:, 0] < nx - 5) & (pred[:, 1] > 5) & (pred[:, 1] < ny - 5)
            use = bright & inb
            if use.sum() < min_good:
                return None
            xw, yw, fl = sep.winpos(data_sub, pred[use, 0], pred[use, 1], float(np.clip(self.psf_sigma, 1.0, 3.0)))
            meas = np.column_stack([xw, yw])
            good = (fl == 0) & np.isfinite(meas).all(axis=1) & (np.hypot(*(meas - pred[use]).T) < max_shift)
            if good.sum() < min_good:
                return None
            z = ref[use][good, 0] + 1j * ref[use][good, 1]
            w = meas[good, 0] + 1j * meas[good, 1]
            for _ in range(2):
                zc, wc = z - z.mean(), w - w.mean()
                a = np.sum(np.conj(zc) * wc) / np.sum(np.abs(zc) ** 2)
                a /= abs(a)  # pure rotation, no scaling
                b = w.mean() - a * z.mean()
                keep = np.abs(a * z + b - w) < 3.0
                if keep.all() or keep.sum() < min_good:
                    break
                z, w = z[keep], w[keep]
            allz = ref[:, 0] + 1j * ref[:, 1]
            fitted = a * allz + b
            pred = np.column_stack([fitted.real, fitted.imag])
        return pred

    def _extract_aperture_frame(self, image, lcs, pos, hopsjd, aperture_radius):
        """Aperture photometry for a single frame."""
        for sid, (x, y) in enumerate(pos):
            ap = CircularAperture((x, y), aperture_radius)
            an = CircularAnnulus((x, y), aperture_radius + 2, aperture_radius + 8)
            flux = aperture_photometry(image, ap)['aperture_sum'][0]
            bg = aperture_photometry(image, an)['aperture_sum'][0] / an.area
            flux_sub = flux - bg * ap.area
            lcs[sid]['time'].append(hopsjd)
            lcs[sid]['flux'].append(flux_sub)
            lcs[sid]['error'].append(np.sqrt(flux_sub / self.gain) if flux_sub > 0 else 0)

    def _extract_sep_frame(self, image, lcs, pos, hopsjd, aperture_radius):
        """Aperture photometry for a single frame using SEP (Source Extractor).

        SEP's spatially-varying background map is subtracted first. The residual local
        sky is the sigma-clipped *median* of a surrounding annulus: a mean (as in
        sep's ``bkgann``) is biased by neighbouring stars or hot pixels falling in the
        annulus, which produces step offsets in a star's light curve whenever such a
        feature enters or leaves the annulus.
        The error combines background rms, Poisson noise (camera gain in e-/ADU,
        self.gain) and the uncertainty of the local sky level.
        """
        data = np.ascontiguousarray(image, dtype=np.float64)
        bkg = sep.Background(data)
        data_sub = data - bkg.back()

        ny, nx = data.shape
        r_in, r_out = aperture_radius + 2, aperture_radius + 8
        margin = r_out + 1  # whole annulus (around the rounded centre) must be on-frame

        inb = (pos[:, 0] >= margin) & (pos[:, 0] < nx - margin) & (pos[:, 1] >= margin) & (pos[:, 1] < ny - margin)
        valid_sids = np.flatnonzero(inb)

        # Mark stars too close to the edge as NaN for this frame
        self._append_nan(lcs, np.flatnonzero(~inb), hopsjd)

        if len(valid_sids) == 0:
            return

        px, py = pos[valid_sids, 0], pos[valid_sids, 1]
        flux, fluxerr, flag = sep.sum_circle(
            data_sub, px, py, aperture_radius, err=bkg.globalrms, gain=self.gain)

        # Sigma-clipped median sky in the annulus, vectorized over all stars
        k = int(np.ceil(r_out))
        dy, dx = np.mgrid[-k:k + 1, -k:k + 1]
        ring = (np.hypot(dx, dy) >= r_in) & (np.hypot(dx, dy) <= r_out)
        dx, dy = dx[ring], dy[ring]
        vals = data_sub[np.rint(py).astype(int)[:, None] + dy[None, :],
                        np.rint(px).astype(int)[:, None] + dx[None, :]]
        med = np.median(vals, axis=1, keepdims=True)
        sig = 1.4826 * np.median(np.abs(vals - med), axis=1, keepdims=True)
        vals = np.where(np.abs(vals - med) <= 3.0 * np.maximum(sig, 1e-9), vals, np.nan)
        sky = np.nanmedian(vals, axis=1)
        n_ann = np.sum(np.isfinite(vals), axis=1)

        ap_area = np.pi * aperture_radius ** 2
        flux = flux - sky * ap_area
        fluxerr = np.sqrt(fluxerr ** 2 + ap_area ** 2 * bkg.globalrms ** 2 / np.maximum(n_ann, 1))

        for i, sid in enumerate(valid_sids):
            lcs[sid]['time'].append(hopsjd)
            if flag[i] != 0 or n_ann[i] < 20:
                lcs[sid]['flux'].append(np.nan)
                lcs[sid]['error'].append(np.nan)
            else:
                lcs[sid]['flux'].append(flux[i])
                lcs[sid]['error'].append(fluxerr[i])
    def _extract_psf_frame(self, image, lcs, pos, hopsjd, psfphot):
        """PSF photometry for a single frame using photutils PSFPhotometry."""
        ny, nx = image.shape
        half = psfphot.fit_shape[0] // 2  # margin needed from image edge

        # Build initial-guess table, filtering out stars too close to edges
        inb = (pos[:, 0] >= half) & (pos[:, 0] < nx - half) & (pos[:, 1] >= half) & (pos[:, 1] < ny - half)
        valid_sids = list(np.flatnonzero(inb))
        xs, ys = pos[inb, 0], pos[inb, 1]

        # Mark edge stars as NaN for this frame
        self._append_nan(lcs, np.flatnonzero(~inb), hopsjd)

        if not valid_sids:
            return

        init_params = Table()
        init_params['x_0'] = xs
        init_params['y_0'] = ys

        # Subtract background estimate
        _, bg_median, _ = sigma_clipped_stats(image, sigma=3.0)
        image_sub = image# - bg_median

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = psfphot(image_sub, init_params=init_params)
            for i, sid in enumerate(valid_sids):
                flux = result['flux_fit'][i]
                flux_err = result['flux_err'][i] if 'flux_err' in result.colnames else 0.0
                # Flag non-converged fits (flags != 0) as NaN
                flag = result['flags'][i] if 'flags' in result.colnames else 0
                if flag != 0:
                    lcs[sid]['time'].append(hopsjd)
                    lcs[sid]['flux'].append(np.nan)
                    lcs[sid]['error'].append(np.nan)
                else:
                    lcs[sid]['time'].append(hopsjd)
                    lcs[sid]['flux'].append(float(flux))
                    lcs[sid]['error'].append(float(flux_err) if flux_err > 0
                                             else (np.sqrt(flux / self.gain) if flux > 0 else 0))
        except Exception as e:
            print(f"  PSF photometry failed for frame (HOPSJD={hopsjd:.6f}): {e}")
            for sid in valid_sids:
                lcs[sid]['time'].append(hopsjd)
                lcs[sid]['flux'].append(np.nan)
                lcs[sid]['error'].append(np.nan)

    def _plot_frame(self, image, fi, r, pos, hopsjd):
        ny, nx = image.shape
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(apply_stretch(image), cmap='gray', origin='lower')
        for sid, (x, y) in enumerate(pos):
            if not (0 <= x < nx and 0 <= y < ny):
                continue  # star is outside this frame
            ax.add_patch(plt.Circle((x, y), r, color='red', fill=False, lw=1.5))
            ax.add_patch(plt.Circle((x, y), r + 2, color='cyan', fill=False, lw=1, ls='--'))
            ax.add_patch(plt.Circle((x, y), r + 8, color='cyan', fill=False, lw=1, ls='--'))
            ax.text(x, y + r + 10, str(sid), color='red', fontsize=7, ha='center', clip_on=True)
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)
        ax.set_title(f'Frame {fi} (HOPSJD: {hopsjd:.6f})')
        fig.savefig(os.path.join(self.output_dir, f'frame_{fi:04d}_apertures.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ── Save / load ───────────────────────────────────────────────

    def save_light_curves(self):
        """Save light curves as pickle and human-readable text."""
        with open(os.path.join(self.output_dir, 'light_curves.pickle'), 'wb') as f:
            pickle.dump(self.light_curves, f)
        with open(os.path.join(self.output_dir, 'light_curves.txt'), 'w') as f:
            for sid, d in self.light_curves.items():
                f.write(f"\n=== Star {sid} at {d['position']} ===\nTime\tFlux\tError\n")
                for t, fl, er in zip(d['time'], d['flux'], d['error']):
                    f.write(f"{t}\t{fl:.2f}\t{er:.2f}\n")
        print(f"Light curves saved to {self.output_dir}")

    def load_light_curves(self, path=None):
        """Load previously saved light curves from pickle."""
        path = path or os.path.join(self.output_dir, 'light_curves.pickle')
        with open(path, 'rb') as f:
            self.light_curves = pickle.load(f)
        print(f"Loaded {len(self.light_curves)} light curves from {path}")

    # ── Robust fitting helper ──────────────────────────────────────

    def _sigma_clip_polyfit(self, t, y, deg, sigma=3.0, max_iters=5):
        """Iterative kappa-sigma clipped polynomial fit.

        Repeatedly fits a degree-``deg`` polynomial, removes points whose
        residuals exceed ``sigma`` standard deviations, and refits until the
        inlier set converges (or ``max_iters`` is reached). Returns the final
        coefficients and the boolean inlier mask (relative to the input arrays).
        """
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

    # ── Differential photometry (normalization) ───────────────────

    @property
    def blend_radius(self):
        return self.blend_radius_factor * self.default_aperture_radius

    def _find_blended(self, positions):
        """Indices of stars that have another star within ``blend_radius`` pixels."""
        pairs = cKDTree(positions).query_pairs(self.blend_radius)
        return set(i for pair in pairs for i in pair)

    def normalize(self, exclude_sids=None):
        """Differential photometry: each star's flux divided by the ensemble of all others.

        Parameters
        ----------
        exclude_sids : set or None
            Star ids to exclude from being used as *comparison* stars (e.g.
            stars flagged as likely variable in a previous pass). Their own
            light curves are still computed normally. If excluding these
            stars would leave a candidate with no valid comparison ensemble,
            they are added back in as a fallback.
        """
        comparison_star_distance_threshold = self.comparison_star_distance_threshold
        exclude_sids = exclude_sids or set()
        lcs_list = [self.light_curves[k] for k in sorted(self.light_curves.keys())]
        self.time = lcs_list[0]['time']
        flux = np.array([lc['flux'] for lc in lcs_list])
        err = np.array([lc['error'] for lc in lcs_list])
        positions = np.array([lc['position'] for lc in lcs_list])  # (n_stars, 2)

        # Stars with a neighbour closer than blend_radius have contaminated photometry
        self.blended_sids = self._find_blended(positions)
        blended = np.zeros(flux.shape[0], bool)
        blended[list(self.blended_sids)] = True
        print(f"{len(self.blended_sids)} blended stars (neighbour within {self.blend_radius:.1f}px) excluded from detection and comparison ensemble")

        rel, rel_err = [], []
        median_flux = np.nanmedian(flux, axis=1)  # median flux per star
        for i in range(flux.shape[0]):
            # Select comparison stars with similar brightness (within 0.5 dex), excluding star i
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ratio = np.abs(np.log10(np.abs(median_flux / median_flux[i])))
            mask = (log_ratio < 0.5)
            mask[i] = False  # exclude current star
            mask &= ~blended  # blended stars make poor comparison stars
            
            # Apply distance threshold if specified
            if comparison_star_distance_threshold is not None:
                dist = np.sqrt(np.sum((positions - positions[i]) ** 2, axis=1))
                mask &= (dist <= comparison_star_distance_threshold)

            # Exclude stars flagged as likely variable from the comparison ensemble,
            # unless doing so would leave nothing to compare against
            if exclude_sids:
                candidate_mask = mask.copy()
                for sid in exclude_sids:
                    candidate_mask[sid] = False
                if candidate_mask.any():
                    mask = candidate_mask

            if not mask.any():
                # fallback: use all other stars if none pass the filters
                mask = np.ones(flux.shape[0], bool)
                mask[i] = False

            # normalized each light curve, then take the median of the ensemble at each time point as the comparison flux
            median_of_each_lc = np.nanmedian(flux[mask], axis=1)
            normalized_comparison_lc = flux[mask] / median_of_each_lc[:, np.newaxis]
            #F_E = np.nanmedian(normalized_comparison_lc, axis=0)
            F_E = sigma_clipped_stats(normalized_comparison_lc, sigma=3.0, axis=0)[1]  # robust median
            
            F_T = flux[i]
            comp = flux[mask]
            comp_rel_err = err[mask] / np.abs(comp)  # fractional errors of comparison stars
            n_comp = np.sum(np.isfinite(comp_rel_err), axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                ens_rel_err = np.sqrt(np.nansum(comp_rel_err ** 2, axis=0)) / n_comp

                # Normalize by the median so the curve is ~1; errors add in quadrature
                r = F_T / F_E
                r = r / np.nanmedian(r)
                rel.append(r)
                rel_err.append(np.abs(r) * np.sqrt((err[i] / F_T) ** 2 + ens_rel_err ** 2))

        self.rel_lcs = np.array(rel)
        self.rel_lcs_err = np.array(rel_err)
        print(f"Normalized {len(rel)} light curves")
        return self.time, self.rel_lcs, self.rel_lcs_err

    # ── Slope statistics ──────────────────────────────────────────

    def compute_slopes(self, sigma_clip=3.0):
        """Fit linear trends to all normalized light curves; store population stats.

        For each light curve a robust linear fit is performed using iterative
        kappa-sigma clipping (``_sigma_clip_polyfit``) so that outlier points
        do not bias the slope estimate.
        """
        t_c = self.time - np.median(self.time)
        slopes = []
        for k, lc in enumerate(self.rel_lcs):
            if k in self.blended_sids:
                slopes.append(np.nan)  # keep blended stars out of the slope statistics
                continue
            try:
                m = ~np.isnan(lc) & (lc > 0)
                coeffs, _ = self._sigma_clip_polyfit(t_c[m], lc[m], 1, sigma=sigma_clip)
                slopes.append(coeffs[0])
            except Exception as e:
                print(f"Warning: failed to fit slope for light curve {k}: {e}")
                slopes.append(np.nan)  # keep slopes aligned with star indices
        self.slopes = np.array(slopes)
        self.slope_mean, self.slope_std = np.nanmean(self.slopes), np.nanstd(self.slopes)
        print(f"Slopes: mean={self.slope_mean:.6e}, std={self.slope_std:.6e}")

    # ── Autocorrelation ───────────────────────────────────────────

    def _autocorr_significance(self, resid):
        """Significance of residual autocorrelation at lags 1..autocorr_max_lag.

        `resid` should be detrended (slow trends removed) and time-ordered.
        White noise gives r_k ~ N(0, 1/N), so z_k = r_k * sqrt(N). Only positive
        correlation counts. Real variability persists over several frames, so
        the statistic is the *minimum* z over all lags: the threshold must be
        exceeded at every lag 1..autocorr_max_lag. Returns (min z, lag of min z).
        """
        resid = np.asarray(resid, dtype=float)
        resid = resid[np.isfinite(resid)]
        n = len(resid)
        max_lag = self.autocorr_max_lag
        if n < max(10, 3 * max_lag):
            return 0.0, 0
        resid = resid - resid.mean()
        denom = np.sum(resid ** 2)
        if denom <= 0:
            return 0.0, 0
        zs = [np.sum(resid[:-k] * resid[k:]) / denom * np.sqrt(n) for k in range(1, max_lag + 1)]
        weakest = int(np.argmin(zs))
        return float(zs[weakest]), weakest + 1

    # ── Nonlinearity detection ────────────────────────────────────

    def detect_nonlinear(self, idx, plot=True, sigma_clip=3.0):
        """F-test comparing linear vs quadratic fit for light curve `idx`.
        Also flags slope outliers and significant positive autocorrelation of
        the quadratic-detrended residuals (short-timescale variability).
        Returns True if anomalous.

        Both the linear and quadratic fits use iterative kappa-sigma clipping
        so a handful of bad photometric points can't masquerade as curvature
        (or hide real curvature). The F-test itself is evaluated on the
        intersection of the two inlier sets to keep the comparison fair.
        """
        lc = self.rel_lcs[idx]
        if idx in self.blended_sids:
            self.flag_reasons[idx] = []
            return False
        m = ~np.isnan(lc)
        y, t = lc[m], (self.time - np.median(self.time))[m]

        if len(y) < 10:  # too few valid points for a meaningful fit
            self.flag_reasons[idx] = []
            return False

        lin, lin_inliers = self._sigma_clip_polyfit(t, y, 1, sigma=sigma_clip)
        quad, quad_inliers = self._sigma_clip_polyfit(t, y, 2, sigma=sigma_clip)

        inliers = lin_inliers & quad_inliers
        y_in, t_in = y[inliers], t[inliers]
        lin_fit = np.polyval(lin, t_in);   lin_res = y_in - lin_fit
        quad_fit = np.polyval(quad, t_in); quad_res = y_in - quad_fit

        sigma = max(np.sqrt(np.mean(lin_res**2)),
                     np.median(np.abs(lin_res - np.median(lin_res))) / 0.6745,
                     np.std(y_in) / 10.0)

        chi2_l = np.sum((lin_res / sigma) ** 2)
        chi2_q = np.sum((quad_res / sigma) ** 2)
        dof_l, dof_q = len(y_in) - 2, len(y_in) - 3
        f_stat = ((chi2_l - chi2_q) / (chi2_q / dof_q)) if chi2_q > 0 and dof_q > 0 else 0

        sig_slope = (self.slope_std > 0 and
                     (abs(lin[0] - self.slope_mean) / self.slope_std) > self.slope_std_threshold)
        nonlinear = f_stat > self.f_threshold
        autocorr_z, autocorr_lag = self._autocorr_significance(y - np.polyval(quad, t))
        autocorr = autocorr_z > self.autocorr_threshold
        flagged = sig_slope or nonlinear or autocorr
        self.flag_reasons[idx] = [n for n, f in (('slope', sig_slope), ('nonlinear', nonlinear),
                                                 ('autocorr', autocorr)) if f]

        if plot and flagged:
            chi2r_l = chi2_l / dof_l if dof_l > 0 else np.inf
            chi2r_q = chi2_q / dof_q if dof_q > 0 else np.inf
            os.makedirs(os.path.join(self.output_dir, 'linear_fits'), exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            outliers = ~inliers
            y_lo, y_hi = np.min(y_in - sigma), np.max(y_in + sigma)
            pad = 0.05 * (y_hi - y_lo)
            t_lo, t_hi = t_in.min(), t_in.max()
            tpad = 0.03 * (t_hi - t_lo)
            ax.errorbar(t[inliers], y[inliers], yerr=sigma, fmt='o', alpha=0.6, capsize=3, label='Data (±empirical σ)')
            if outliers.any():
                ax.errorbar(t[outliers], y[outliers], yerr=sigma, fmt='x', color='k',
                            alpha=0.8, capsize=3, label='Clipped outliers')
            t_line = np.linspace(t_lo, t_hi, 100)
            ax.plot(t_line, np.polyval(lin, t_line), 'r-', lw=2.5, label=f'Linear (χ²_red={chi2r_l:.4f})')
            ax.plot(t_line, np.polyval(quad, t_line), 'g--', lw=2.5, label=f'Quadratic (χ²_red={chi2r_q:.4f})')
            flags = []
            if sig_slope: flags.append('Significant Slope')
            if nonlinear: flags.append(f'Nonlinear (F={f_stat:.2f})')
            if autocorr: flags.append(f'Autocorrelation (z>{autocorr_z:.1f} at all lags 1-{self.autocorr_max_lag})')
            ax.set_title(f'Star {idx} [{" + ".join(flags)}]')
            ax.set_xlabel('Time - median (days)'); ax.set_ylabel('Relative Flux')
            ax.set_xlim(t_lo - tpad, t_hi + tpad)  # limits from inliers only
            ax.set_ylim(y_lo - pad, y_hi + pad)
            ax.legend(); ax.grid(alpha=0.3)
            fig.savefig(os.path.join(self.output_dir, 'linear_fits', f'curve_{idx:03d}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        return flagged


    def plot_nonlinear_positions(self, nonlinear, ref_frame=20):
        """Plot the reference frame with the flagged stars highlighted (all other stars faint)."""
        image, _ = self._load_fits(self.fnames[ref_frame])
        ny, nx = image.shape
        r = getattr(self, 'default_aperture_radius', 10)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(apply_stretch(image), cmap='gray', origin='lower')
        flagged = set(nonlinear)
        for sid, (x, y) in enumerate(self.star_positions):
            if sid in flagged:
                ax.add_patch(plt.Circle((x, y), r + 4, color='orange', fill=False, lw=2))
                ax.text(x, y + r + 8, str(sid), color='orange', fontsize=8, ha='center', clip_on=True)
            else:
                blended = sid in self.blended_sids
                ax.add_patch(plt.Circle((x, y), r + 4, color='lightblue' if blended else 'white',
                                        fill=False, lw=0.5, alpha=0.8 if blended else 0.4))
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)
        ax.set_title(f'Flagged Stars (Frame {ref_frame}): {len(flagged)} of {len(self.star_positions)} '
                     f'(light blue = {len(self.blended_sids)} blended, skipped)')
        fig.savefig(os.path.join(self.output_dir, 'nonlinear_star_positions.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved nonlinear star position map with {len(nonlinear)} stars")

    # ── Full pipeline ─────────────────────────────────────────────

    def run(self, ref_frame=20, skip_extraction=False, method='sep',
             psf_sigma=None, fit_shape=11, max_ensemble_iterations=3, aperture_radius=None):
        """
        Run end-to-end: detect → extract → save → normalize → slopes → nonlinearity.
        Set skip_extraction=True to load previously saved light curves instead.
        method: 'sep' (default), 'aperture' or 'psf' — passed to extract_light_curves.
        aperture_radius / psf_sigma: None derives them from the HOPS 'PSF' header.

        Stars flagged as nonlinear/variable are iteratively excluded from
        acting as comparison stars in the ensemble normalization (up to
        ``max_ensemble_iterations`` passes, or until the flagged set stops
        changing), so a variable star can no longer contaminate the
        comparison flux used for other stars.
        """
        if skip_extraction:
            self.load_light_curves()
        else:
            self.detect_stars(ref_frame)
            self.extract_light_curves(ref_frame, method=method, aperture_radius=aperture_radius,
                                      psf_sigma=psf_sigma, fit_shape=fit_shape)
            self.save_light_curves()

        # Iteratively refine the comparison ensemble, excluding stars that
        # look variable so they no longer bias other stars' light curves.
        exclude_sids = set()
        nonlinear = []
        for iteration in range(max_ensemble_iterations):
            self.normalize(exclude_sids=exclude_sids)
            self.compute_slopes()
            new_nonlinear = set(i for i in range(len(self.rel_lcs))
                                 if self.detect_nonlinear(i, plot=False))
            print(f"Ensemble iteration {iteration}: {len(new_nonlinear)} candidate variable stars")
            if new_nonlinear == exclude_sids:
                break
            exclude_sids = new_nonlinear

        # Final pass with plotting enabled for the converged flagged set
        nonlinear = [i for i in range(len(self.rel_lcs))
                     if self.detect_nonlinear(i, plot=True)]

        self.plot_nonlinear_positions(nonlinear, getattr(self, 'ref_frame', ref_frame))


        # Overview plot of flagged curves
        if nonlinear:
            fig, ax = plt.subplots(figsize=(6, 4))
            for i in nonlinear:
                valid_mask = (self.rel_lcs[i] > 0) & (self.rel_lcs_err[i] > 0)
                normalized_flux = (self.rel_lcs[i]/np.nanmedian(self.rel_lcs[i]))[valid_mask]
                normalized_error = (self.rel_lcs_err[i]/np.nanmedian(self.rel_lcs[i]))[valid_mask]
                #ax.errorbar(self.time[valid_mask], normalized_flux, yerr=normalized_error,
                #            marker='o', label=f'Star {i}', alpha=0.5)
                ax.plot(self.time[valid_mask], normalized_flux, marker='o', label=f'Star {i}', alpha=0.5)
            ax.legend()
            ax.set_xlabel('Time (days)'); ax.set_ylabel('Normalized Flux')
            ax.set_ylim(0.5, 1.5)
            fig.savefig(os.path.join(self.output_dir, 'normalized_light_curves.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"\nDone — {len(nonlinear)} variable candidates (slope OR nonlinear OR autocorr): {nonlinear}")
        for i in nonlinear:
            print(f"  Star {i}: {', '.join(self.flag_reasons.get(i, []))}")
        print(f"Skipped {len(self.blended_sids)} blended stars: {sorted(self.blended_sids)}")
        return nonlinear
