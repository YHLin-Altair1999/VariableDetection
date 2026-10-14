"""Per-frame photometry of stars at known positions. All fluxes and errors are in ADU."""
from __future__ import annotations

import warnings

import numpy as np
import sep
from astropy.modeling.fitting import TRFLSQFitter
from astropy.table import Table
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.psf import CircularGaussianPRF, PSFPhotometry

from .config import PhotometryConfig


class Photometer:
    """Measures flux and error for stars at given positions in a frame."""

    def __init__(self, cfg: PhotometryConfig, gain: float):
        self.cfg = cfg
        self.gain = gain

    def radii(self, r: float) -> tuple[float, float, float]:
        """(aperture, annulus inner, annulus outer) radii."""
        r_in = r + self.cfg.annulus_gap
        return r, r_in, r_in + self.cfg.annulus_width

    def margin(self, r: float) -> float:
        """Distance from the frame edge a star needs so the whole annulus (or fit box) is on-frame."""
        m = self.radii(r)[2] + 1
        return max(m, self.cfg.psf_fit_shape // 2 + 1) if self.cfg.method == "psf" else m

    def measure(self, data: np.ndarray, pos: np.ndarray, r: float, fwhm: float
                ) -> tuple[np.ndarray, np.ndarray]:
        """Flux and error for every star (NaN where unmeasurable) using aperture radius ``r``
        and, for the PSF method, a Gaussian PRF of the frame's ``fwhm``."""
        n = len(pos)
        flux, err = np.full(n, np.nan), np.full(n, np.nan)
        ny, nx = data.shape
        m = self.margin(r)
        ok = (pos[:, 0] >= m) & (pos[:, 0] < nx - m) & (pos[:, 1] >= m) & (pos[:, 1] < ny - m)
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            return flux, err
        bkg = sep.Background(data)
        sub = data - bkg.back()
        radii = self.radii(r)
        if self.cfg.method == "sep":
            flux[idx], err[idx] = self._sep(sub, pos[idx], bkg.globalrms, radii)
        elif self.cfg.method == "aperture":
            flux[idx], err[idx] = self._aperture(sub, pos[idx], bkg.globalrms, radii)
        else:
            flux[idx], err[idx] = self._psf(sub, pos[idx], radii, fwhm)
        return flux, err

    @staticmethod
    def _sky(sub: np.ndarray, pos: np.ndarray, r_in: float, r_out: float):
        """Sigma-clipped median sky in the annulus around each star (robust against neighbours)."""
        k = int(np.ceil(r_out))
        dy, dx = np.mgrid[-k:k + 1, -k:k + 1]
        rr = np.hypot(dx, dy)
        ring = (rr >= r_in) & (rr <= r_out)
        dx, dy = dx[ring], dy[ring]
        vals = sub[np.rint(pos[:, 1]).astype(int)[:, None] + dy[None, :],
                   np.rint(pos[:, 0]).astype(int)[:, None] + dx[None, :]]
        med = np.median(vals, axis=1, keepdims=True)
        sig = 1.4826 * np.median(np.abs(vals - med), axis=1, keepdims=True)
        vals = np.where(np.abs(vals - med) <= 3.0 * np.maximum(sig, 1e-9), vals, np.nan)
        return np.nanmedian(vals, axis=1), np.sum(np.isfinite(vals), axis=1)

    def _sep(self, sub, pos, rms, radii):
        r, r_in, r_out = radii
        flux, err, flag = sep.sum_circle(sub, pos[:, 0], pos[:, 1], r, err=rms, gain=self.gain)
        sky, n_ann = self._sky(sub, pos, r_in, r_out)
        area = np.pi * r ** 2
        flux = flux - sky * area
        err = np.sqrt(err ** 2 + area ** 2 * rms ** 2 / np.maximum(n_ann, 1))
        bad = (flag != 0) | (n_ann < self.cfg.min_annulus_pixels)
        return np.where(bad, np.nan, flux), np.where(bad, np.nan, err)

    def _aperture(self, sub, pos, rms, radii):
        r, r_in, r_out = radii
        ap = CircularAperture(pos, r)
        an = CircularAnnulus(pos, r_in, r_out)
        raw = np.asarray(aperture_photometry(sub, ap)["aperture_sum"], float)
        sky = np.asarray(aperture_photometry(sub, an)["aperture_sum"], float) / an.area
        flux = raw - sky * ap.area
        var = np.maximum(flux, 0.0) / self.gain + ap.area * rms ** 2 * (1 + ap.area / an.area)
        return flux, np.sqrt(var)

    def _psf(self, sub, pos, radii, fwhm):
        shape = (self.cfg.psf_fit_shape, self.cfg.psf_fit_shape)
        psfphot = PSFPhotometry(CircularGaussianPRF(fwhm=max(fwhm, 1.5)), fit_shape=shape,
                                fitter=TRFLSQFitter(), aperture_radius=radii[0])
        init = Table({"x": pos[:, 0], "y": pos[:, 1]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = psfphot(sub, init_params=init)
            except Exception:
                return np.full(len(pos), np.nan), np.full(len(pos), np.nan)
        flux = np.asarray(res["flux_fit"], float)
        err = np.asarray(res["flux_err"], float) if "flux_err" in res.colnames else np.full(len(pos), np.nan)
        bad = np.asarray(res["flags"]) != 0 if "flags" in res.colnames else np.zeros(len(pos), bool)
        fallback = np.sqrt(np.maximum(flux, 0.0) / self.gain)
        err = np.where(np.isfinite(err) & (err > 0), err, fallback)
        return np.where(bad, np.nan, flux), np.where(bad, np.nan, err)
