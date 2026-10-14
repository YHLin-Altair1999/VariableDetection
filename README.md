# VariableDetection

Automatic detection of variable-star candidates in a time series of calibrated FITS images.
No dependency on HOPS or any other reduction package: frames are registered, photometry is
done on a common star set, and each star is tested for variability.

## Pipeline

1. **Register** every frame to a reference frame with [astroalign](https://astroalign.quatrope.org/)
   (triangle pattern matching, tolerant of shifts, rotation and meridian flips).
2. **Common region**: the largest axis-aligned rectangle covered by *all* frames (minus an edge
   buffer). Only stars inside it are used, so every star exists in every frame. If a few badly
   shifted frames would shrink it too much, they are dropped (`[registration]` settings).
3. **PSF width and aperture.** The FWHM of every frame is measured (`[psf] fwhm_method`):
   `sep` uses the median half-light radius of the brightest stars (fast); `photutils` fits a
   Gaussian PRF to isolated bright stars (slower, more rigorous). With
   `photometry.aperture_mode = "fixed"` one aperture (`aperture_fwhm_factor` x median FWHM) is used
   for all frames; with `"per_frame"` the radius follows each frame's seeing. Frames much worse than
   the median can be dropped with `psf.max_fwhm_factor`.
4. **Photometry** (`sep`, `aperture` or `psf`) with photon-noise errors from the camera gain.
5. **Differential photometry** against an ensemble of similar-brightness, non-blended stars,
   iteratively excluding suspected variables from the ensemble.
6. **Tests** per star: slope outlier, linear-vs-quadratic F-test, residual autocorrelation.
7. **Plate solve** (optional) the reference frame via astrometry.net to get RA/Dec of flagged stars.

## Install

```
pip install -e .          # or: pip install -r requirements.txt
```

Python >= 3.11.

## Data layout

```
<root>/
  <target_name>/
    reduced/            # name configurable: [data] reduced_subdir
      *.fits            # pattern configurable: [data] file_pattern
```

Frame time is read from `JD`, `MJD-OBS` or `DATE-OBS` (+ half the exposure); see `[data] time_keywords`.
`discover_targets()` skips targets that already have an `output_light_curves` folder.

## Usage

Everything is driven from Python. Settings come from a `Config` object, either built in code or
loaded from a TOML file (`load_config("my.toml")`; all keys are optional, the full list with
defaults is `DEFAULT_CONFIG_PATH`, i.e. `src/variabledetection/config.default.toml`).

```python
from variabledetection import Config, TargetPipeline

cfg = Config()
cfg.camera.gain = 0.28            # e-/ADU: drives the photon-noise error bars
result = TargetPipeline("D:/obsdata/M31", cfg).run()
print(result.flagged, result.reasons)
```

Ready-to-edit scripts are in [examples/](examples):

| Script | Shows |
|--------|-------|
| `01_single_target.py` | one target, settings in code |
| `02_batch_with_config.py` | all new targets under a root, settings from `config.toml` |
| `03_seeing_aware_photometry.py` | per-frame PSF fit, per-frame aperture, poor-seeing rejection |
| `04_rerun_statistics.py` | new thresholds from saved light curves (`run(reuse=True)`) |
| `05_plate_solve_flagged.py` | RA/Dec of flagged stars via astrometry.net |
| `06_inspect_results.py` | plot a star's raw and normalised light curve |

Each script takes the data root as an optional first argument.

## Output (`<target>/output_light_curves/`)

| File | Content |
|------|---------|
| `light_curves.npz` | raw time, flux, error, star positions (reference frame) |
| `normalized_light_curves.npz` | differential light curves and errors |
| `star_statistics.csv`, `flagged_stars.csv` | per-star test statistics, flag reasons, RA/Dec if plate solved |
| `frames.csv` | per-frame status, registration transform, FWHM and aperture radius |
| `detected_stars.png`, `flagged_star_positions.png` | reference frame with stars and common region |
| `linear_fits/curve_NNN.png`, `normalized_light_curves.png` | plots of flagged stars |
| `reference_wcs.hdr` | WCS of the reference frame (plate solving only) |

## Plate solving

Set `platesolve.enabled = true` and export your [astrometry.net](https://nova.astrometry.net/)
API key in the environment variable named by `platesolve.api_key_env` (default `ASTROMETRY_API_KEY`).
Only the pixel positions of the brightest stars are uploaded, not the image. Pixel-scale and
position hints in the config speed up solving. If solving fails the photometry results are kept.

## Tests

```
pip install -e .[test]
pytest
```

The tests use synthetic frames; no real data is required.
