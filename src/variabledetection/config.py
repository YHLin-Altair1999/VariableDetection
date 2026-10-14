"""Configuration: TOML file -> validated dataclasses."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Optional, Union

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.default.toml")


@dataclass
class DataConfig:
    reduced_subdir: str = "reduced"
    file_pattern: str = "*.fits"
    output_subdir: str = "output_light_curves"
    time_keywords: list[str] = field(default_factory=lambda: ["JD", "MJD-OBS", "DATE-OBS"])
    exposure_keyword: str = "EXPTIME"


@dataclass
class CameraConfig:
    gain: float = 1.0  # e-/ADU

    def __post_init__(self):
        if self.gain <= 0:
            raise ValueError("camera.gain must be > 0")


@dataclass
class DetectionConfig:
    threshold_sigma: float = 5.0
    fwhm: Optional[float] = None
    edge_buffer: int = 50
    max_elongation: float = 2.0
    saturation_adu: Optional[float] = None


@dataclass
class RegistrationConfig:
    reference_frame: Union[int, str] = "auto"
    max_control_points: int = 50
    min_matches: int = 8
    max_scale_deviation: float = 0.02
    min_region_fraction: float = 0.8
    max_dropped_fraction: float = 0.2

    def __post_init__(self):
        if self.reference_frame != "auto" and not isinstance(self.reference_frame, int):
            raise ValueError('registration.reference_frame must be "auto" or an integer')
        if not 0 < self.min_region_fraction <= 1:
            raise ValueError("registration.min_region_fraction must be in (0, 1]")
        if not 0 <= self.max_dropped_fraction < 1:
            raise ValueError("registration.max_dropped_fraction must be in [0, 1)")


@dataclass
class PhotometryConfig:
    method: str = "sep"
    aperture_mode: str = "fixed"
    aperture_radius: Optional[float] = None
    aperture_fwhm_factor: float = 2.0
    aperture_min: float = 3.0
    aperture_max: float = 20.0
    annulus_gap: float = 2.0
    annulus_width: float = 6.0
    min_annulus_pixels: int = 20
    psf_fit_shape: int = 11

    def __post_init__(self):
        if self.method not in ("sep", "aperture", "psf"):
            raise ValueError("photometry.method must be 'sep', 'aperture' or 'psf'")
        if self.aperture_mode not in ("fixed", "per_frame"):
            raise ValueError("photometry.aperture_mode must be 'fixed' or 'per_frame'")
        if self.aperture_mode == "per_frame" and self.aperture_radius is not None:
            raise ValueError("photometry.aperture_radius cannot be combined with aperture_mode = 'per_frame'")


@dataclass
class PsfConfig:
    fwhm_method: str = "sep"          # "sep" (half-light radius) or "photutils" (Gaussian PRF fit)
    n_stars: int = 20
    fit_shape: int = 15
    max_fwhm_factor: Optional[float] = None  # drop frames with FWHM > factor * median FWHM

    def __post_init__(self):
        if self.fwhm_method not in ("sep", "photutils"):
            raise ValueError("psf.fwhm_method must be 'sep' or 'photutils'")
        if self.max_fwhm_factor is not None and self.max_fwhm_factor <= 1:
            raise ValueError("psf.max_fwhm_factor must be > 1")


@dataclass
class NormalizationConfig:
    brightness_window_dex: float = 0.5
    max_comparison_distance: Optional[float] = 300.0
    blend_radius_factor: float = 2.0
    ensemble_sigma_clip: float = 3.0
    max_ensemble_iterations: int = 3

    def __post_init__(self):
        if self.max_comparison_distance is not None and self.max_comparison_distance <= 0:
            self.max_comparison_distance = None


@dataclass
class VariabilityConfig:
    slope_sigma_threshold: float = 5.0
    f_threshold: float = 50.0
    autocorr_threshold: float = 5.0
    autocorr_max_lag: int = 5
    sigma_clip: float = 3.0
    min_points: int = 10
    plot_flagged: bool = True


@dataclass
class PlateSolveConfig:
    enabled: bool = False
    api_key_env: str = "ASTROMETRY_API_KEY"
    max_sources: int = 100
    solve_timeout: int = 300
    scale_lower: Optional[float] = None
    scale_upper: Optional[float] = None
    ra_hint: Optional[float] = None
    dec_hint: Optional[float] = None
    radius_hint: float = 1.0


@dataclass
class OutputConfig:
    frame_plots: int = 5
    dpi: int = 200


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    psf: PsfConfig = field(default_factory=PsfConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    variability: VariabilityConfig = field(default_factory=VariabilityConfig)
    platesolve: PlateSolveConfig = field(default_factory=PlateSolveConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _build(cls, raw: dict, path: str = ""):
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown config key(s) in [{path or 'root'}]: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            sub_cls = type(known[name].default_factory())
            if not is_dataclass(sub_cls):
                raise ValueError(f"Config key '{path}.{name}' must not be a table")
            kwargs[name] = _build(sub_cls, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load a TOML config; missing keys keep their defaults. ``None`` gives all defaults."""
    if path is None:
        return Config()
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return _build(Config, raw)
