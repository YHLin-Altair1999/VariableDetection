"""Automatic variable-star detection from calibrated FITS time series."""
from .config import DEFAULT_CONFIG_PATH, Config, load_config
from .io import discover_targets
from .pipeline import RunResult, TargetPipeline

__version__ = "0.1.0"
__all__ = ["Config", "DEFAULT_CONFIG_PATH", "RunResult", "TargetPipeline", "discover_targets", "load_config"]
