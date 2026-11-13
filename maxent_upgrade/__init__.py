"""Reweight a positive-weight sample using fixed-order moments.

    moments, config = load_moments("dy")
    result = upgrade(events, moments, config)

For a new process, define the observable maps in config and construct the same
moments in your fixed-order calculation. No process registration is required.
"""
from .constraints import (
    alpha_s, check_seam, chebyshev_moment, combine_moments, compute_fo_moments,
    matching_scale, profile_w, resolved_order,
)
from .released import load_moments, sigma_fid
from . import released
from .result import UpgradeResult
from .reweight import upgrade
from .uncertainty import (
    EnsembleResult, Variation, histogram_bands, upgrade_ensemble, upgrade_with_uncertainties,
)

__version__ = "1.1.0"

__all__ = [
    "upgrade", "load_moments", "sigma_fid", "released",
    "compute_fo_moments", "chebyshev_moment", "profile_w", "matching_scale",
    "check_seam", "resolved_order", "combine_moments", "UpgradeResult",
    "upgrade_with_uncertainties", "upgrade_ensemble", "Variation", "EnsembleResult", "histogram_bands",
    "upgrade_from_histograms", "moment_snr", "FOHist", "fo_from_dat", "DEFAULTS",
]


def __getattr__(name):
    # Keep old imports working without loading the histogram engine for normal use.
    if name in {"upgrade_from_histograms", "moment_snr", "FOHist", "fo_from_dat", "DEFAULTS"}:
        from . import legacy
        value = getattr(legacy, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
