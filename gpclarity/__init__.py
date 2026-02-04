"""
gpclarity: Interpretability Toolkit for Gaussian Processes
==============================================================

Extends GPy and emukit with human-readable diagnostics, uncertainty
analysis, and model interpretability tools. Designed for researchers
and practitioners who need trustworthy GP models.

Public API Philosophy
---------------------
This module exposes only high-level interpretability interfaces.
Internal utilities and helpers are intentionally not exposed to maintain
a clean, stable API surface for users.

Import Safety
-------------
Heavy scientific computing dependencies (GPy, emukit) are wrapped in
lazy loading mechanisms to enable fast imports and lightweight installs
for documentation or analysis-only workflows.

Quick Start
-----------
>>> from gpclarity import UncertaintyProfiler, summarize_kernel
>>> # Features unavailable without dependencies raise informative errors
"""

import warnings
from importlib.util import find_spec

# Version source of truth - keep this lightweight (no heavy imports)
from ._version import __version__

# Dependency availability flags (checked without importing)
_GPY_AVAILABLE = find_spec("GPy") is not None
_EMUKIT_AVAILABLE = find_spec("emukit") is not None

# Build __all__ dynamically based on availability
__all__ = ["__version__", "AVAILABLE"]

# Public availability flag for programmatic checks
AVAILABLE = {
    "gpy": _GPY_AVAILABLE,
    "emukit": _EMUKIT_AVAILABLE,
    "full": _GPY_AVAILABLE and _EMUKIT_AVAILABLE,
}


class _UnavailableFeature:
    """
    Stub that raises informative ImportError when accessed.
    Allows `from gpclarity import X` to succeed, but fails gracefully on usage.
    """
    
    def __init__(self, name: str, dependency: str, install_extra: str):
        self.name = name
        self.dependency = dependency
        self.install_extra = install_extra
    
    def __call__(self, *args, **kwargs):
        raise ImportError(
            f"{self.name} requires {self.dependency} which is not installed. "
            f"Install with: pip install gpclarity[{self.install_extra}]"
        )
    
    def __getattr__(self, attr):
        return self.__call__
    
    def __repr__(self):
        return f"<Unavailable: {self.name} (requires {self.dependency})>"


def __getattr__(name: str):
    """
    Lazy loader for heavy modules.
    Only imports GPy/emukit-dependent code when actually accessed.
    """
    # Kernel Summary
    if name in ("summarize_kernel", "format_kernel_tree", 
                "interpret_lengthscale", "interpret_variance"):
        if not _GPY_AVAILABLE:
            return _UnavailableFeature(name, "GPy", "full")
        from .kernel_summary import (
            summarize_kernel,
            format_kernel_tree,
            interpret_lengthscale,
            interpret_variance,
        )
        return locals()[name]
    
    # Uncertainty Analysis
    if name == "UncertaintyProfiler":
        if not _EMUKIT_AVAILABLE:
            return _UnavailableFeature(name, "emukit", "full")
        from .uncertainty_analysis import UncertaintyProfiler
        return UncertaintyProfiler
    
    # Hyperparameter Tracking
    if name == "HyperparameterTracker":
        if not _GPY_AVAILABLE:
            return _UnavailableFeature(name, "GPy", "full")
        from .hyperparam_tracker import HyperparameterTracker
        return HyperparameterTracker
    
    # Model Complexity
    if name in ("compute_complexity_score", "count_kernel_components",
                "compute_roughness_score", "compute_noise_ratio"):
        if not _GPY_AVAILABLE:
            return _UnavailableFeature(name, "GPy", "full")
        from .model_complexity import (
            compute_complexity_score,
            count_kernel_components,
            compute_roughness_score,
            compute_noise_ratio,
        )
        return locals()[name]
    
    # Data Influence
    if name == "DataInfluenceMap":
        if not _GPY_AVAILABLE:
            return _UnavailableFeature(name, "GPy", "full")
        from .data_influence import DataInfluenceMap
        return DataInfluenceMap
    
    # Utilities
    if name in ("check_model_health", "extract_kernel_params_flat",
                "get_lengthscale", "get_noise_variance"):
        if not _GPY_AVAILABLE:
            return _UnavailableFeature(name, "GPy", "full")
        from .utils import (
            check_model_health,
            extract_kernel_params_flat,
            get_lengthscale,
            get_noise_variance,
        )
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Enable tab completion for available features."""
    return sorted(__all__)


# Populate __all__ with potentially available features
__all__.extend([
    # Kernel Summary
    "summarize_kernel",
    "format_kernel_tree",
    "interpret_lengthscale",
    "interpret_variance",
    # Uncertainty Analysis
    "UncertaintyProfiler",
    # Hyperparameter Tracking
    "HyperparameterTracker",
    # Model Complexity
    "compute_complexity_score",
    "count_kernel_components",
    "compute_roughness_score",
    "compute_noise_ratio",
    # Data Influence
    "DataInfluenceMap",
    # Utilities
    "get_lengthscale",
    "get_noise_variance",
    "extract_kernel_params_flat",
    "check_model_health",
])


# Warn once if running in limited mode
if not AVAILABLE["full"]:
    missing = []
    if not _GPY_AVAILABLE:
        missing.append("GPy")
    if not _EMUKIT_AVAILABLE:
        missing.append("emukit")
    
    warnings.warn(
        f"gpclarity running in limited mode. Missing: {', '.join(missing)}. "
        f"Install with 'pip install gpclarity[full]' for complete functionality.",
        ImportWarning,
        stacklevel=2,
    )


# Package metadata
__author__ = "Angad Kumar"
__email__ = "angadkumar16ak@gmail.com"
__description__ = "Interpretability and Diagnostics for Gaussian Processes"