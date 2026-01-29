"""
gptools: Interpretability Toolkit for Gaussian Processes
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
try/except blocks to enable lightweight installs for documentation
or analysis-only workflows.
"""

# Version source of truth - prevents circular imports in complex packages
from ._version import __version__

# Public API: High-level interpretability and diagnostic tools
# Internal utilities are intentionally not exposed here

try:
    # Core interpretability modules
    from .data_influence import DataInfluenceMap
    from .hyperparam_tracker import HyperparameterTracker
    from .kernel_summary import (
        format_kernel_tree,
        interpret_lengthscale,
        interpret_variance,
        summarize_kernel,
    )
    from .model_complexity import (
        compute_complexity_score,
        compute_noise_ratio,
        compute_roughness_score,
        count_kernel_components,
    )
    from .uncertainty_analysis import UncertaintyProfiler
    from .utils import (
        check_model_health,
        extract_kernel_params_flat,
        get_lengthscale,
        get_noise_variance,
    )

    # Flag indicating successful import of all features
    _FULL_IMPORT_SUCCESS = True

except ImportError as e:  # pragma: no cover
    # Graceful degradation for minimal installs
    import warnings

    _FULL_IMPORT_SUCCESS = False
    _IMPORT_ERROR = str(e)

    warnings.warn(
        f"gptools: Some features unavailable due to missing dependency: {e}\n"
        "Install with 'pip install gptools[full]' for complete functionality.",
        ImportWarning,
        stacklevel=2,
    )

    # Define stubs to prevent NameError
    summarize_kernel = None
    format_kernel_tree = None
    interpret_lengthscale = None
    interpret_variance = None
    UncertaintyProfiler = None
    HyperparameterTracker = None
    compute_complexity_score = None
    count_kernel_components = None
    compute_roughness_score = None
    compute_noise_ratio = None
    DataInfluenceMap = None
    get_lengthscale = None
    get_noise_variance = None
    extract_kernel_params_flat = None
    check_model_health = None


# Explicit public API declaration
__all__ = [
    # Version
    "__version__",
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
]

# Package metadata
__author__ = "Angad Kumar"
__email__ = "angadkumar16ak@gmail.com"
__description__ = "Interpretability and Diagnostics for Gaussian Processes"
